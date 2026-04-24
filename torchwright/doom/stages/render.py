"""RENDER stage: chunked column fill + state machine for autoregressive loop.

One RENDER token paints ``chunk_size`` vertical pixels of one screen
column.  The token carries four bounded integers: ``col`` (screen
column), ``chunk_k`` (vertical chunk index), ``wall_counter`` (sort
position), and ``wall_index`` (which wall).  Wall identity details
(tex_id, vis_lo, vis_hi) are read from the WALL position via the
geometry attention — not carried on the token.

Per-token flow:

1. **Wall geometry attention**: convert ``wall_index`` to a one-hot,
   attend to the matching WALL position, read (ax, ay, bx, by,
   tex_id, vis_lo, vis_hi).
2. **Precompute**: from raw geometry plus player state (position and
   cos/sin from PLAYER broadcasts), compute the rotation products
   (sort_den, C, D, E, sort_num_t) and derive H_inv.
3. The **active column** is ``col`` (SORTED sets this to ``vis_lo``
   for each new wall).
4. Compute wall height + texture u-coordinate.
5. TEX_COL attention fetches the matching texture column pixels.
6. Compute **active_start** from ``chunk_k``:
   ``vis_top + chunk_k × chunk_size``.
7. Fill ``chunk_size`` rows into the column pixel strip.
8. Compute state transitions — three exclusive cases:

   * **more chunks**: ``active_start + cs < vis_bottom`` → stay on this
     column, advance ``chunk_k`` by 1.
   * **advance col**: no more chunks, ``col + 1 ≤ vis_hi`` →
     move to the next column, reset ``chunk_k`` to 0.
   * **advance wall**: no more columns → if ``wall_counter`` equals
     ``max_walls``, emit ``done``.  Otherwise emit next-token-type =
     ``E8_SORTED_WALL`` so the transformer picks the next wall.

Next-token-type is ``E8_SORTED_WALL`` on wall transitions (not done),
``E8_RENDER`` otherwise.  The host just bitblits — no conditional logic.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    bool_to_01,
    clamp,
    compare,
    multiply_const,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    subtract,
    sum_nodes,
    thermometer_floor_div,
)
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_most_recent_matching,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true, bool_not, cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.embedding import D_EMBED, VALUE_RANGE_BY_NAME, embed_lookup
from torchwright.doom.graph_constants import (
    DIFF_BP,
    TEX_E8_OFFSET,
    TRIG_BP,
)
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.renderer import _textured_column_fill
from torchwright.doom.thinking_readback import ThinkingReadback

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class RenderToken:
    """Three bounded integers — the minimal autoregressive state.

    ``wall_index`` used to live here as a fourth overlaid integer, but
    Phase A's M3 step moved it into the KV cache: RENDER now reads the
    current wall index via ``attend_most_recent_matching`` against the
    most-recent ``E8_SORTED_WALL`` position rather than from an
    overlaid input slot.
    """

    col: Node  # current screen column (0..W)
    chunk_k: Node  # chunk index within current column (0..ceil(H/cs))
    wall_counter: Node  # how many walls sorted so far (0..max_walls)


@dataclass
class RenderKVInput:
    """Data at other positions read via attention."""

    # From WALL positions (attend_argmax_dot on wall_index one-hot).
    wall_ax: Node
    wall_ay: Node
    wall_bx: Node
    wall_by: Node
    wall_tex_id: Node  # host-fed at WALL positions
    wall_position_onehot: Node  # from WallKVOutput

    # From PLAYER broadcasts.
    player_x: Node  # resolved x
    player_y: Node  # resolved y
    player_cos: Node  # cos(θ)
    player_sin: Node  # sin(θ)

    # From TEX_COL positions (attend_argmax_dot on tex_id + col).
    texture_id_e8: Node  # host-fed at TEX_COL positions (8-wide)
    tex_pixels: Node  # host-fed at TEX_COL positions
    tc_onehot_01: Node  # from TexColKVOutput

    # Host-fed wall counter for RENDER's termination check
    # (``wall_counter >= max_walls`` → emit DONE).
    wall_counter: Node

    # Phase B Part 2: wall identity + visibility extent come from
    # thinking-phase VALUE tokens, not from prefill WALL or overlay.
    # ``readback`` decodes the most recent SORT_RESULT VALUE (for
    # wall_index) and the (wall-indexed) VIS_HI VALUE (for the
    # column-range upper bound).  ``embedding`` and
    # ``value_wall_index_onehot`` back the VIS_HI content attention's
    # key.  ``is_thinking_value`` gates the payload extraction.
    readback: "ThinkingReadback"
    embedding: Node
    value_wall_index_onehot: Node
    is_thinking_value: Node


@dataclass
class RenderTokenOutput:
    """Overlay + overflow outputs at RENDER positions."""

    # 72-wide next-token embedding (goes to _assemble_output as part
    # of next_token_embedding overflow): embed_lookup("SORTED_WALL")
    # on wall advance, embed_lookup("RENDER") otherwise.
    render_next_type: Node
    next_col: Node
    next_chunk_k: Node
    next_wall_counter: Node  # forwarded unchanged

    # Overflow (host reads).
    pixels: Node  # chunk_size * 3 floats
    active_col: Node
    active_start: Node
    chunk_length: Node
    done_flag: Node
    advance_wall: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_render(
    token: RenderToken,
    kv: RenderKVInput,
    *,
    is_render: Node,
    is_wall: Node,
    is_tex_col: Node,
    pos_encoding: PosEncoding,
    config: RenderConfig,
    textures: List[np.ndarray],
    chunk_size: int,
    max_coord: float,
    max_walls: int,
    tex_sample_batch_size: int = 8,
) -> RenderTokenOutput:
    H = config.screen_height
    W = config.screen_width
    fov = config.fov_columns
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]
    cs = chunk_size

    with annotate("render/wall_index_readback"):
        # Phase B Part 2: wall_index rides as a VALUE token emitted by
        # the SORT_RESULT identifier.  The host echoes that token's id
        # (VALUE_{wall_index}) back at the next position, where the
        # embedding lookup places the factored 4+4+4+4 payload in
        # cols [8:72] at layer 0.  RENDER reads it via
        # ``attend_most_recent_matching(is_SORT_RESULT_value)`` and
        # decodes through the readback Linear — no dependency on any
        # compute chain at the producing position.
        wall_index = kv.readback.get_value_after_last("SORT_RESULT")

    with annotate("render/wall_index_onehot"):
        wall_index_clamped = clamp(wall_index, 0.0, float(max_walls - 1))
        wall_index_p1 = add_const(wall_index_clamped, 1.0)
        wall_j_onehot = bool_to_01(
            in_range(wall_index_clamped, wall_index_p1, max_walls)
        )

    with annotate("render/wall_geom_attention"):
        sel_ax, sel_ay, sel_bx, sel_by, sel_tex_id = _attend_wall_geometry(
            pos_encoding,
            is_render=is_render,
            is_wall=is_wall,
            wall_j_onehot=wall_j_onehot,
            wall_position_onehot=kv.wall_position_onehot,
            wall_ax=kv.wall_ax,
            wall_ay=kv.wall_ay,
            wall_bx=kv.wall_bx,
            wall_by=kv.wall_by,
            wall_tex_id=kv.wall_tex_id,
        )

    with annotate("render/vis_hi_content_attention"):
        # Phase B Part 2: vis_hi comes from the thinking VIS_HI VALUE
        # token for this wall, not from the prefill WALL stage's FOV
        # clip.  Match against thinking VIS_HI VALUE positions keyed
        # by ``(identifier=VIS_HI, wall_index)`` and decode the
        # matched payload back to a scalar.
        sel_vis_hi = _content_attend_thinking_value(
            readback=kv.readback,
            embedding=kv.embedding,
            value_wall_index_onehot=kv.value_wall_index_onehot,
            query_wall_onehot=wall_j_onehot,
            consumer_gate=is_render,
            name="VIS_HI",
            pos_encoding=pos_encoding,
        )

    with annotate("render/precompute"):
        sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv = _compute_precomputes(
            sel_ax,
            sel_ay,
            sel_bx,
            sel_by,
            kv.player_x,
            kv.player_y,
            kv.player_cos,
            kv.player_sin,
            H=H,
            max_coord=max_coord,
        )

    with annotate("render/state_machine"):
        active_col = token.col
        angle_offset = _compute_angle_offset(active_col, W=W, fov=fov)

    with annotate("render/wall_height"):
        tan_o, tan_val_bp = _compute_angle_offset_tan(angle_offset, fov=fov)
        den_over_cos, abs_den_over_cos = _compute_den_over_cos(
            sort_den,
            precomp_C,
            tan_o,
            tan_val_bp,
        )
        wall_top, wall_bottom, wall_height = _compute_wall_height(
            precomp_H_inv,
            abs_den_over_cos,
            H=H,
            max_coord=max_coord,
        )

    with annotate("render/tex_coord"):
        tex_col_idx = _compute_texture_column(
            precomp_D,
            precomp_E,
            tan_o,
            tan_val_bp,
            abs_den_over_cos,
            max_coord=max_coord,
            tex_w=tex_w,
        )

    with annotate("render/tex_attention"):
        tex_column_colors = _attend_to_texture_column(
            pos_encoding,
            is_render=is_render,
            is_tex_col=is_tex_col,
            fb_tex_id=sel_tex_id,
            tex_col_idx=tex_col_idx,
            tc_onehot_01=kv.tc_onehot_01,
            texture_id_e8=kv.texture_id_e8,
            tex_pixels=kv.tex_pixels,
            num_tex=len(textures),
            tex_w=tex_w,
        )

    with annotate("render/column_fill"):
        active_start, chunk_length, pixels = _chunk_fill(
            wall_top,
            wall_bottom,
            wall_height,
            tex_column_colors,
            render_chunk_k=token.chunk_k,
            config=config,
            tex_h=tex_h,
            chunk_size=cs,
            max_coord=max_coord,
            tex_sample_batch_size=tex_sample_batch_size,
        )

    with annotate("render/state_transitions"):
        (
            done_flag,
            render_next_type,
            next_col,
            next_chunk_k,
            advance_wall,
        ) = _compute_next_state(
            active_col=active_col,
            active_start=active_start,
            wall_bottom_clamped=clamp(wall_bottom, 0.0, float(H)),
            vis_hi=sel_vis_hi,
            wall_counter=token.wall_counter,
            chunk_k=token.chunk_k,
            chunk_size=cs,
            max_walls=max_walls,
        )

    return RenderTokenOutput(
        render_next_type=render_next_type,
        next_col=next_col,
        next_chunk_k=next_chunk_k,
        next_wall_counter=token.wall_counter,
        pixels=pixels,
        active_col=active_col,
        active_start=active_start,
        chunk_length=chunk_length,
        done_flag=done_flag,
        advance_wall=advance_wall,
    )


# ---------------------------------------------------------------------------
# Wall geometry attention
# ---------------------------------------------------------------------------


def _attend_wall_geometry(
    pos_encoding: PosEncoding,
    *,
    is_render: Node,
    is_wall: Node,
    wall_j_onehot: Node,
    wall_position_onehot: Node,
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    wall_tex_id: Node,
) -> tuple[Node, Node, Node, Node, Node]:
    """Read (ax, ay, bx, by, tex_id) from the WALL position matching
    wall_j_onehot.  All values are host-fed (available at layer 0), so this
    attention can fire early — no dependency on WALL stage computation.
    """
    _GEOM_WIDTH = 5
    GEOM_MATCH_GAIN = 1000.0
    wall_geom = attend_argmax_dot(
        query_vector=cond_gate(is_render, wall_j_onehot),
        key_vector=cond_gate(is_wall, wall_position_onehot),
        value=cond_gate(
            is_wall,
            Concatenate([wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id]),
        ),
        match_gain=GEOM_MATCH_GAIN,
        assert_hardness_gt=0.99,
    )
    sel_ax = extract_from(wall_geom, _GEOM_WIDTH, 0, 1, "rsel_ax")
    sel_ay = extract_from(wall_geom, _GEOM_WIDTH, 1, 1, "rsel_ay")
    sel_bx = extract_from(wall_geom, _GEOM_WIDTH, 2, 1, "rsel_bx")
    sel_by = extract_from(wall_geom, _GEOM_WIDTH, 3, 1, "rsel_by")
    sel_tex_id = extract_from(wall_geom, _GEOM_WIDTH, 4, 1, "rsel_tex_id")
    return sel_ax, sel_ay, sel_bx, sel_by, sel_tex_id


def _content_attend_thinking_value(
    *,
    readback: ThinkingReadback,
    embedding: Node,
    value_wall_index_onehot: Node,
    query_wall_onehot: Node,
    consumer_gate: Node,
    name: str,
    pos_encoding: PosEncoding,
) -> Node:
    """Read the per-wall ``name``-VALUE payload via content attention
    keyed by ``(identifier=name, wall_index)``.

    Mirrors the vis_lo lookup in ``stages/sorted.py``: the query at the
    consumer position (gated by ``consumer_gate``) is
    ``[1, query_wall_onehot]``; the key at thinking ``name``-VALUE
    positions is ``[is_name_value_01, value_wall_index_onehot_gated]``.
    The matching VALUE's 64-wide payload is extracted and decoded back
    to a scalar float via the dequantize affine in
    ``VALUE_RANGE_BY_NAME``.
    """
    # Key-side type indicator from the shared readback handle.
    is_name_value = readback.is_value_of(name)

    key_type = bool_to_01(is_name_value)
    key_wall = cond_gate(is_name_value, value_wall_index_onehot)
    composite_key = Concatenate([key_type, key_wall])

    one_literal = create_literal_value(
        torch.tensor([1.0]), name=f"render_{name.lower()}_q_one"
    )
    query_raw = Concatenate([one_literal, query_wall_onehot])
    query_gated = cond_gate(consumer_gate, query_raw)

    payload = extract_from(
        embedding, D_EMBED, 8, D_EMBED - 8, f"render_{name.lower()}_payload"
    )
    gated_payload = cond_gate(is_name_value, payload)

    # Content match-gain sized for ~1000-position causal windows.  Same
    # regime as the thinking-phase prev-id attention (12000 for
    # 20-wide slot keys).  Dot on match is 2 (one type + one
    # wall-index match); on non-match ≤ 1.  ``12000·(2-1)`` = 12000
    # logit gap, which resolves softmax to ≥ 0.999 concentration.
    matched_payload = attend_most_recent_matching(
        pos_encoding=pos_encoding,
        query_vector=query_gated,
        key_vector=composite_key,
        value=gated_payload,
        match_gain=12000.0,
    )

    # Decode 4+4+4+4 one-hots → dequantized float.
    from torchwright.ops.quantization import DEFAULT_N_LEVELS
    from torchwright.graph.asserts import assert_in_range

    lo, hi = VALUE_RANGE_BY_NAME[name]
    inv_scale = (hi - lo) / (DEFAULT_N_LEVELS - 1)
    _hex_block = 16
    weights = torch.zeros(D_EMBED - 8, 1)
    for i in range(_hex_block):
        weights[0 * _hex_block + i, 0] = i * 4096.0
        weights[1 * _hex_block + i, 0] = i * 256.0
        weights[2 * _hex_block + i, 0] = i * 16.0
        weights[3 * _hex_block + i, 0] = float(i)
    weights = weights * inv_scale
    bias = torch.tensor([lo])
    decoded = Linear(
        matched_payload, weights, bias, name=f"render_decode_{name.lower()}"
    )
    return assert_in_range(decoded, lo, hi)


# ---------------------------------------------------------------------------
# Precompute from raw geometry + player state
# ---------------------------------------------------------------------------


def _compute_precomputes(
    sel_ax: Node,
    sel_ay: Node,
    sel_bx: Node,
    sel_by: Node,
    player_x: Node,
    player_y: Node,
    player_cos: Node,
    player_sin: Node,
    *,
    H: int,
    max_coord: float,
) -> tuple[Node, Node, Node, Node, Node]:
    """Compute (sort_den, C, D, E, H_inv) from raw wall geometry + player state.

    Same math as the former WALL-stage ``_compute_render_precomputation``
    and ``_compute_central_ray_intersection``, now computed at RENDER time.
    """
    w_ex = subtract(sel_bx, sel_ax)
    w_ey = subtract(sel_by, sel_ay)
    w_fx = subtract(sel_ax, player_x)
    w_gy = subtract(player_y, sel_ay)

    # sort_den = ey*cos - ex*sin
    r_ey_cos = piecewise_linear_2d(
        w_ey,
        player_cos,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_ey_cos",
    )
    r_ex_sin = piecewise_linear_2d(
        w_ex,
        player_sin,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_ex_sin",
    )
    sort_den = subtract(r_ey_cos, r_ex_sin)

    # C = ey*sin + ex*cos
    r_ey_sin = piecewise_linear_2d(
        w_ey,
        player_sin,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_ey_sin",
    )
    r_ex_cos = piecewise_linear_2d(
        w_ex,
        player_cos,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_ex_cos",
    )
    precomp_C = add(r_ey_sin, r_ex_cos)

    # D = fx*sin + gy*cos
    r_fx_sin = piecewise_linear_2d(
        w_fx,
        player_sin,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_fx_sin",
    )
    r_gy_cos = piecewise_linear_2d(
        w_gy,
        player_cos,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_gy_cos",
    )
    precomp_D = add(r_fx_sin, r_gy_cos)

    # E = fx*cos - gy*sin
    r_fx_cos = piecewise_linear_2d(
        w_fx,
        player_cos,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_fx_cos",
    )
    r_gy_sin = piecewise_linear_2d(
        w_gy,
        player_sin,
        DIFF_BP,
        TRIG_BP,
        lambda a, b: a * b,
        name="r_gy_sin",
    )
    precomp_E = subtract(r_fx_cos, r_gy_sin)

    # sort_num_t = ey*fx + ex*gy
    r_ey_fx = piecewise_linear_2d(
        w_ey,
        w_fx,
        DIFF_BP,
        DIFF_BP,
        lambda a, b: a * b,
        name="r_ey_fx",
    )
    r_ex_gy = piecewise_linear_2d(
        w_ex,
        w_gy,
        DIFF_BP,
        DIFF_BP,
        lambda a, b: a * b,
        name="r_ex_gy",
    )
    sort_num_t = add(r_ey_fx, r_ex_gy)

    # H_inv = H / |sort_num_t|
    abs_num_t = abs(sort_num_t)
    inv_abs_num_t = reciprocal(
        abs_num_t,
        min_value=0.3,
        max_value=2.0 * max_coord * max_coord,
        step=1.0,
    )
    precomp_H_inv = multiply_const(inv_abs_num_t, float(H))

    return sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv


# ---------------------------------------------------------------------------
# Column features
# ---------------------------------------------------------------------------


def _compute_angle_offset(active_col: Node, *, W: int, fov: int) -> Node:
    """Horizontal angle offset of ``active_col`` from the screen center (units: trig-table steps).

    ``angle_offset = (active_col * fov / W) - fov/2``, implemented via
    ``thermometer_floor_div`` so the result stays in the
    piecewise-linear domain downstream uses.
    """
    col_times_fov = multiply_const(active_col, float(fov))
    ao_raw = thermometer_floor_div(col_times_fov, W, fov * (W - 1))
    return add_const(ao_raw, float(-(fov // 2)))


def _compute_angle_offset_tan(angle_offset: Node, *, fov: int):
    """``tan(angle_offset)`` lookup, plus breakpoints for downstream 2D product grids."""
    half_fov = fov // 2
    tan_bp = [float(i) for i in range(-half_fov, half_fov + 1)]
    tan_o = piecewise_linear(
        angle_offset,
        tan_bp,
        lambda x: math.tan(x * 2.0 * math.pi / 256.0),
        name="tan_offset",
    )
    max_tan = math.tan(half_fov * 2.0 * math.pi / 256.0) * 1.1
    tan_val_bp = [-max_tan + i * (2 * max_tan / 10) for i in range(11)]
    return tan_o, tan_val_bp


def _compute_den_over_cos(
    sort_den: Node,
    precomp_C: Node,
    tan_o: Node,
    tan_val_bp,
):
    """``den/cos = sort_den - C*tan(offset)`` — per-column horizontal projection factor."""
    C_tan = piecewise_linear_2d(
        precomp_C,
        tan_o,
        DIFF_BP,
        tan_val_bp,
        lambda a, b: a * b,
        name="C_tan_o",
    )
    den_over_cos = subtract(sort_den, C_tan)
    abs_den_over_cos = abs(den_over_cos)
    return den_over_cos, abs_den_over_cos


# ---------------------------------------------------------------------------
# Wall height + texture coord
# ---------------------------------------------------------------------------


def _compute_wall_height(
    precomp_H_inv: Node,
    abs_den_over_cos: Node,
    *,
    H: int,
    max_coord: float,
):
    """Wall height = H_inv * |den/cos|, clamped.  Wall span is centered on H/2.

    Uses a log-spaced breakpoint grid for ``H_inv`` (values span ``0.01`` to
    ``H/0.3``) because the division step produces a highly non-uniform
    distribution.
    """
    max_h_inv = float(H) / 0.3
    h_inv_n = 16
    h_inv_ratio = (max_h_inv / 0.01) ** (1.0 / (h_inv_n - 1))
    height_inv_bp = [0.01 * (h_inv_ratio**k) for k in range(h_inv_n)]
    height_inv_bp[0] = 0.0
    height_inv_bp[-1] = max_h_inv

    doc_max = 2.5 * max_coord
    doc_bp = [doc_max * i / 15 for i in range(16)]

    wall_height_raw = piecewise_linear_2d(
        precomp_H_inv,
        abs_den_over_cos,
        height_inv_bp,
        doc_bp,
        lambda a, b: a * b,
        name="wall_height_raw",
    )
    wall_height = clamp(wall_height_raw, 0.0, float(H))

    center = float(H) / 2.0
    half_height = multiply_const(wall_height, 0.5)
    wall_top = Linear(
        half_height,
        torch.tensor([[-1.0]]),
        torch.tensor([center]),
        name="wall_top",
    )
    wall_bottom = Linear(
        half_height,
        torch.tensor([[1.0]]),
        torch.tensor([center]),
        name="wall_bottom",
    )
    return wall_top, wall_bottom, wall_height


def _compute_texture_column(
    precomp_D: Node,
    precomp_E: Node,
    tan_o: Node,
    tan_val_bp,
    abs_den_over_cos: Node,
    *,
    max_coord: float,
    tex_w: int,
) -> Node:
    """Texture column index via thermometer comparison (no division).

    Instead of computing ``u = abs_nuc / abs_den`` — which requires a
    piecewise-linear approximation of division whose error at the critical
    boundary ``u = 0.5`` can flip the tex_col — we determine tex_col as
    a count:

        tex_col = |{k ∈ 1..tex_w-1 : tex_w · abs_nuc ≥ k · abs_den}|

    Both ``tex_w · abs_nuc`` and ``k · abs_den`` are exact linear scalings
    (``multiply_const``), so the subtraction carries no approximation error
    beyond what is already in ``abs_nuc`` and ``abs_den_over_cos``.

    Threshold ``−0.5`` instead of 0: the exact boundary (diff = 0,
    i.e. u = k/tex_w) counts as TRUE, matching ``floor_int``'s behavior of
    returning ``k`` when its input equals ``k`` exactly.  For the box room
    (tex_w=8, |den|=10) the minimum margin per comparison is ≈ 0.48 —
    far above the comparison transition half-width of 0.05.
    """
    E_tan = piecewise_linear_2d(
        precomp_E,
        tan_o,
        DIFF_BP,
        tan_val_bp,
        lambda a, b: a * b,
        name="E_tan_o",
    )
    num_u_over_cos = add(precomp_D, E_tan)
    abs_nuc = abs(num_u_over_cos)

    # Scale abs_nuc by tex_w once (exact).
    nuc_scaled = multiply_const(abs_nuc, float(tex_w))

    # Threshold slightly below 0 so exact-boundary case counts as TRUE.
    _THRESH = -0.5
    bits = []
    for k in range(1, tex_w):
        k_den = multiply_const(abs_den_over_cos, float(k))
        diff = subtract(nuc_scaled, k_den)
        bits.append(bool_to_01(compare(diff, _THRESH)))

    return sum_nodes(bits)


def _attend_to_texture_column(
    pos_encoding: PosEncoding,
    *,
    is_render: Node,
    is_tex_col: Node,
    fb_tex_id: Node,
    tex_col_idx: Node,
    tc_onehot_01: Node,
    texture_id_e8: Node,
    tex_pixels: Node,
    num_tex: int,
    tex_w: int,
) -> Node:
    """Argmax-dot attention: RENDER token's (tex_id, col) → TEX_COL token's pixels."""
    tex_e8_query = piecewise_linear(
        fb_tex_id,
        [float(i) for i in range(num_tex)],
        lambda tid: [
            float(v) for v in index_to_vector(int(round(tid)) + TEX_E8_OFFSET)
        ],
        name="tex_id_to_e8",
    )
    tex_col_p1 = add_const(tex_col_idx, 1.0)
    rc_onehot_01 = bool_to_01(in_range(tex_col_idx, tex_col_p1, tex_w))

    COL_SCALE = 10.0
    TEX_MATCH_GAIN = 1000.0
    scaled_rc = multiply_const(rc_onehot_01, COL_SCALE)
    scaled_tc = multiply_const(tc_onehot_01, COL_SCALE)
    return attend_argmax_dot(
        query_vector=cond_gate(is_render, Concatenate([tex_e8_query, scaled_rc])),
        key_vector=cond_gate(is_tex_col, Concatenate([texture_id_e8, scaled_tc])),
        value=cond_gate(is_tex_col, tex_pixels),
        match_gain=TEX_MATCH_GAIN,
        assert_hardness_gt=0.99,
    )


# ---------------------------------------------------------------------------
# Chunk fill + state transitions
# ---------------------------------------------------------------------------


def _chunk_fill(
    wall_top: Node,
    wall_bottom: Node,
    wall_height: Node,
    tex_column_colors: Node,
    *,
    render_chunk_k: Node,
    config: RenderConfig,
    tex_h: int,
    chunk_size: int,
    max_coord: float,
    tex_sample_batch_size: int = 8,
):
    """Determine active_start, chunk_length, and paint the chunk's pixels."""
    H = config.screen_height
    vis_top_render = clamp(wall_top, 0.0, float(H))
    vis_bottom_render = clamp(wall_bottom, 0.0, float(H))

    active_start = add(
        vis_top_render, multiply_const(render_chunk_k, float(chunk_size))
    )

    chunk_length = clamp(
        subtract(vis_bottom_render, active_start),
        0.0,
        float(chunk_size),
    )

    pixels = _textured_column_fill(
        wall_top,
        wall_bottom,
        wall_height,
        tex_column_colors,
        tex_h,
        config,
        max_coord=max_coord,
        patch_row_start=active_start,
        rows_per_patch=chunk_size,
        tex_sample_batch_size=tex_sample_batch_size,
    )
    return active_start, chunk_length, pixels


def _compute_next_state(
    *,
    active_col: Node,
    active_start: Node,
    wall_bottom_clamped: Node,
    vis_hi: Node,
    wall_counter: Node,
    chunk_k: Node,
    chunk_size: int,
    max_walls: int,
):
    """Three-way state transition: more chunks / advance col / advance wall.

    On advance_wall (and not done), next token type is SORTED_WALL so the
    transformer picks the next wall.  Otherwise next token is RENDER.
    """
    next_chunk_start_val = add_const(active_start, float(chunk_size))
    has_more_chunks = compare(
        subtract(wall_bottom_clamped, next_chunk_start_val),
        0.5,
    )

    col_p1 = add_const(active_col, 1.0)
    not_more_chunks = bool_not(has_more_chunks)
    has_more_cols = compare(subtract(vis_hi, col_p1), 0.5)
    advance_col = bool_all_true([not_more_chunks, has_more_cols])
    advance_wall = bool_all_true([not_more_chunks, bool_not(has_more_cols)])

    all_walls_done = compare(wall_counter, float(max_walls) - 0.5)
    done_flag = bool_all_true([advance_wall, all_walls_done])

    zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
    next_col_output = select(
        has_more_chunks,
        active_col,
        select(advance_col, col_p1, zero_col),
    )
    chunk_k_plus_1 = add_const(chunk_k, 1.0)
    zero_chunk_k = create_literal_value(torch.tensor([0.0]), name="zero_chunk_k")
    next_chunk_k = select(has_more_chunks, chunk_k_plus_1, zero_chunk_k)

    type_render = create_literal_value(embed_lookup("RENDER"), name="type_render")
    type_sorted = create_literal_value(
        embed_lookup("SORTED_WALL"), name="type_sorted_wall"
    )
    advance_not_done = bool_all_true([advance_wall, bool_not(all_walls_done)])
    # approximate=False: both branches are fixed E8 literals with
    # magnitude 30.  In approximate mode, cond drift ε on the order of
    # 1e-3 gets amplified by M=30 into per-component output drift ~0.03.
    # That drift feeds back into the next step's input token_type,
    # pushing d=inp@E8_RENDER-800 in equals_vector into its
    # [-1, 0] transition zone, which then bleeds across is_render /
    # is_sorted flags.  Over several hundred RENDER steps this compounds
    # into the off_center[3,2,20] hang.  The non-approximate mode is
    # float-exact on the winning branch and immune to cond noise, at the
    # cost of one extra sublayer in this one op.
    render_next_type = select(
        advance_not_done, type_sorted, type_render, approximate=False
    )

    return (
        done_flag,
        render_next_type,
        next_col_output,
        next_chunk_k,
        advance_wall,
    )
