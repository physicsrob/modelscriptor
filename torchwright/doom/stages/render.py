"""RENDER stage: chunked column fill + state machine for autoregressive loop.

One RENDER token paints ``chunk_size`` vertical pixels of one screen
column.  The wall being rendered is *not* picked here — it was chosen by
the preceding THINKING token and its parameters arrive via the
``render_feedback`` overlay (``fb_sort_den, fb_C, ...``).

Per-token flow:

1. Derive the **active column**: if ``render_is_new_wall`` the column
   resets to the wall's ``fb_col_lo``; otherwise it's ``render_col``.
2. Compute wall height + texture u-coordinate using the feedback wall
   data and the per-column angle offset.
3. TEX_COL attention fetches the matching texture column pixels.
4. Determine the **active chunk_start**: sentinel (-1) in the feedback
   means "start at the wall's visible top"; otherwise continue from
   ``render_chunk_start``.
5. Fill ``chunk_size`` rows starting at ``active_start`` into the
   column pixel strip.
6. Compute state transitions — three exclusive cases:

   * **more chunks**: ``active_start + cs < vis_bottom`` → stay on this
     column, advance chunk_start by ``cs``.
   * **advance col**: no more chunks, ``active_col + 1 ≤ fb_col_hi`` →
     move to the next column, reset chunk_start sentinel.
   * **advance wall**: no more chunks, no more columns → update
     ``render_mask`` with ``fb_onehot``.  If all walls are masked, emit
     ``done``.

Next-token-type logic (``E8_THINKING`` on advance_wall, else
``E8_RENDER``) is exposed separately so the orchestrator can compose
it with the type selects for the other stages.
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
    subtract,
    sum_nodes,
    thermometer_floor_div,
)
from torchwright.ops.attention_ops import attend_argmax_dot
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true, bool_not, cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import (
    DIFF_BP,
    E8_RENDER,
    E8_THINKING,
    TEX_E8_OFFSET,
)
from torchwright.doom.renderer import _textured_column_fill

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class RenderInputs:
    """Per-RENDER inputs.

    Most fields come from ``render_feedback`` (an overlaid output).  The
    orchestrator extracts each field from the packed feedback vector
    before handing it here, so stage code stays flat.
    """

    # Iteration state (overlay).
    render_mask: Node  # max_walls-wide, walls fully rendered so far
    render_col: Node  # current column
    render_is_new_wall: Node  # +1 if just transitioned to a new wall
    render_chunk_start: Node  # current chunk start row; sentinel -1 = "new col"

    # Wall data (overlay; populated by THINKING, forwarded by prior RENDER).
    fb_sort_den: Node
    fb_C: Node
    fb_D: Node
    fb_E: Node
    fb_H_inv: Node
    fb_tex_id: Node
    fb_col_lo: Node
    fb_col_hi: Node
    fb_onehot: Node  # max_walls-wide

    # TEX_COL inputs (host-fed at TEX_COL positions).
    texture_id_e8: Node  # 8-wide
    tex_pixels: Node

    # TEX_COL stage output (per-TEX_COL-position one-hot).
    tc_onehot_01: Node

    # Token-type flags.
    is_render: Node
    is_tex_col: Node

    pos_encoding: PosEncoding


@dataclass
class RenderOutputs:
    """Outputs at RENDER positions.

    * ``pixels`` / ``active_col`` / ``active_start`` / ``chunk_length``
      land in overflow (host bitblits them to the framebuffer).
    * ``next_render_feedback`` feeds back through the render_feedback
      overlay (5-wide precomputes, forwarded unchanged).
    * Discrete state (mask, col, chunk, tex_id, vis bounds, onehot)
      are separate overlaid outputs copied by the host.
    * ``render_next_type`` is the next token type at RENDER positions:
      ``E8_THINKING`` on advance_wall, else ``E8_RENDER``.
    * ``done_flag`` signals to the host that all walls are fully
      rendered (it can stop feeding RENDER tokens).
    """

    pixels: Node  # chunk_size * 3 floats
    active_col: Node
    active_start: Node
    chunk_length: Node
    done_flag: Node
    next_render_feedback: Node  # 5-wide precomputes
    render_next_type: Node  # 8-wide: E8_THINKING or E8_RENDER
    # Discrete state outputs (overlaid separately from render_feedback).
    next_render_mask: Node
    next_col: Node
    next_is_new_wall: Node
    next_chunk: Node
    next_tex_id: Node
    next_vis_lo: Node
    next_vis_hi: Node
    next_wall_j_onehot: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_render(
    inputs: RenderInputs,
    config: RenderConfig,
    textures: List[np.ndarray],
    chunk_size: int,
    max_coord: float,
    max_walls: int,
    tex_sample_batch_size: int = 8,
) -> RenderOutputs:
    H = config.screen_height
    W = config.screen_width
    fov = config.fov_columns
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]
    cs = chunk_size

    with annotate("render/state_machine"):
        active_col = select(
            inputs.render_is_new_wall, inputs.fb_col_lo, inputs.render_col
        )
        angle_offset = _compute_angle_offset(active_col, W=W, fov=fov)

    with annotate("render/wall_height"):
        tan_o, tan_val_bp = _compute_angle_offset_tan(angle_offset, fov=fov)
        den_over_cos, abs_den_over_cos = _compute_den_over_cos(
            inputs.fb_sort_den,
            inputs.fb_C,
            tan_o,
            tan_val_bp,
        )
        wall_top, wall_bottom, wall_height = _compute_wall_height(
            inputs.fb_H_inv,
            abs_den_over_cos,
            H=H,
            max_coord=max_coord,
        )

    with annotate("render/tex_coord"):
        tex_col_idx = _compute_texture_column(
            inputs.fb_D,
            inputs.fb_E,
            tan_o,
            tan_val_bp,
            abs_den_over_cos,
            max_coord=max_coord,
            tex_w=tex_w,
        )

    with annotate("render/tex_attention"):
        tex_column_colors = _attend_to_texture_column(
            inputs.pos_encoding,
            is_render=inputs.is_render,
            is_tex_col=inputs.is_tex_col,
            fb_tex_id=inputs.fb_tex_id,
            tex_col_idx=tex_col_idx,
            tc_onehot_01=inputs.tc_onehot_01,
            texture_id_e8=inputs.texture_id_e8,
            tex_pixels=inputs.tex_pixels,
            num_tex=len(textures),
            tex_w=tex_w,
        )

    with annotate("render/column_fill"):
        active_start, chunk_length, pixels = _chunk_fill(
            wall_top,
            wall_bottom,
            wall_height,
            tex_column_colors,
            render_chunk_start=inputs.render_chunk_start,
            config=config,
            tex_h=tex_h,
            chunk_size=cs,
            max_coord=max_coord,
            tex_sample_batch_size=tex_sample_batch_size,
        )

    with annotate("render/state_transitions"):
        (
            next_render_feedback,
            done_flag,
            render_next_type,
            next_render_mask,
            next_col,
            next_is_new_wall,
            next_chunk,
            next_tex_id,
            next_vis_lo,
            next_vis_hi,
            next_wall_j_onehot,
        ) = _compute_next_state(
            active_col=active_col,
            active_start=active_start,
            wall_bottom_clamped=clamp(wall_bottom, 0.0, float(H)),
            render_mask=inputs.render_mask,
            fb_onehot=inputs.fb_onehot,
            fb_col_hi=inputs.fb_col_hi,
            fb_sort_den=inputs.fb_sort_den,
            fb_C=inputs.fb_C,
            fb_D=inputs.fb_D,
            fb_E=inputs.fb_E,
            fb_H_inv=inputs.fb_H_inv,
            fb_tex_id=inputs.fb_tex_id,
            fb_col_lo=inputs.fb_col_lo,
            chunk_size=cs,
            max_walls=max_walls,
        )

    return RenderOutputs(
        pixels=pixels,
        active_col=active_col,
        active_start=active_start,
        chunk_length=chunk_length,
        done_flag=done_flag,
        next_render_feedback=next_render_feedback,
        render_next_type=render_next_type,
        next_render_mask=next_render_mask,
        next_col=next_col,
        next_is_new_wall=next_is_new_wall,
        next_chunk=next_chunk,
        next_tex_id=next_tex_id,
        next_vis_lo=next_vis_lo,
        next_vis_hi=next_vis_hi,
        next_wall_j_onehot=next_wall_j_onehot,
    )


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
    fb_sort_den: Node,
    fb_C: Node,
    tan_o: Node,
    tan_val_bp,
):
    """``den/cos = sort_den - C*tan(offset)`` — per-column horizontal projection factor."""
    C_tan = piecewise_linear_2d(
        fb_C,
        tan_o,
        DIFF_BP,
        tan_val_bp,
        lambda a, b: a * b,
        name="C_tan_o",
    )
    den_over_cos = subtract(fb_sort_den, C_tan)
    abs_den_over_cos = abs(den_over_cos)
    return den_over_cos, abs_den_over_cos


# ---------------------------------------------------------------------------
# Wall height + texture coord
# ---------------------------------------------------------------------------


def _compute_wall_height(
    fb_H_inv: Node,
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
        fb_H_inv,
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
    fb_D: Node,
    fb_E: Node,
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
        fb_E,
        tan_o,
        DIFF_BP,
        tan_val_bp,
        lambda a, b: a * b,
        name="E_tan_o",
    )
    num_u_over_cos = add(fb_D, E_tan)
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
        pos_encoding,
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
    render_chunk_start: Node,
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

    # Chunk-start sentinel: host writes -1 on "new column", meaning
    # "start at the wall's visible top".
    neg_half = create_literal_value(torch.tensor([-0.5]), name="neg_half")
    is_new_col = compare(subtract(neg_half, render_chunk_start), 0.0)
    active_start = select(is_new_col, vis_top_render, render_chunk_start)

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
    render_mask: Node,
    fb_onehot: Node,
    fb_col_hi: Node,
    fb_sort_den: Node,
    fb_C: Node,
    fb_D: Node,
    fb_E: Node,
    fb_H_inv: Node,
    fb_tex_id: Node,
    fb_col_lo: Node,
    chunk_size: int,
    max_walls: int,
):
    """Three-way state transition: more chunks / advance col / advance wall.

    Returns the 5-wide precompute feedback (forwarded unchanged), the
    discrete state outputs, ``done_flag``, and ``render_next_type``.
    """
    next_chunk_start_val = add_const(active_start, float(chunk_size))
    has_more_chunks = compare(
        subtract(wall_bottom_clamped, next_chunk_start_val),
        0.5,
    )

    col_p1 = add_const(active_col, 1.0)
    not_more_chunks = bool_not(has_more_chunks)
    has_more_cols = compare(subtract(fb_col_hi, col_p1), 0.5)
    advance_col = bool_all_true([not_more_chunks, has_more_cols])
    advance_wall = bool_all_true([not_more_chunks, bool_not(has_more_cols)])

    mask_with_new = add(render_mask, fb_onehot)
    next_render_mask = select(advance_wall, mask_with_new, render_mask)

    mask_sum = Linear(
        mask_with_new,
        torch.ones(max_walls, 1),
        name="render_mask_sum",
    )
    all_walls_done = compare(mask_sum, max_walls - 0.5)
    done_flag = bool_all_true([advance_wall, all_walls_done])

    zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
    next_col_output = select(
        has_more_chunks,
        active_col,
        select(advance_col, col_p1, zero_col),
    )
    chunk_sentinel = create_literal_value(
        torch.tensor([-1.0]),
        name="chunk_sentinel",
    )
    next_chunk = select(has_more_chunks, next_chunk_start_val, chunk_sentinel)

    pos_one = create_literal_value(torch.tensor([1.0]), name="pos_one")
    neg_one = create_literal_value(torch.tensor([-1.0]), name="neg_one")
    next_is_new_wall = select(advance_wall, pos_one, neg_one)

    render_next_type = select(
        advance_wall,
        create_literal_value(E8_THINKING, name="type_thinking"),
        create_literal_value(E8_RENDER, name="type_render"),
    )

    next_render_feedback = Concatenate(
        [fb_sort_den, fb_C, fb_D, fb_E, fb_H_inv]
    )

    return (
        next_render_feedback,
        done_flag,
        render_next_type,
        next_render_mask,
        next_col_output,
        next_is_new_wall,
        next_chunk,
        fb_tex_id,
        fb_col_lo,
        fb_col_hi,
        fb_onehot,
    )
