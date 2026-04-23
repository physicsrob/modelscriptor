"""Walls-as-tokens game graph — orchestrator.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M → WALL×N →
    EOS → PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
    [SORTED_WALL → RENDER×k]×N  (interleaved sort + render)

Token carrier: every position consumes a single integer ``token_ids``
input and produces a single integer ``next_token_id`` via the
embedding lookup + argmax head in :class:`CompiledToken`.  Detectors
for each named category compare the per-position embedding leaf
against the appropriate ``W_EMBED`` row (see
``torchwright.doom.embedding``).  Bypasses (wall geometry, player
state, texture pixels, overlaid autoregressive state) flow through
separate residual leaves alongside the embedding.

Every cross-position data dependency flows through attention.  See
each stage file for per-stage math.  This module does orchestration
only.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.embedding import Embedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import bool_any_true, equals_vector
from torchwright.ops.map_select import select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.embedding import (
    D_EMBED,
    E8_VALUE,
    IDENTIFIER_NAMES,
    V,
    build_doom_embedding,
    embed_lookup,
)
from torchwright.doom.graph_constants import TEX_E8_OFFSET

__all__ = [
    "GameGraphIO",
    "build_game_graph",
    "TEX_E8_OFFSET",
]
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.bsp import BspToken, build_bsp
from torchwright.doom.stages.input import InputToken, build_input
from torchwright.doom.stages.player import PlayerKVInput, PlayerToken, build_player
from torchwright.doom.stages.render import RenderKVInput, RenderToken, build_render
from torchwright.doom.stages.sorted import SortedKVInput, SortedToken, build_sorted
from torchwright.doom.stages.tex_col import TexColToken, build_tex_col
from torchwright.doom.stages.thinking_wall import (
    ThinkingWallKVInput,
    build_thinking_wall,
)
from torchwright.doom.stages.wall import WallKVInput, WallToken, build_wall

# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------


@dataclass
class GameGraphIO:
    """Structured return value of :func:`build_game_graph`.

    ``inputs`` maps name → input node (host-fed at every position).
    Includes ``token_ids`` (1-wide integer slot feeding the embedding
    leaf) plus every bypass field.

    ``overlaid_outputs`` maps name → output node whose value lands
    back at the matching input's columns via delta transfer — these
    carry autoregressive bypass feedback (render_col, render_chunk_k,
    wall_counter).

    ``overflow_outputs`` maps name → output node placed after the
    input region.  Includes ``next_token_embedding`` (72-wide) which
    the :class:`CompiledToken` wrapper argmaxes against ``W_EMBED.T``
    to produce ``next_token_id``.

    ``embedding`` is the :class:`Embedding` leaf hand-wired to read
    from ``token_ids`` — callers need it to build the
    :class:`CompiledToken` wrapper.
    """

    inputs: Dict[str, Node]
    overlaid_outputs: Dict[str, Node]
    overflow_outputs: Dict[str, Node]
    embedding: Embedding

    def concat_output(self) -> Node:
        """Single concatenated output node for legacy callers."""
        parts = []
        for name, node in self.overlaid_outputs.items():
            parts.append(node)
        for name, node in self.overflow_outputs.items():
            parts.append(node)
        return Concatenate(parts)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_game_graph(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    chunk_size: int = 20,
    max_bsp_nodes: int = 48,
    tex_sample_batch_size: int = 8,
) -> Tuple[GameGraphIO, PosEncoding]:
    """Build the walls-as-tokens game graph.

    See the module docstring for the per-token-type layout.
    ``max_bsp_nodes`` is the width of the BSP side vector (one slot per
    BSP_NODE token); it bounds how deep × broad the BSP tree can be.
    """
    import time as _time

    _t = _time.perf_counter()

    def _mark(label: str) -> None:
        nonlocal _t
        print(
            f"  build_game_graph/{label}: {_time.perf_counter() - _t:.1f}s", flush=True
        )
        _t = _time.perf_counter()

    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]
    cs = chunk_size

    pos_encoding = create_pos_encoding()

    inputs = _create_inputs(
        max_walls=max_walls,
        tex_h=tex_h,
        max_bsp_nodes=max_bsp_nodes,
        max_coord=max_coord,
    )
    # The embedding is the per-position 72-wide residual leaf produced
    # by looking up ``inputs["token_ids"]`` in ``W_EMBED``.  All
    # token-type detectors run against this leaf (specific-ID
    # comparisons or a category-only slice for ``is_thinking_value``).
    embedding = build_doom_embedding(input_name="token_ids")
    tf = _detect_token_types(embedding)
    _mark("inputs+detect")

    # ---------- INPUT ----------
    input_out = build_input(
        InputToken(
            player_angle=inputs["player_angle"],
            input_turn_left=inputs["input_turn_left"],
            input_turn_right=inputs["input_turn_right"],
            input_forward=inputs["input_forward"],
            input_backward=inputs["input_backward"],
            input_strafe_left=inputs["input_strafe_left"],
            input_strafe_right=inputs["input_strafe_right"],
        ),
        is_input=tf["is_input"],
        pos_encoding=pos_encoding,
        turn_speed=turn_speed,
        move_speed=move_speed,
    )
    _mark("input")

    # ---------- TEX_COL ----------
    tex_col_out = build_tex_col(
        TexColToken(tex_col_input=inputs["tex_col_input"]),
        tex_w=tex_w,
    )
    _mark("tex_col")

    # ---------- BSP ----------
    bsp_out = build_bsp(
        BspToken(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            bsp_plane_nx=inputs["bsp_plane_nx"],
            bsp_plane_ny=inputs["bsp_plane_ny"],
            bsp_plane_d=inputs["bsp_plane_d"],
            bsp_node_id_onehot=inputs["bsp_node_id_onehot"],
        ),
        is_bsp_node=tf["is_bsp_node"],
        pos_encoding=pos_encoding,
        max_coord=max_coord,
        max_bsp_nodes=max_bsp_nodes,
    )
    _mark("bsp")

    # ---------- WALL ----------
    wall_out = build_wall(
        WallToken(
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_tex_id=inputs["wall_tex_id"],
            wall_index=inputs["wall_index"],
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            wall_bsp_coeffs=inputs["wall_bsp_coeffs"],
            wall_bsp_const=inputs["wall_bsp_const"],
        ),
        WallKVInput(
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
            move_cos=input_out.move_cos,
            move_sin=input_out.move_sin,
            side_P_vec=bsp_out.side_P_vec,
        ),
        is_wall=tf["is_wall"],
        config=config,
        max_walls=max_walls,
        max_coord=max_coord,
        max_bsp_nodes=max_bsp_nodes,
    )
    _mark("wall")

    # EOS stage: gutted in Phase A Part 4.  The EOS token remains in
    # the prompt as an end-of-prompt marker but performs no graph
    # computation.  Collision resolution moved to the RESOLVED_X /
    # RESOLVED_Y identifiers in the thinking phase; the post-turn
    # angle is carried by INPUT's ``new_angle`` broadcast and emitted
    # at the RESOLVED_ANGLE identifier step.

    # ---------- PLAYER ----------
    player_out = build_player(
        PlayerToken(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
        ),
        PlayerKVInput(new_angle=input_out.new_angle),
        is_player_x=tf["is_player_x"],
        is_player_y=tf["is_player_y"],
        is_player_angle=tf["is_player_angle"],
        pos_encoding=pos_encoding,
    )
    _mark("player")

    # ---------- THINKING_WALL ----------
    # Phase A Part 4: the thinking-wall stage now carries the full
    # 16-identifier state machine including the 3 RESOLVED slots at the
    # frame boundary.  Pre-collision player position comes from the
    # PLAYER broadcast (post-Part-4 the PLAYER broadcast is pre-collision
    # — the host feeds pre-collision game_state to PLAYER_X / PLAYER_Y);
    # the previous pre-collision bypass at thinking positions is gone.
    # RESOLVED_X/Y aggregate WALL-stage collision flags via
    # ``attend_mean_where``; RESOLVED_ANGLE forwards the post-turn angle
    # from INPUT's broadcast.
    thinking_wall_out = build_thinking_wall(
        ThinkingWallKVInput(
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_position_onehot=wall_out.position_onehot,
            wall_bsp_coeffs=inputs["wall_bsp_coeffs"],
            wall_bsp_const=inputs["wall_bsp_const"],
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
            move_cos=input_out.move_cos,
            move_sin=input_out.move_sin,
            side_P_vec=bsp_out.side_P_vec,
            player_x=player_out.px,
            player_y=player_out.py,
            new_angle=input_out.new_angle,
            hit_full=wall_out.collision.hit_full,
            hit_x=wall_out.collision.hit_x,
            hit_y=wall_out.collision.hit_y,
        ),
        embedding=embedding,
        is_wall=tf["is_wall"],
        is_thinking_wall_marker=tf["is_thinking_wall_marker"],
        is_thinking_wall_n=tf["is_thinking_wall_n"],
        is_any_identifier=tf["is_any_identifier"],
        is_identifier_by_slot=tf["is_identifier_by_slot"],
        is_thinking_value=tf["is_thinking_value"],
        pos_encoding=pos_encoding,
        max_walls=max_walls,
        max_coord=max_coord,
        max_bsp_nodes=max_bsp_nodes,
        config=config,
    )
    _mark("thinking_wall")

    # ---------- SORTED ----------
    sorted_out = build_sorted(
        SortedToken(
            wall_counter=inputs["wall_counter"],
        ),
        SortedKVInput(
            sort_score=wall_out.sort_score,
            sort_value=wall_out.sort_value,
            indicators_above=wall_out.indicators_above,
        ),
        is_sorted=tf["is_sorted"],
        is_wall=tf["is_wall"],
        pos_encoding=pos_encoding,
        max_walls=max_walls,
    )
    _mark("sorted")

    # ---------- RENDER ----------
    # Phase A Part 4: post-collision (x, y) come from the RESOLVED_X /
    # RESOLVED_Y thinking tokens via the shared readback helper rather
    # than the PLAYER broadcast (which is pre-collision now).  cos/sin
    # still come from PLAYER_ANGLE's broadcast — collision doesn't
    # change angle.
    resolved_x_readback = thinking_wall_out.readback.get_value_after_last("RESOLVED_X")
    _mark("render/resolved_x_readback")
    resolved_y_readback = thinking_wall_out.readback.get_value_after_last("RESOLVED_Y")
    _mark("render/resolved_y_readback")
    render_out = build_render(
        RenderToken(
            col=inputs["render_col"],
            chunk_k=inputs["render_chunk_k"],
            wall_counter=inputs["wall_counter"],
        ),
        RenderKVInput(
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_tex_id=inputs["wall_tex_id"],
            wall_vis_hi=wall_out.vis_hi,
            wall_position_onehot=wall_out.position_onehot,
            player_x=resolved_x_readback,
            player_y=resolved_y_readback,
            player_cos=player_out.cos_theta,
            player_sin=player_out.sin_theta,
            texture_id_e8=inputs["texture_id_e8"],
            tex_pixels=inputs["tex_pixels"],
            tc_onehot_01=tex_col_out.tc_onehot_01,
            # Phase A M3: wall identity reaches RENDER via attention to
            # the most recent SORTED position.  No more overlaid
            # render_wall_index feedback.  Score is the host-fed
            # ``wall_counter`` (monotonically increasing at SORTED);
            # validity is is_sorted.
            wall_counter=inputs["wall_counter"],
            sorted_wall_index=sorted_out.wall_index,
            is_sorted=tf["is_sorted"],
        ),
        is_render=tf["is_render"],
        is_wall=tf["is_wall"],
        is_tex_col=tf["is_tex_col"],
        pos_encoding=pos_encoding,
        config=config,
        textures=textures,
        chunk_size=cs,
        max_coord=max_coord,
        max_walls=max_walls,
        tex_sample_batch_size=tex_sample_batch_size,
    )
    _mark("render")

    # ---------- Output assembly ----------
    overlaid, overflow = _assemble_output(
        token_flags=tf,
        inputs=inputs,
        input_out=input_out,
        sorted_out=sorted_out,
        render_out=render_out,
        thinking_wall_out=thinking_wall_out,
        chunk_size=cs,
    )
    _mark("assemble_output")

    return (
        GameGraphIO(
            inputs=inputs,
            overlaid_outputs=overlaid,
            overflow_outputs=overflow,
            embedding=embedding,
        ),
        pos_encoding,
    )


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _create_inputs(
    *,
    max_walls: int,
    tex_h: int,
    max_bsp_nodes: int,
    max_coord: float,
) -> Dict[str, Node]:
    """Create host-fed input nodes.

    One token-ID slot plus bypass leaves.  Overlaid bypasses
    (render_col, render_chunk_k, wall_counter) carry autoregressive
    state via delta-transfer; everything else is prompt-fed.
    """
    inputs: Dict[str, Node] = {}

    # Phase A Part 1: discrete token ID slot (one per position).  The
    # graph's ``Embedding`` leaf reads this and produces the 72-wide
    # per-position embedding; detectors run against that.
    inputs["token_ids"] = create_input("token_ids", 1, value_range=(0.0, float(V - 1)))
    inputs["player_x"] = create_input(
        "player_x", 1, value_range=(-max_coord, max_coord)
    )
    inputs["player_y"] = create_input(
        "player_y", 1, value_range=(-max_coord, max_coord)
    )
    inputs["player_angle"] = create_input("player_angle", 1, value_range=(0.0, 255.0))
    for k in (
        "input_forward",
        "input_backward",
        "input_turn_left",
        "input_turn_right",
        "input_strafe_left",
        "input_strafe_right",
    ):
        inputs[k] = create_input(k, 1, value_range=(0.0, 1.0))
    for k in ("wall_ax", "wall_ay", "wall_bx", "wall_by"):
        inputs[k] = create_input(k, 1, value_range=(-max_coord, max_coord))
    for k in ("wall_tex_id", "wall_index"):
        inputs[k] = create_input(k, 1, value_range=(0.0, 255.0))
    inputs["tex_col_input"] = create_input("tex_col_input", 1, value_range=(0.0, 255.0))
    inputs["tex_pixels"] = create_input("tex_pixels", tex_h * 3, value_range=(0.0, 1.0))
    inputs["texture_id_e8"] = create_input(
        "texture_id_e8", 8, value_range=(-30.0, 30.0)
    )

    inputs["bsp_plane_nx"] = create_input("bsp_plane_nx", 1, value_range=(-1.0, 1.0))
    inputs["bsp_plane_ny"] = create_input("bsp_plane_ny", 1, value_range=(-1.0, 1.0))
    inputs["bsp_plane_d"] = create_input(
        "bsp_plane_d", 1, value_range=(-max_coord, max_coord)
    )
    inputs["bsp_node_id_onehot"] = create_input(
        "bsp_node_id_onehot",
        max_bsp_nodes,
        value_range=(0.0, 1.0),
    )
    inputs["wall_bsp_coeffs"] = create_input(
        "wall_bsp_coeffs",
        max_bsp_nodes,
        value_range=(-max_coord, max_coord),
    )
    inputs["wall_bsp_const"] = create_input(
        "wall_bsp_const",
        1,
        value_range=(-max_coord, max_coord),
    )

    # Autoregressive state — 3 bounded integers plus token_type.
    # Phase A M3: render_wall_index is no longer an overlaid input;
    # RENDER reads the current wall index via attention to the most
    # recent SORTED position.
    inputs["wall_counter"] = create_input(
        "wall_counter", 1, value_range=(0.0, float(max_walls))
    )
    inputs["render_col"] = create_input("render_col", 1, value_range=(0.0, 255.0))
    inputs["render_chunk_k"] = create_input(
        "render_chunk_k", 1, value_range=(0.0, 20.0)
    )

    return inputs


def _detect_token_types(embedding: Node) -> Dict[str, Any]:
    """Derive per-type boolean flags from the 72-wide embedding leaf.

    Every detector is an ``equals_vector`` against a fixed ``W_EMBED``
    row (for specific-ID categories) or against the ``E8_VALUE``
    category code (for the category-only ``is_thinking_value`` — all
    65,536 VALUE rows share this 8-wide prefix).

    Per-identifier detectors (Part 2): one detector per entry in
    ``IDENTIFIER_NAMES`` (16 total — 13 per-wall + 3 RESOLVED).  Both a
    dict-keyed convenience form (``is_bsp_rank_id``, ``is_hit_full_id``,
    ``is_resolved_x_id``, …) and an ordered ``is_identifier_by_slot``
    list are exposed; the thinking-wall stage indexes by slot while
    other stages read by name.
    """
    with annotate("token_type"):
        flags: Dict[str, Any] = {
            "is_input": equals_vector(embedding, embed_lookup("INPUT")),
            "is_wall": equals_vector(embedding, embed_lookup("WALL")),
            "is_eos": equals_vector(embedding, embed_lookup("EOS")),
            "is_sorted": equals_vector(embedding, embed_lookup("SORTED_WALL")),
            "is_render": equals_vector(embedding, embed_lookup("RENDER")),
            "is_tex_col": equals_vector(embedding, embed_lookup("TEX_COL")),
            "is_bsp_node": equals_vector(embedding, embed_lookup("BSP_NODE")),
            "is_player_x": equals_vector(embedding, embed_lookup("PLAYER_X")),
            "is_player_y": equals_vector(embedding, embed_lookup("PLAYER_Y")),
            "is_player_angle": equals_vector(embedding, embed_lookup("PLAYER_ANGLE")),
            # Category-only: any VALUE token (cols [0:8] == E8_VALUE).
            "is_thinking_value": equals_vector(
                extract_from(embedding, D_EMBED, 0, 8, "value_category_cols"),
                E8_VALUE,
            ),
        }
        # Per-identifier detectors (16 total).  Built in IDENTIFIER_NAMES
        # order; both indexed (by slot) and keyed (by name) for downstream
        # convenience.
        is_identifier_by_slot = [
            equals_vector(embedding, embed_lookup(name)) for name in IDENTIFIER_NAMES
        ]
        flags["is_identifier_by_slot"] = is_identifier_by_slot
        for name, detector in zip(IDENTIFIER_NAMES, is_identifier_by_slot):
            flags[f"is_{name.lower()}_id"] = detector
        flags["is_any_identifier"] = bool_any_true(is_identifier_by_slot)
        # Per-marker detectors (8 walls).  is_thinking_wall_marker is the
        # OR — used as the validity signal in the value step's "find
        # current wall_index" attention.
        is_thinking_wall_n = [
            equals_vector(embedding, embed_lookup(f"THINKING_WALL_{i}"))
            for i in range(8)
        ]
        flags["is_thinking_wall_n"] = is_thinking_wall_n
        flags["is_thinking_wall_marker"] = bool_any_true(is_thinking_wall_n)
        return flags


def _assemble_output(
    *,
    token_flags: Dict[str, Node],
    inputs: Dict[str, Node],
    input_out,
    sorted_out,
    render_out,
    thinking_wall_out,
    chunk_size: int,
) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Build the overlaid outputs + overflow outputs.

    Overlaid outputs fed back into the next step's input (bypass
    autoregressive state — delta-transferred via matching input slots):
        render_col, render_chunk_k, wall_counter

    Overflow outputs (placed after the input region):
        next_token_embedding (72-wide — CompiledToken argmaxes it to
        pick the next ``token_ids`` input),
        pixels, col, start, length, done, advance_wall,
        sort_done, sort_vis_hi, sort_wall_index

    ``sort_wall_index`` is host-visible but not fed back — the trace
    harness reads it for walkthrough recording and the autoregressive
    loop doesn't need it as input (RENDER gets its wall identity via
    attention, not via this field).

    Phase A Part 4: the former per-EOS-token overflow outputs for
    resolved state are gone — collision-resolved state now flows
    through the RESOLVED_X/Y/ANGLE thinking-token stream, and the
    host reads it from argmax outputs at known step offsets.  The
    PLAYER_ANGLE step emits ``THINKING_WALL_0`` as its next-token
    prediction so the host no longer synthesizes the first thinking
    token.
    """
    with annotate("output"):
        zero_1 = create_literal_value(torch.tensor([0.0]), name="zero_1")
        zero_embedding = create_literal_value(
            torch.zeros(D_EMBED), name="zero_embedding"
        )
        zero_pixels = create_literal_value(
            torch.zeros(chunk_size * 3),
            name="zero_pixels",
        )
        neg_one = create_literal_value(torch.tensor([-1.0]), name="neg_one_out")

        # render_col: SORTED sets to vis_lo; RENDER advances.
        out_render_col = select(
            token_flags["is_sorted"],
            sorted_out.col,
            select(token_flags["is_render"], render_out.next_col, zero_1),
        )

        # render_chunk_k: SORTED resets to 0; RENDER advances.
        out_render_chunk_k = select(
            token_flags["is_render"], render_out.next_chunk_k, zero_1
        )

        # wall_counter: SORTED increments; RENDER forwards.
        out_wall_counter = select(
            token_flags["is_sorted"],
            sorted_out.next_wall_counter,
            select(
                token_flags["is_render"],
                render_out.next_wall_counter,
                zero_1,
            ),
        )

        # sort_wall_index: SORTED emits the picked wall index as an
        # overflow field.  RENDER doesn't forward it (it reads wall
        # identity from the KV cache via attention).  Only populated at
        # SORTED positions; zero elsewhere.  Host-visible for the trace
        # harness, not fed back as input.
        out_sort_wall_index = select(
            token_flags["is_sorted"],
            sorted_out.wall_index,
            zero_1,
        )

        # --- Next-token embedding ---
        # Phase A Part 1: every position emits the 72-wide embedding of
        # the next token the transformer should receive.  Thinking
        # positions drive the first half of the sequence (marker →
        # identifier → value → ...); at SORTED positions the next token
        # is always RENDER; at RENDER positions it's RENDER or
        # SORTED_WALL depending on wall-advance state.
        #
        # Phase A Part 4: PLAYER_ANGLE's step now emits ``THINKING_WALL_0``
        # as its next-token prediction so the autoregressive loop picks
        # it up naturally — the host no longer synthesizes the first
        # thinking token.  Elsewhere (prompt positions before
        # PLAYER_ANGLE) the emitted value is a don't-care zero — the
        # host never argmaxes these because it feeds the known prompt
        # tokens directly.
        type_render = create_literal_value(
            embed_lookup("RENDER"), name="out_type_render"
        )
        type_thinking_wall_0 = create_literal_value(
            embed_lookup("THINKING_WALL_0"), name="out_type_thinking_wall_0"
        )
        out_next_token_embedding = select(
            thinking_wall_out.is_thinking_active,
            thinking_wall_out.next_token_embedding,
            select(
                token_flags["is_sorted"],
                type_render,
                select(
                    token_flags["is_render"],
                    render_out.render_next_type,
                    select(
                        token_flags["is_player_angle"],
                        type_thinking_wall_0,
                        zero_embedding,
                    ),
                ),
            ),
        )

        out_pixels = select(token_flags["is_render"], render_out.pixels, zero_pixels)
        out_col = select(token_flags["is_render"], render_out.active_col, zero_1)
        out_start = select(token_flags["is_render"], render_out.active_start, zero_1)
        out_length = select(token_flags["is_render"], render_out.chunk_length, zero_1)
        out_done = select(token_flags["is_render"], render_out.done_flag, neg_one)
        out_advance_wall = select(
            token_flags["is_render"], render_out.advance_wall, neg_one
        )

        out_sort_done = select(token_flags["is_sorted"], sorted_out.sort_done, neg_one)
        out_sort_vis_hi = select(token_flags["is_sorted"], sorted_out.vis_hi, zero_1)

    overlaid = {
        "render_col": out_render_col,
        "render_chunk_k": out_render_chunk_k,
        "wall_counter": out_wall_counter,
    }
    overflow = {
        "next_token_embedding": out_next_token_embedding,
        "pixels": out_pixels,
        "col": out_col,
        "start": out_start,
        "length": out_length,
        "done": out_done,
        "advance_wall": out_advance_wall,
        "sort_done": out_sort_done,
        "sort_vis_hi": out_sort_vis_hi,
        "sort_wall_index": out_sort_wall_index,
    }
    return overlaid, overflow
