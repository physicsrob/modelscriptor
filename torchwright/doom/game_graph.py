"""Walls-as-tokens game graph — orchestrator.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M → WALL×N →
    EOS → PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
    [SORTED_WALL → RENDER×k]×N  (interleaved sort + render)

Token types (E8 spherical codes):

    TEX_COL (5)      Texture column pixel data.
    INPUT (0)        Player state + movement inputs.
    BSP_NODE (7)     BSP splitting plane classification.
    WALL (1)         Wall geometry + BSP rank + visibility.
    EOS (2)          Collision resolution + state broadcast.
    PLAYER_X (240)   Broadcast resolved x position.
    PLAYER_Y (241)   Broadcast resolved y position.
    PLAYER_ANGLE (242) Broadcast cos(θ)/sin(θ).
    SORTED_WALL (3)  Autoregressive front-to-back sort.
    RENDER (4)       Pixel generation (chunked column fill).

Every cross-position data dependency flows through attention.  See each
stage file for per-stage math.  This module does orchestration only.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import bool_any_true, equals_vector
from torchwright.ops.map_select import select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import (
    E8_BSP_NODE,
    E8_EOS,
    E8_HIT_FULL_ID,
    E8_HIT_X_ID,
    E8_HIT_Y_ID,
    E8_INPUT,
    E8_PLAYER_ANGLE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_THINKING_VALUE,
    E8_THINKING_WALL,
    E8_WALL,
    TEX_E8_OFFSET,
)

# Re-export token constants for legacy callers that import them from
# game_graph (compile.py, tooling, tests).  graph_constants is the
# authoritative definition; these re-exports are just a compatibility
# surface.
__all__ = [
    "GameGraphIO",
    "build_game_graph",
    "E8_INPUT",
    "E8_WALL",
    "E8_EOS",
    "E8_SORTED_WALL",
    "E8_RENDER",
    "E8_TEX_COL",
    "E8_BSP_NODE",
    "E8_PLAYER_X",
    "E8_PLAYER_Y",
    "E8_PLAYER_ANGLE",
    "E8_THINKING_WALL",
    "E8_HIT_FULL_ID",
    "E8_HIT_X_ID",
    "E8_HIT_Y_ID",
    "E8_THINKING_VALUE",
    "TEX_E8_OFFSET",
]
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.bsp import BspToken, build_bsp
from torchwright.doom.stages.eos import EosKVInput, EosToken, build_eos
from torchwright.doom.stages.input import InputToken, build_input
from torchwright.doom.stages.player import PlayerToken, build_player
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
    ``overlaid_outputs`` maps name → output node whose value lands back
    at the matching input's columns via delta transfer — these carry
    autoregressive feedback.  ``overflow_outputs`` maps name → output
    node placed after the input region (pixels + metadata).
    """

    inputs: Dict[str, Node]
    overlaid_outputs: Dict[str, Node]
    overflow_outputs: Dict[str, Node]

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
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]
    cs = chunk_size

    pos_encoding = create_pos_encoding()

    inputs = _create_inputs(
        max_walls=max_walls,
        tex_h=tex_h,
        max_bsp_nodes=max_bsp_nodes,
        max_coord=max_coord,
    )
    tf = _detect_token_types(inputs["token_type"])

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

    # ---------- TEX_COL ----------
    tex_col_out = build_tex_col(
        TexColToken(tex_col_input=inputs["tex_col_input"]),
        tex_w=tex_w,
    )

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

    # ---------- EOS ----------
    eos_out = build_eos(
        EosToken(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
        ),
        EosKVInput(
            collision=wall_out.collision,
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
        ),
        is_wall=tf["is_wall"],
        pos_encoding=pos_encoding,
    )

    # ---------- PLAYER ----------
    player_out = build_player(
        PlayerToken(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            player_angle=inputs["player_angle"],
        ),
        is_player_x=tf["is_player_x"],
        is_player_y=tf["is_player_y"],
        is_player_angle=tf["is_player_angle"],
        pos_encoding=pos_encoding,
    )

    # ---------- THINKING_WALL ----------
    # Phase A M4: emits hit_full/hit_x/hit_y per wall as autoregressive
    # value tokens.  Lives between PLAYER and SORTED in the token stream
    # but participates in the per-position graph the same way every
    # other stage does — token-type detectors gate which positions emit.
    #
    # Pre-collision player position note: PLAYER tokens currently carry
    # the EOS-resolved (post-collision) player state, since RENDER needs
    # post-collision to project the actual on-screen view.  HIT_*
    # computation, however, needs pre-collision position (the question
    # is whether this *intended* velocity ray hits anything before
    # resolution).  We grab pre-collision via the raw ``inputs["player_x"]``
    # / ``inputs["player_y"]`` slots — the host fills them with
    # pre-collision values at thinking-token positions (see
    # ``compile.step_frame``).
    thinking_wall_out = build_thinking_wall(
        ThinkingWallKVInput(
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_position_onehot=wall_out.position_onehot,
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
        ),
        is_wall=tf["is_wall"],
        is_thinking_wall_marker=tf["is_thinking_wall_marker"],
        is_thinking_wall_n=tf["is_thinking_wall_n"],
        is_any_identifier=tf["is_any_identifier"],
        is_hit_full_id=tf["is_hit_full_id"],
        is_hit_x_id=tf["is_hit_x_id"],
        is_hit_y_id=tf["is_hit_y_id"],
        is_thinking_value=tf["is_thinking_value"],
        pos_encoding=pos_encoding,
        max_walls=max_walls,
    )

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

    # ---------- RENDER ----------
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
            player_x=player_out.px,
            player_y=player_out.py,
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

    # ---------- Output assembly ----------
    overlaid, overflow = _assemble_output(
        token_flags=tf,
        inputs=inputs,
        eos_out=eos_out,
        input_out=input_out,
        sorted_out=sorted_out,
        render_out=render_out,
        thinking_wall_out=thinking_wall_out,
        chunk_size=cs,
    )

    return (
        GameGraphIO(
            inputs=inputs,
            overlaid_outputs=overlaid,
            overflow_outputs=overflow,
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

    No feedback vectors — all state is carried by discrete overlaid
    fields that the host copies verbatim from output to input.
    """
    inputs: Dict[str, Node] = {}

    inputs["token_type"] = create_input("token_type", 8, value_range=(-30.0, 30.0))
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

    # Phase A M4: thinking-token value payload.  At THINKING_VALUE
    # positions the host re-feeds the previous output's quantized
    # integer here.  HIT_* values are 0/1, but we keep the wider range
    # so the same slot can carry M5+ values (cross/dot in [-40, 40],
    # vis_lo/hi in [-2, 122], etc.) without an IO-surface change.
    inputs["thinking_value"] = create_input(
        "thinking_value", 1, value_range=(-128.0, 65535.0)
    )

    return inputs


def _detect_token_types(token_type: Node) -> Dict[str, Any]:
    """Derive per-type boolean flags from the 8-wide token_type vector.

    Most entries are ``Node`` booleans, but ``is_thinking_wall_n`` is a
    ``list[Node]`` — one detector per marker — hence the wider ``Any``
    value type.
    """
    with annotate("token_type"):
        flags: Dict[str, Any] = {
            "is_input": equals_vector(token_type, E8_INPUT),
            "is_wall": equals_vector(token_type, E8_WALL),
            "is_eos": equals_vector(token_type, E8_EOS),
            "is_sorted": equals_vector(token_type, E8_SORTED_WALL),
            "is_render": equals_vector(token_type, E8_RENDER),
            "is_tex_col": equals_vector(token_type, E8_TEX_COL),
            "is_bsp_node": equals_vector(token_type, E8_BSP_NODE),
            "is_player_x": equals_vector(token_type, E8_PLAYER_X),
            "is_player_y": equals_vector(token_type, E8_PLAYER_Y),
            "is_player_angle": equals_vector(token_type, E8_PLAYER_ANGLE),
            # Phase A M4: thinking-token detectors.
            "is_hit_full_id": equals_vector(token_type, E8_HIT_FULL_ID),
            "is_hit_x_id": equals_vector(token_type, E8_HIT_X_ID),
            "is_hit_y_id": equals_vector(token_type, E8_HIT_Y_ID),
            "is_thinking_value": equals_vector(token_type, E8_THINKING_VALUE),
        }
        # Per-marker detectors (8 walls).  is_thinking_wall_marker is the
        # OR — used as the validity signal in the value step's "find
        # current wall_index" attention.
        is_thinking_wall_n = [
            equals_vector(token_type, E8_THINKING_WALL[i]) for i in range(8)
        ]
        flags["is_thinking_wall_n"] = is_thinking_wall_n
        flags["is_thinking_wall_marker"] = bool_any_true(is_thinking_wall_n)
        flags["is_any_identifier"] = bool_any_true(
            [
                flags["is_hit_full_id"],
                flags["is_hit_x_id"],
                flags["is_hit_y_id"],
            ]
        )
        return flags


def _assemble_output(
    *,
    token_flags: Dict[str, Node],
    inputs: Dict[str, Node],
    eos_out,
    input_out,
    sorted_out,
    render_out,
    thinking_wall_out,
    chunk_size: int,
) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Build the overlaid outputs + overflow outputs.

    Overlaid outputs fed back into the next step's input:
        token_type, render_col, render_chunk_k, wall_counter

    SORTED sets col for the first RENDER token, then outputs
    token_type = E8_RENDER.  RENDER tokens advance col/chunk_k and
    output token_type = E8_SORTED_WALL on wall transitions (not done).
    Wall identity travels through the KV cache now (Phase A M3): RENDER
    reads ``sorted_out.wall_index`` from the most-recent SORTED position
    via ``attend_most_recent_matching``.

    Overflow outputs bitblitted/inspected by the host:
        pixels, col, start, length, done, advance_wall,
        sort_done, sort_vis_hi, sort_wall_index,
        eos_resolved_x, eos_resolved_y, eos_new_angle

    ``sort_wall_index`` is host-visible but not fed back — the trace
    harness reads it for walkthrough recording and the autoregressive
    loop doesn't need it as input (RENDER gets its wall identity via
    attention, not via this field).
    """
    with annotate("output"):
        zero_1 = create_literal_value(torch.tensor([0.0]), name="zero_1")
        zero_8 = create_literal_value(torch.zeros(8), name="zero_8")
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

        # --- Token type ---
        # Phase A M4: thinking-token positions emit their own next type
        # (marker → HIT_FULL_ID, identifier → THINKING_VALUE, value →
        # next identifier or marker or SORTED).  Outside thinking,
        # SORTED→RENDER and RENDER→SORTED_WALL/RENDER as before.
        type_render = create_literal_value(E8_RENDER, name="out_type_render")
        out_token_type = select(
            thinking_wall_out.is_thinking_active,
            thinking_wall_out.next_token_type,
            select(
                token_flags["is_sorted"],
                type_render,
                select(
                    token_flags["is_render"],
                    render_out.render_next_type,
                    zero_8,
                ),
            ),
        )

        # Phase A M4: thinking_value overlaid output.  Carries the
        # quantized integer the host re-feeds at the next THINKING_VALUE
        # input.  Zero at non-VALUE positions.
        out_thinking_value = select(
            token_flags["is_thinking_value"],
            thinking_wall_out.thinking_value,
            zero_1,
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

        out_eos_rx = select(token_flags["is_eos"], eos_out.resolved_x, zero_1)
        out_eos_ry = select(token_flags["is_eos"], eos_out.resolved_y, zero_1)
        out_eos_angle = select(token_flags["is_eos"], input_out.new_angle, zero_1)

    overlaid = {
        "token_type": out_token_type,
        "render_col": out_render_col,
        "render_chunk_k": out_render_chunk_k,
        "wall_counter": out_wall_counter,
        "thinking_value": out_thinking_value,
    }
    overflow = {
        "pixels": out_pixels,
        "col": out_col,
        "start": out_start,
        "length": out_length,
        "done": out_done,
        "advance_wall": out_advance_wall,
        "sort_done": out_sort_done,
        "sort_vis_hi": out_sort_vis_hi,
        "sort_wall_index": out_sort_wall_index,
        "eos_resolved_x": out_eos_rx,
        "eos_resolved_y": out_eos_ry,
        "eos_new_angle": out_eos_angle,
    }
    return overlaid, overflow
