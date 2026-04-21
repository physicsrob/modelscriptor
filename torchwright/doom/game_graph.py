"""Walls-as-tokens game graph — orchestrator.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M → WALL×N →
    EOS → PLAYER_X → PLAYER_Y → PLAYER_ANGLE →
    SORTED_WALL×N → RENDER×(dynamic)

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
from typing import Dict, List, Tuple

import numpy as np
import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import (
    E8_BSP_NODE,
    E8_EOS,
    E8_INPUT,
    E8_PLAYER_ANGLE,
    E8_PLAYER_X,
    E8_PLAYER_Y,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
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
    "TEX_E8_OFFSET",
]
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.bsp import BspInputs, build_bsp
from torchwright.doom.stages.eos import EosInputs, build_eos
from torchwright.doom.stages.input import InputInputs, build_input
from torchwright.doom.stages.player import PlayerInputs, build_player
from torchwright.doom.stages.render import RenderInputs, build_render
from torchwright.doom.stages.sorted import SortedInputs, build_sorted
from torchwright.doom.stages.tex_col import TexColInputs, build_tex_col
from torchwright.doom.stages.wall import WallInputs, build_wall

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
        InputInputs(
            player_angle=inputs["player_angle"],
            input_turn_left=inputs["input_turn_left"],
            input_turn_right=inputs["input_turn_right"],
            input_forward=inputs["input_forward"],
            input_backward=inputs["input_backward"],
            input_strafe_left=inputs["input_strafe_left"],
            input_strafe_right=inputs["input_strafe_right"],
            is_input=tf["is_input"],
            pos_encoding=pos_encoding,
        ),
        turn_speed=turn_speed,
        move_speed=move_speed,
    )

    # ---------- TEX_COL ----------
    tex_col_out = build_tex_col(
        TexColInputs(tex_col_input=inputs["tex_col_input"]),
        tex_w=tex_w,
    )

    # ---------- BSP ----------
    bsp_out = build_bsp(
        BspInputs(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            bsp_plane_nx=inputs["bsp_plane_nx"],
            bsp_plane_ny=inputs["bsp_plane_ny"],
            bsp_plane_d=inputs["bsp_plane_d"],
            bsp_node_id_onehot=inputs["bsp_node_id_onehot"],
            is_bsp_node=tf["is_bsp_node"],
            pos_encoding=pos_encoding,
        ),
        max_coord=max_coord,
        max_bsp_nodes=max_bsp_nodes,
    )

    # ---------- WALL ----------
    wall_out = build_wall(
        WallInputs(
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_tex_id=inputs["wall_tex_id"],
            wall_index=inputs["wall_index"],
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            is_wall=tf["is_wall"],
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
            move_cos=input_out.move_cos,
            move_sin=input_out.move_sin,
            wall_bsp_coeffs=inputs["wall_bsp_coeffs"],
            wall_bsp_const=inputs["wall_bsp_const"],
            side_P_vec=bsp_out.side_P_vec,
        ),
        config=config,
        max_walls=max_walls,
        max_coord=max_coord,
        max_bsp_nodes=max_bsp_nodes,
    )

    # ---------- EOS ----------
    eos_out = build_eos(
        EosInputs(
            is_wall=tf["is_wall"],
            is_eos=tf["is_eos"],
            collision=wall_out.collision,
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            vel_dx=input_out.vel_dx,
            vel_dy=input_out.vel_dy,
            new_angle=input_out.new_angle,
            pos_encoding=pos_encoding,
        )
    )

    # ---------- PLAYER ----------
    player_out = build_player(
        PlayerInputs(
            player_x=inputs["player_x"],
            player_y=inputs["player_y"],
            player_angle=inputs["player_angle"],
            is_player_x=tf["is_player_x"],
            is_player_y=tf["is_player_y"],
            is_player_angle=tf["is_player_angle"],
            pos_encoding=pos_encoding,
        ),
    )

    # ---------- SORTED ----------
    sorted_out = build_sorted(
        SortedInputs(
            sort_score=wall_out.sort_score,
            sort_value=wall_out.sort_value,
            indicators_above=wall_out.indicators_above,
            position_index=inputs["sort_position_index"],
            is_sorted=tf["is_sorted"],
            is_wall=tf["is_wall"],
            pos_encoding=pos_encoding,
        ),
        max_walls=max_walls,
    )

    # ---------- RENDER ----------
    render_out = build_render(
        RenderInputs(
            render_mask=inputs["render_mask"],
            render_col=inputs["render_col"],
            render_chunk_k=inputs["render_chunk_k"],
            render_tex_id=inputs["render_tex_id"],
            render_vis_lo=inputs["render_vis_lo"],
            render_vis_hi=inputs["render_vis_hi"],
            render_wall_j_onehot=inputs["render_wall_j_onehot"],
            wall_ax=inputs["wall_ax"],
            wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"],
            wall_by=inputs["wall_by"],
            wall_position_onehot=wall_out.position_onehot,
            player_x=player_out.px,
            player_y=player_out.py,
            player_cos=player_out.cos_theta,
            player_sin=player_out.sin_theta,
            texture_id_e8=inputs["texture_id_e8"],
            tex_pixels=inputs["tex_pixels"],
            tc_onehot_01=tex_col_out.tc_onehot_01,
            is_render=tf["is_render"],
            is_wall=tf["is_wall"],
            is_tex_col=tf["is_tex_col"],
            pos_encoding=pos_encoding,
        ),
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
        eos_out=eos_out,
        input_out=input_out,
        sorted_out=sorted_out,
        render_out=render_out,
        max_walls=max_walls,
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

    inputs["token_type"] = create_input("token_type", 8, value_range=(-1.0, 1.0))
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
    inputs["texture_id_e8"] = create_input("texture_id_e8", 8, value_range=(-1.0, 1.0))

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

    inputs["sort_position_index"] = create_input(
        "sort_position_index", 1, value_range=(0.0, float(max_walls))
    )

    # Discrete render state — overlaid inputs copied by the host.
    inputs["render_mask"] = create_input(
        "render_mask", max_walls, value_range=(0.0, 1.0)
    )
    inputs["render_col"] = create_input("render_col", 1, value_range=(0.0, 255.0))
    inputs["render_chunk_k"] = create_input(
        "render_chunk_k", 1, value_range=(0.0, 20.0)
    )
    inputs["render_tex_id"] = create_input("render_tex_id", 1, value_range=(0.0, 255.0))
    inputs["render_vis_lo"] = create_input(
        "render_vis_lo", 1, value_range=(-2.0, 255.0)
    )
    inputs["render_vis_hi"] = create_input(
        "render_vis_hi", 1, value_range=(-2.0, 255.0)
    )
    inputs["render_wall_j_onehot"] = create_input(
        "render_wall_j_onehot", max_walls, value_range=(0.0, 1.0)
    )

    return inputs


def _detect_token_types(token_type: Node) -> Dict[str, Node]:
    """Derive per-type boolean flags from the 8-wide token_type vector."""
    with annotate("token_type"):
        return {
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
        }


def _assemble_output(
    *,
    token_flags: Dict[str, Node],
    eos_out,
    input_out,
    sorted_out,
    render_out,
    max_walls: int,
    chunk_size: int,
) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Build the overlaid outputs + overflow outputs.

    Overlaid outputs fed back into the next step's input:
        token_type, render_mask, render_col, render_chunk_k,
        render_tex_id, render_vis_lo, render_vis_hi,
        render_wall_j_onehot

    At SORTED positions the render_* overlaid outputs carry the
    sorted wall's identity so the host can cache the sorted order.

    Overflow outputs bitblitted by the host:
        pixels, col, start, length, done, advance_wall,
        sort_done, eos_resolved_x, eos_resolved_y, eos_new_angle
    """
    with annotate("output"):
        zero_1 = create_literal_value(torch.tensor([0.0]), name="zero_1")
        zero_8 = create_literal_value(torch.zeros(8), name="zero_8")
        zero_mw = create_literal_value(torch.zeros(max_walls), name="zero_mw")
        zero_pixels = create_literal_value(
            torch.zeros(chunk_size * 3),
            name="zero_pixels",
        )
        neg_one = create_literal_value(torch.tensor([-1.0]), name="neg_one_out")

        # --- Discrete render state (overlaid outputs) ---
        # SORTED populates them for host caching.
        # RENDER advances them via state machine.

        # render_mask: RENDER advances on wall transitions.
        out_render_mask = select(
            token_flags["is_render"], render_out.next_render_mask, zero_mw
        )

        # render_col: SORTED sets to vis_lo; RENDER advances.
        out_render_col = select(
            token_flags["is_sorted"],
            sorted_out.vis_lo,
            select(token_flags["is_render"], render_out.next_col, zero_1),
        )

        # render_chunk_k: RENDER advances.
        out_render_chunk_k = select(
            token_flags["is_render"], render_out.next_chunk_k, zero_1
        )

        # render_tex_id: SORTED sets from wall; RENDER forwards.
        out_render_tex_id = select(
            token_flags["is_sorted"],
            sorted_out.sel_tex_id,
            select(token_flags["is_render"], render_out.next_tex_id, zero_1),
        )

        # render_vis_lo/hi: SORTED sets from wall; RENDER forwards.
        out_render_vis_lo = select(
            token_flags["is_sorted"],
            sorted_out.vis_lo,
            select(token_flags["is_render"], render_out.next_vis_lo, zero_1),
        )
        out_render_vis_hi = select(
            token_flags["is_sorted"],
            sorted_out.vis_hi,
            select(token_flags["is_render"], render_out.next_vis_hi, zero_1),
        )

        # render_wall_j_onehot: SORTED sets from wall; RENDER forwards.
        out_render_wall_j_onehot = select(
            token_flags["is_sorted"],
            sorted_out.sel_onehot,
            select(
                token_flags["is_render"],
                render_out.next_wall_j_onehot,
                zero_mw,
            ),
        )

        # --- Token type ---
        out_token_type = select(
            token_flags["is_render"],
            render_out.render_next_type,
            zero_8,
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

        out_eos_rx = select(token_flags["is_eos"], eos_out.resolved_x, zero_1)
        out_eos_ry = select(token_flags["is_eos"], eos_out.resolved_y, zero_1)
        out_eos_angle = select(token_flags["is_eos"], input_out.new_angle, zero_1)

    overlaid = {
        "token_type": out_token_type,
        "render_mask": out_render_mask,
        "render_col": out_render_col,
        "render_chunk_k": out_render_chunk_k,
        "render_tex_id": out_render_tex_id,
        "render_vis_lo": out_render_vis_lo,
        "render_vis_hi": out_render_vis_hi,
        "render_wall_j_onehot": out_render_wall_j_onehot,
    }
    overflow = {
        "pixels": out_pixels,
        "col": out_col,
        "start": out_start,
        "length": out_length,
        "done": out_done,
        "advance_wall": out_advance_wall,
        "sort_done": out_sort_done,
        "eos_resolved_x": out_eos_rx,
        "eos_resolved_y": out_eos_ry,
        "eos_new_angle": out_eos_angle,
    }
    return overlaid, overflow
