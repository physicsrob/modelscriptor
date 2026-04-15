"""Walls-as-tokens game graph — orchestrator.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → BSP_NODE×M → WALL×N →
    EOS → SORTED_WALL×N → (THINKING → RENDER×k)×N

Eight token types (E8 spherical codes):

    TEX_COL (5)      Texture column pixel data.  Each token carries one
                     column of one texture (tex_h × 3 floats).  RENDER
                     tokens retrieve the right column via attention.
    INPUT (0)        Player state + movement inputs.
    BSP_NODE (7)     One splitting plane of the BSP tree.  Each token
                     classifies the player as FRONT/BACK of its plane;
                     the result lands in a shared ``side_P_vec`` used
                     by WALL tokens to derive BSP rank.
    WALL (1)         Wall geometry + BSP rank precomputation + runtime
                     collision against the player's velocity + per-wall
                     render precompute.
    EOS (2)          End of prefill.  Resolves collisions and seeds the
                     autoregressive sort loop.
    SORTED_WALL (3)  Autoregressive sort output.  attend_argmin_unmasked
                     finds the next-closest wall by BSP rank.
    THINKING (6)     Wall selector hoisted out of RENDER — one per wall,
                     populates the render_feedback overlay that feeds
                     the next block of RENDER tokens.
    RENDER (4)       Autoregressive column fill.  A state machine
                     transitions through chunks, columns, and walls
                     and emits ``done`` when finished.

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
    create_input, create_literal_value, create_pos_encoding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import (
    E8_BSP_NODE, E8_EOS, E8_INPUT, E8_RENDER, E8_SORTED_WALL,
    E8_TEX_COL, E8_THINKING, E8_WALL, TEX_E8_OFFSET,
)

# Re-export token constants for legacy callers that import them from
# game_graph (compile.py, tooling, tests).  graph_constants is the
# authoritative definition; these re-exports are just a compatibility
# surface.
__all__ = [
    "GameGraphIO", "build_game_graph",
    "E8_INPUT", "E8_WALL", "E8_EOS", "E8_SORTED_WALL", "E8_RENDER",
    "E8_TEX_COL", "E8_THINKING", "E8_BSP_NODE",
    "TEX_E8_OFFSET",
]
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.bsp import BspInputs, build_bsp
from torchwright.doom.stages.eos import EosInputs, build_eos
from torchwright.doom.stages.input import InputInputs, build_input
from torchwright.doom.stages.render import RenderInputs, build_render
from torchwright.doom.stages.sorted import SortedInputs, build_sorted
from torchwright.doom.stages.tex_col import TexColInputs, build_tex_col
from torchwright.doom.stages.thinking import ThinkingInputs, build_thinking
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
        """Single concatenated output node for legacy callers.

        Order: token_type, sort_feedback, render_feedback, pixels, col,
        start, length, done.
        """
        return Concatenate([
            self.overlaid_outputs["token_type"],
            self.overlaid_outputs["sort_feedback"],
            self.overlaid_outputs["render_feedback"],
            self.overflow_outputs["pixels"],
            self.overflow_outputs["col"],
            self.overflow_outputs["start"],
            self.overflow_outputs["length"],
            self.overflow_outputs["done"],
        ])


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

    inputs, fb_fields = _create_inputs(
        max_walls=max_walls, tex_h=tex_h, max_bsp_nodes=max_bsp_nodes,
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
            wall_ax=inputs["wall_ax"], wall_ay=inputs["wall_ay"],
            wall_bx=inputs["wall_bx"], wall_by=inputs["wall_by"],
            wall_tex_id=inputs["wall_tex_id"],
            wall_index=inputs["wall_index"],
            player_x=inputs["player_x"], player_y=inputs["player_y"],
            is_wall=tf["is_wall"],
            vel_dx=input_out.vel_dx, vel_dy=input_out.vel_dy,
            move_cos=input_out.move_cos, move_sin=input_out.move_sin,
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
    eos_out = build_eos(EosInputs(
        is_wall=tf["is_wall"],
        is_eos=tf["is_eos"],
        collision=wall_out.collision,
        player_x=inputs["player_x"], player_y=inputs["player_y"],
        vel_dx=input_out.vel_dx, vel_dy=input_out.vel_dy,
        new_angle=input_out.new_angle,
        pos_encoding=pos_encoding,
    ))

    # ---------- SORTED ----------
    sorted_out = build_sorted(
        SortedInputs(
            sort_score=wall_out.sort_score,
            is_renderable=wall_out.is_renderable,
            position_onehot=wall_out.position_onehot,
            sort_value=wall_out.sort_value,
            prev_mask=fb_fields["prev_mask"],
            eos_px=eos_out.px, eos_py=eos_out.py, eos_angle=eos_out.angle,
            is_sorted=tf["is_sorted"],
            is_wall=tf["is_wall"],
            pos_encoding=pos_encoding,
        ),
        config=config,
        max_walls=max_walls,
        max_coord=max_coord,
    )

    # ---------- THINKING ----------
    thinking_out = build_thinking(
        ThinkingInputs(
            sort_rank=sorted_out.sort_rank,
            sort_rank_onehot=sorted_out.sort_rank_onehot,
            gated_render_data=sorted_out.gated_render_data,
            vis_lo=sorted_out.vis_lo,
            vis_hi=sorted_out.vis_hi,
            render_mask=fb_fields["render_mask"],
            is_sorted=tf["is_sorted"],
            pos_encoding=pos_encoding,
        ),
        max_walls=max_walls,
    )

    # ---------- RENDER ----------
    render_out = build_render(
        RenderInputs(
            render_mask=fb_fields["render_mask"],
            render_col=fb_fields["render_col"],
            render_is_new_wall=fb_fields["render_is_new_wall"],
            render_chunk_start=fb_fields["render_chunk_start"],
            fb_sort_den=fb_fields["fb_sort_den"],
            fb_C=fb_fields["fb_C"],
            fb_D=fb_fields["fb_D"],
            fb_E=fb_fields["fb_E"],
            fb_H_inv=fb_fields["fb_H_inv"],
            fb_tex_id=fb_fields["fb_tex_id"],
            fb_col_lo=fb_fields["fb_col_lo"],
            fb_col_hi=fb_fields["fb_col_hi"],
            fb_onehot=fb_fields["fb_onehot"],
            texture_id_e8=inputs["texture_id_e8"],
            tex_pixels=inputs["tex_pixels"],
            tc_onehot_01=tex_col_out.tc_onehot_01,
            is_render=tf["is_render"],
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
        fb_fields=fb_fields,
        eos_out=eos_out, input_out=input_out,
        sorted_out=sorted_out, thinking_out=thinking_out,
        render_out=render_out,
        max_walls=max_walls, chunk_size=cs,
    )

    return GameGraphIO(
        inputs=inputs,
        overlaid_outputs=overlaid,
        overflow_outputs=overflow,
    ), pos_encoding


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _create_inputs(
    *, max_walls: int, tex_h: int, max_bsp_nodes: int,
) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Create host-fed input nodes + unpack the overlaid feedback vectors.

    Feedback unpacking lives here because the layouts are shared with
    ``_assemble_output`` — keeping the two in the same module prevents
    silent drift.
    """
    inputs: Dict[str, Node] = {}

    inputs["token_type"] = create_input("token_type", 8)
    inputs["player_x"] = create_input("player_x", 1)
    inputs["player_y"] = create_input("player_y", 1)
    inputs["player_angle"] = create_input("player_angle", 1)
    for k in (
        "input_forward", "input_backward",
        "input_turn_left", "input_turn_right",
        "input_strafe_left", "input_strafe_right",
    ):
        inputs[k] = create_input(k, 1)
    for k in ("wall_ax", "wall_ay", "wall_bx", "wall_by", "wall_tex_id", "wall_index"):
        inputs[k] = create_input(k, 1)
    inputs["tex_col_input"] = create_input("tex_col_input", 1)
    inputs["tex_pixels"] = create_input("tex_pixels", tex_h * 3)
    inputs["texture_id_e8"] = create_input("texture_id_e8", 8)

    inputs["bsp_plane_nx"] = create_input("bsp_plane_nx", 1)
    inputs["bsp_plane_ny"] = create_input("bsp_plane_ny", 1)
    inputs["bsp_plane_d"] = create_input("bsp_plane_d", 1)
    inputs["bsp_node_id_onehot"] = create_input("bsp_node_id_onehot", max_bsp_nodes)
    inputs["wall_bsp_coeffs"] = create_input("wall_bsp_coeffs", max_bsp_nodes)
    inputs["wall_bsp_const"] = create_input("wall_bsp_const", 1)

    d_sort_out = 8 + 5 + 3 + 2 * max_walls
    sort_feedback = create_input("sort_feedback", d_sort_out)
    inputs["sort_feedback"] = sort_feedback

    d_render_fb = 2 * max_walls + 11
    render_feedback = create_input("render_feedback", d_render_fb)
    inputs["render_feedback"] = render_feedback

    # Feedback field layout (must stay in sync with _assemble_output).
    fields: Dict[str, Node] = {
        "prev_mask": extract_from(
            sort_feedback, d_sort_out, 8 + 5 + 3 + max_walls, max_walls, "prev_mask",
        ),
        "render_mask": extract_from(
            render_feedback, d_render_fb, 0, max_walls, "render_mask",
        ),
        "render_col": extract_from(
            render_feedback, d_render_fb, max_walls, 1, "render_col",
        ),
        "render_is_new_wall": extract_from(
            render_feedback, d_render_fb, max_walls + 1, 1, "render_is_new_wall",
        ),
        "render_chunk_start": extract_from(
            render_feedback, d_render_fb, max_walls + 2, 1, "render_chunk_start",
        ),
        "fb_sort_den": extract_from(
            render_feedback, d_render_fb, max_walls + 3, 1, "fb_sort_den",
        ),
        "fb_C": extract_from(render_feedback, d_render_fb, max_walls + 4, 1, "fb_C"),
        "fb_D": extract_from(render_feedback, d_render_fb, max_walls + 5, 1, "fb_D"),
        "fb_E": extract_from(render_feedback, d_render_fb, max_walls + 6, 1, "fb_E"),
        "fb_H_inv": extract_from(render_feedback, d_render_fb, max_walls + 7, 1, "fb_H_inv"),
        "fb_tex_id": extract_from(render_feedback, d_render_fb, max_walls + 8, 1, "fb_tex_id"),
        "fb_col_lo": extract_from(render_feedback, d_render_fb, max_walls + 9, 1, "fb_col_lo"),
        "fb_col_hi": extract_from(render_feedback, d_render_fb, max_walls + 10, 1, "fb_col_hi"),
        "fb_onehot": extract_from(
            render_feedback, d_render_fb, max_walls + 11, max_walls, "fb_onehot",
        ),
    }
    return inputs, fields


def _detect_token_types(token_type: Node) -> Dict[str, Node]:
    """Derive per-type boolean flags from the 8-wide token_type vector."""
    with annotate("token_type"):
        return {
            "is_input": equals_vector(token_type, E8_INPUT),
            "is_wall": equals_vector(token_type, E8_WALL),
            "is_eos": equals_vector(token_type, E8_EOS),
            "is_sorted": equals_vector(token_type, E8_SORTED_WALL),
            "is_render": equals_vector(token_type, E8_RENDER),
            "is_thinking": equals_vector(token_type, E8_THINKING),
            "is_tex_col": equals_vector(token_type, E8_TEX_COL),
            "is_bsp_node": equals_vector(token_type, E8_BSP_NODE),
        }


def _assemble_output(
    *,
    token_flags: Dict[str, Node],
    fb_fields: Dict[str, Node],
    eos_out,
    input_out,
    sorted_out,
    thinking_out,
    render_out,
    max_walls: int,
    chunk_size: int,
) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Build the three overlaid outputs + five overflow outputs.

    Overlaid outputs fed back into the next step's input:
        token_type, sort_feedback, render_feedback
    Overflow outputs bitblitted by the host:
        pixels, col, start, length, done
    """
    d_sort_out = 8 + 5 + 3 + 2 * max_walls
    d_render_fb = 2 * max_walls + 11

    with annotate("output"):
        zero_1 = create_literal_value(torch.tensor([0.0]), name="zero_1")
        zero_8 = create_literal_value(torch.zeros(8), name="zero_8")
        zero_rf = create_literal_value(torch.zeros(d_render_fb), name="zero_rf")
        zero_sf = create_literal_value(torch.zeros(d_sort_out), name="zero_sf")
        zero_pixels = create_literal_value(
            torch.zeros(chunk_size * 3), name="zero_pixels",
        )
        pos_one = create_literal_value(torch.tensor([1.0]), name="pos_one_out")
        chunk_sentinel = create_literal_value(
            torch.tensor([-1.0]), name="chunk_sentinel_out",
        )

        # THINKING's render_feedback output: seed the next block of RENDER
        # tokens with the picked wall's data.  render_mask is forwarded
        # unchanged (the advance_wall bit was added by the RENDER token
        # that transitioned to THINKING, then fed back through the
        # overlay — so fb_fields["render_mask"] already includes it).
        thinking_render_fb = Concatenate([
            fb_fields["render_mask"],
            thinking_out.t_col_lo,     # render_col = first column of this wall
            pos_one,                   # is_new_wall = +1
            chunk_sentinel,            # chunk_start = -1 → start at wall_top
            thinking_out.t_sort_den, thinking_out.t_C, thinking_out.t_D,
            thinking_out.t_E, thinking_out.t_H_inv, thinking_out.t_tex_id,
            thinking_out.t_col_lo, thinking_out.t_col_hi, thinking_out.t_onehot,
        ])

        sort_feedback_out = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="sort_type"),
            sorted_out.sel_wall_data,
            sorted_out.sort_rank,
            sorted_out.vis_lo,
            sorted_out.vis_hi,
            sorted_out.sel_onehot,
            sorted_out.updated_mask,
        ])

        # EOS seeds the sort loop with a SORTED_WALL-type vector plus
        # resolved player pose.  The rest of the layout is zero-padded
        # since no wall has been picked yet.
        eos_sort_seed = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="eos_sort_seed"),
            eos_out.resolved_x, eos_out.resolved_y, input_out.new_angle,
            create_literal_value(
                torch.zeros(2 + 3 + 2 * max_walls), name="eos_sort_pad",
            ),
        ])

        # Next token type per source:
        #   THINKING → E8_RENDER (start rendering the picked wall)
        #   RENDER   → render_out.render_next_type (E8_THINKING or E8_RENDER)
        #   SORTED   → E8_SORTED_WALL
        #   EOS      → E8_SORTED_WALL
        out_token_type = select(
            token_flags["is_thinking"],
            create_literal_value(E8_RENDER, name="thinking_next_type"),
            select(
                token_flags["is_render"], render_out.render_next_type,
                select(
                    token_flags["is_sorted"],
                    create_literal_value(E8_SORTED_WALL, name="sort_next_type"),
                    select(
                        token_flags["is_eos"],
                        create_literal_value(E8_SORTED_WALL, name="eos_next_type"),
                        zero_8))))

        out_render_fb = select(
            token_flags["is_thinking"], thinking_render_fb,
            select(
                token_flags["is_render"], render_out.next_render_feedback,
                zero_rf))

        out_sort_fb = select(
            token_flags["is_sorted"], sort_feedback_out,
            select(token_flags["is_eos"], eos_sort_seed, zero_sf))

        out_pixels = select(token_flags["is_render"], render_out.pixels, zero_pixels)
        out_col = select(token_flags["is_render"], render_out.active_col, zero_1)
        out_start = select(token_flags["is_render"], render_out.active_start, zero_1)
        out_length = select(token_flags["is_render"], render_out.chunk_length, zero_1)
        out_done = select(token_flags["is_render"], render_out.done_flag, zero_1)

    overlaid = {
        "token_type": out_token_type,
        "sort_feedback": out_sort_fb,
        "render_feedback": out_render_fb,
    }
    overflow = {
        "pixels": out_pixels,
        "col": out_col,
        "start": out_start,
        "length": out_length,
        "done": out_done,
    }
    return overlaid, overflow
