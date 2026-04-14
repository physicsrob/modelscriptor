"""Walls-as-tokens game graph: multi-phase renderer with runtime wall geometry.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → WALL×N → EOS → SORTED_WALL×N → RENDER×(W × H/rp)

Six token types (E8 spherical codes):

    TEX_COL (5)      Texture column pixel data.  Each token carries one
                     column of one texture (tex_h × 3 floats).  RENDER
                     tokens retrieve the right column via attention.
    INPUT (0)        Player state + movement inputs.  Controls are fed
                     only here; an attention head distributes velocity
                     and trig values to WALL and EOS positions.
    WALL (1)         Wall geometry (ax, ay, bx, by, tex_id).
                     Graph computes distance from player for sort score.
    EOS (2)          End of prefill.  Outputs E8_SORTED_WALL type to
                     seed the autoregressive sort loop.
    SORTED_WALL (3)  Autoregressive sort output.  attend_argmin_unmasked
                     finds the next closest unmasked wall.  The host
                     feeds each output back as the next input.
    RENDER (4)       Autoregressive render output.  Attention selects
                     the wall for this ray, parametric intersection +
                     full texture pipeline produces pixels.

Every cross-position data dependency flows through attention:
INPUT→WALL/EOS (velocity, trig), WALL→EOS (collision), EOS→SORTED
(resolved state), WALL→SORTED (argmin), SORTED→RENDER (wall for
column), TEX_COL→RENDER (texture pixels).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.doom.graph_constants import (
    E8_EOS,
    E8_INPUT,
    E8_RENDER,
    E8_SORTED_WALL,
    E8_TEX_COL,
    E8_WALL,
    TEX_E8_OFFSET,
)
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import select

from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.eos import EosInputs, build_eos
from torchwright.doom.stages.input import InputInputs, build_input
from torchwright.doom.stages.render import RenderInputs, build_render
from torchwright.doom.stages.sorted import SortedInputs, build_sorted
from torchwright.doom.stages.tex_col import TexColInputs, build_tex_col
from torchwright.doom.stages.wall import WallInputs, build_wall
from torchwright.reference_renderer.types import RenderConfig


# Re-exported for callers that still import these symbols from
# ``torchwright.doom.game_graph`` (tests, compile.py, external tools).
# The canonical home for the constants is ``graph_constants.py``.
__all__ = [
    "E8_EOS",
    "E8_INPUT",
    "E8_RENDER",
    "E8_SORTED_WALL",
    "E8_TEX_COL",
    "E8_WALL",
    "TEX_E8_OFFSET",
    "build_game_graph",
]


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------


def build_game_graph(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    rows_per_patch: Optional[int] = None,
) -> Tuple[Node, PosEncoding]:
    """Build the walls-as-tokens game graph.

    Collision detection is runtime: each WALL token tests the player's
    velocity ray against its wall segment and outputs three hit flags
    (full, x-only, y-only).  The host resolves wall sliding.

    Returns ``(output_node, pos_encoding)``.
    """
    H = config.screen_height
    W = config.screen_width
    fov = config.fov_columns
    tex_w, tex_h = textures[0].shape[0], textures[0].shape[1]
    rp = rows_per_patch if rows_per_patch is not None else H

    pos_encoding = create_pos_encoding()

    # --- Inputs (host-fed at every position) ---
    token_type = create_input("token_type", 8)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    player_angle = create_input("player_angle", 1)
    input_forward = create_input("input_forward", 1)
    input_backward = create_input("input_backward", 1)
    input_turn_left = create_input("input_turn_left", 1)
    input_turn_right = create_input("input_turn_right", 1)
    input_strafe_left = create_input("input_strafe_left", 1)
    input_strafe_right = create_input("input_strafe_right", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_tex_id = create_input("wall_tex_id", 1)
    wall_index = create_input("wall_index", 1)
    d_sort_out = 8 + 5 + 2 * max_walls
    sort_feedback = create_input("sort_feedback", d_sort_out)
    prev_mask = extract_from(
        sort_feedback, d_sort_out, 8 + 5 + max_walls, max_walls, "prev_mask",
    )
    col_idx = create_input("col_idx", 1)
    patch_idx = create_input("patch_idx", 1)
    tex_col_input = create_input("tex_col_input", 1)
    tex_pixels = create_input("tex_pixels", tex_h * 3)
    texture_id_e8 = create_input("texture_id_e8", 8)

    # --- Token type detection ---
    with annotate("token_type"):
        is_input = equals_vector(token_type, E8_INPUT)
        is_wall = equals_vector(token_type, E8_WALL)
        is_eos = equals_vector(token_type, E8_EOS)
        is_sorted = equals_vector(token_type, E8_SORTED_WALL)
        is_render = equals_vector(token_type, E8_RENDER)
        is_tex_col = equals_vector(token_type, E8_TEX_COL)

    # --- TEX_COL: column one-hot for key matching ---
    tc_onehot_01 = build_tex_col(
        TexColInputs(tex_col_input=tex_col_input), tex_w=tex_w,
    ).tc_onehot_01

    # =====================================================================
    # INPUT: game logic (angle update + velocity) + broadcast to all positions
    # =====================================================================
    input_out = build_input(
        InputInputs(
            player_angle=player_angle,
            input_turn_left=input_turn_left,
            input_turn_right=input_turn_right,
            input_forward=input_forward,
            input_backward=input_backward,
            input_strafe_left=input_strafe_left,
            input_strafe_right=input_strafe_right,
            is_input=is_input,
            pos_encoding=pos_encoding,
        ),
        turn_speed=turn_speed,
        move_speed=move_speed,
    )
    attn_vel_dx = input_out.vel_dx
    attn_vel_dy = input_out.vel_dy
    attn_move_cos = input_out.move_cos
    attn_move_sin = input_out.move_sin
    attn_new_angle = input_out.new_angle

    # =====================================================================
    # WALL: collision flags + sort score + packed render precomputation
    # =====================================================================
    wall_out = build_wall(
        WallInputs(
            wall_ax=wall_ax, wall_ay=wall_ay,
            wall_bx=wall_bx, wall_by=wall_by,
            wall_tex_id=wall_tex_id, wall_index=wall_index,
            player_x=player_x, player_y=player_y,
            is_wall=is_wall,
            vel_dx=input_out.vel_dx, vel_dy=input_out.vel_dy,
            move_cos=input_out.move_cos, move_sin=input_out.move_sin,
        ),
        config=config, max_walls=max_walls, max_coord=max_coord,
    )

    # =====================================================================
    # EOS: aggregate per-WALL collision flags + broadcast resolved state
    # =====================================================================
    eos_out = build_eos(EosInputs(
        is_wall=is_wall,
        is_eos=is_eos,
        collision=wall_out.collision,
        player_x=player_x,
        player_y=player_y,
        vel_dx=input_out.vel_dx,
        vel_dy=input_out.vel_dy,
        new_angle=input_out.new_angle,
        pos_encoding=pos_encoding,
    ))

    # =====================================================================
    # SORTED_WALL: argmin over WALLs + per-wall visibility mask
    # =====================================================================
    sorted_out = build_sorted(
        SortedInputs(
            sort_score=wall_out.sort_score,
            position_onehot=wall_out.position_onehot,
            sort_value=wall_out.sort_value,
            prev_mask=prev_mask,
            eos_px=eos_out.px,
            eos_py=eos_out.py,
            eos_angle=eos_out.angle,
            is_sorted=is_sorted,
            pos_encoding=pos_encoding,
        ),
        config=config, max_walls=max_walls, max_coord=max_coord,
    )

    # =====================================================================
    # RENDER: visibility-masked wall pick → height → texture → column fill
    # =====================================================================
    render_out = build_render(
        RenderInputs(
            col_idx=col_idx,
            patch_idx=patch_idx,
            texture_id_e8=texture_id_e8,
            tex_pixels=tex_pixels,
            is_render=is_render,
            is_sorted=is_sorted,
            is_tex_col=is_tex_col,
            gated_render_data=sorted_out.gated_render_data,
            gated_vis_mask=sorted_out.gated_vis_mask,
            tc_onehot_01=tc_onehot_01,
            pos_encoding=pos_encoding,
        ),
        config=config, textures=textures,
        rows_per_patch=rp, max_coord=max_coord,
    )

    # =====================================================================
    # Output: gated by token type
    # =====================================================================
    output = _assemble_output(
        is_eos=is_eos,
        is_sorted=is_sorted,
        is_render=is_render,
        is_tex_col=is_tex_col,
        sel_wall_data=sorted_out.sel_wall_data,
        sel_onehot=sorted_out.sel_onehot,
        updated_mask=sorted_out.updated_mask,
        pixels=render_out.pixels,
        resolved_x=eos_out.resolved_x,
        resolved_y=eos_out.resolved_y,
        new_angle=input_out.new_angle,
        rows_per_patch=rp,
        max_walls=max_walls,
    )
    return output, pos_encoding


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def _assemble_output(
    *,
    is_eos: Node,
    is_sorted: Node,
    is_render: Node,
    is_tex_col: Node,
    sel_wall_data: Node,
    sel_onehot: Node,
    updated_mask: Node,
    pixels: Node,
    resolved_x: Node,
    resolved_y: Node,
    new_angle: Node,
    rows_per_patch: int,
    max_walls: int,
) -> Node:
    """Token-type-gated output: pad each per-phase payload to a common width and select."""
    with annotate("output"):
        # SORTED_WALL output: type + wall data + onehot + updated mask.
        sort_output = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="sort_type"),
            sel_wall_data, sel_onehot, updated_mask,
        ])

        # RENDER output: type + pixels.
        render_output = Concatenate([
            create_literal_value(E8_RENDER, name="render_type"),
            pixels,
        ])

        # INPUT output: type + padding (host ignores INPUT output).
        input_output = Concatenate([
            create_literal_value(E8_INPUT, name="input_type"),
            create_literal_value(torch.zeros(3), name="input_pad"),
        ])

        # EOS output: seeds the sort loop with E8_SORTED_WALL type + resolved
        # state at offsets 8-10 + zeros for the sort mask.
        eos_output = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="eos_sort_seed"),
            resolved_x, resolved_y, new_angle,
            create_literal_value(
                torch.zeros(2 + 2 * max_walls), name="eos_sort_pad"),
        ])

        # TEX_COL output: type + padding (host ignores TEX_COL output).
        tex_col_output = Concatenate([
            create_literal_value(E8_TEX_COL, name="tex_col_type"),
            create_literal_value(torch.zeros(3), name="tc_pad"),
        ])

        # Pad all to the same width so we can select between them.
        d_sort_out = 8 + 5 + 2 * max_walls
        d_render_out = 8 + rows_per_patch * 3
        d_input_out = 8 + 3
        d_eos_out = d_sort_out  # EOS seeds the sort loop
        d_tc_out = 8 + 3
        d_out = max(d_sort_out, d_render_out, d_input_out, d_eos_out, d_tc_out)

        def _pad(node, cur_width):
            if cur_width >= d_out:
                return node
            return Concatenate([node, create_literal_value(
                torch.zeros(d_out - cur_width), name="pad")])

        sort_padded = _pad(sort_output, d_sort_out)
        render_padded = _pad(render_output, d_render_out)
        input_padded = _pad(input_output, d_input_out)
        eos_padded = _pad(eos_output, d_eos_out)
        tc_padded = _pad(tex_col_output, d_tc_out)

        inner1 = select(is_eos, eos_padded, input_padded)
        inner2 = select(is_tex_col, tc_padded, inner1)
        inner3 = select(is_sorted, sort_padded, inner2)
        return select(is_render, render_padded, inner3)
