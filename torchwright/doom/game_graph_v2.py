"""Walls-as-tokens game graph: multi-phase renderer with runtime wall geometry.

Token sequence per frame:

    START → WALL×N → EOS → SORTED_WALL×N → RENDER×(W × H/rp)

Five token types (E8 spherical codes):

    START (0)        Player state + movement inputs.  Game logic
                     (angle update, velocity) runs here.
    WALL (1)         Wall geometry (ax, ay, bx, by, tex_id).
                     Graph computes distance from player for sort score.
    EOS (2)          End of prefill.
    SORTED_WALL (3)  Autoregressive sort output.  attend_argmin_unmasked
                     finds the next closest unmasked wall.
    RENDER (4)       Autoregressive render output.  Attention selects
                     the wall for this ray, parametric intersection +
                     full texture pipeline produces pixels.

The host feeds player state at every position (no graph-side
get_prev_value) and wall indices at WALL positions (no prefix_sum).
The only cross-position attention in the graph is the sort head
(attend_argmin_unmasked) and the render head (angular-similarity
wall selection).  This keeps the critical path short: ~12 layers
overhead vs the baked renderer at N=4, constant in N.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from torchwright.graph import Attn, Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    compare,
    mod_const,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    square_signed,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, equals_vector
from torchwright.ops.map_select import in_range, select

from torchwright.doom.renderer import (
    _textured_column_fill,
    _u_norm_lookup,
    _wall_height_lookup,
    trig_lookup,
)
from torchwright.reference_renderer.types import RenderConfig


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

TOKEN_START = 0
TOKEN_WALL = 1
TOKEN_EOS = 2
TOKEN_SORTED_WALL = 3
TOKEN_RENDER = 4

E8_START = index_to_vector(TOKEN_START)
E8_WALL = index_to_vector(TOKEN_WALL)
E8_EOS = index_to_vector(TOKEN_EOS)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_RENDER = index_to_vector(TOKEN_RENDER)


# ---------------------------------------------------------------------------
# Breakpoints for piecewise_linear_2d products
# ---------------------------------------------------------------------------

_DIFF_BP = [
    -40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5,
    0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40,
]
_TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
_SQRT_BP = [0, 0.25, 1, 2, 4, 9, 16, 25, 36, 49, 64, 100, 225, 400, 900, 1600, 3200]

BIG_DISTANCE = 1000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_from(node: Node, d_total: int, start: int, width: int, name: str) -> Node:
    """Extract ``width`` columns starting at ``start`` from a ``d_total``-wide node."""
    m = torch.zeros(d_total, width)
    for i in range(width):
        m[start + i, i] = 1.0
    return Linear(node, m, name=name)


# ---------------------------------------------------------------------------
# Game logic (segment-independent, reused from game_graph.py)
# ---------------------------------------------------------------------------


def _compute_new_angle(
    old_angle: Node,
    input_turn_left: Node,
    input_turn_right: Node,
    turn_speed: int,
) -> Node:
    """new_angle = (old_angle + turn_right*speed - turn_left*speed) % 256"""
    turn_r = multiply_const(input_turn_right, float(turn_speed))
    turn_l = multiply_const(input_turn_left, float(turn_speed))
    turn_delta = subtract(turn_r, turn_l)
    raw_angle = add(old_angle, turn_delta)
    shifted = add_const(raw_angle, 256.0)
    return mod_const(shifted, 256, 512 + turn_speed)


def _compute_velocity(
    new_angle: Node,
    input_forward: Node,
    input_backward: Node,
    input_strafe_left: Node,
    input_strafe_right: Node,
    move_speed: float,
) -> Tuple[Node, Node]:
    """Compute (dx, dy) from player inputs and facing angle."""
    move_cos, move_sin = trig_lookup(new_angle)
    speed_cos = multiply_const(move_cos, move_speed)
    speed_sin = multiply_const(move_sin, move_speed)
    neg_speed_cos = negate(speed_cos)
    neg_speed_sin = negate(speed_sin)
    zero = LiteralValue(torch.tensor([0.0]), name="zero_vel")
    is_fwd = compare(input_forward, 0.5)
    is_bwd = compare(input_backward, 0.5)
    is_sl = compare(input_strafe_left, 0.5)
    is_sr = compare(input_strafe_right, 0.5)
    dx = add(
        add(select(is_fwd, speed_cos, zero), select(is_bwd, neg_speed_cos, zero)),
        add(select(is_sl, speed_sin, zero), select(is_sr, neg_speed_sin, zero)),
    )
    dy = add(
        add(select(is_fwd, speed_sin, zero), select(is_bwd, neg_speed_sin, zero)),
        add(select(is_sl, neg_speed_cos, zero), select(is_sr, speed_cos, zero)),
    )
    return dx, dy


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------


def build_game_graph_v2(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_walls: int = 8,
    max_coord: float = 20.0,
    move_speed: float = 0.3,
    turn_speed: int = 4,
    rows_per_patch: Optional[int] = None,
) -> Tuple[Node, PosEncoding]:
    """Build the walls-as-tokens game graph.

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
    sort_mask = create_input("sort_mask", max_walls)
    col_idx = create_input("col_idx", 1)
    patch_idx = create_input("patch_idx", 1)

    # --- Token type detection ---
    is_start = equals_vector(token_type, E8_START)
    is_wall = equals_vector(token_type, E8_WALL)
    is_sorted = equals_vector(token_type, E8_SORTED_WALL)
    is_render = equals_vector(token_type, E8_RENDER)

    # =====================================================================
    # START: game logic (angle update + velocity, no collision)
    # =====================================================================

    new_angle = _compute_new_angle(
        player_angle, input_turn_left, input_turn_right, turn_speed,
    )
    vel_dx, vel_dy = _compute_velocity(
        new_angle, input_forward, input_backward,
        input_strafe_left, input_strafe_right, move_speed,
    )
    # Apply velocity directly (no collision detection for now).
    # At START: new_x = player_x + dx, new_y = player_y + dy.
    # At non-START: inputs are zero → dx=dy=0 → no-op.
    new_x = add(player_x, vel_dx)
    new_y = add(player_y, vel_dy)

    # =====================================================================
    # WALL: compute distance score + angular key + wall value
    # =====================================================================

    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    w_dx = subtract(mid_x, player_x)
    w_dy = subtract(mid_y, player_y)
    dx_sq = square_signed(w_dx, max_abs=40.0, step=1.0)
    dy_sq = square_signed(w_dy, max_abs=40.0, step=1.0)
    dist_sq = add(dx_sq, dy_sq)
    wall_dist = piecewise_linear(
        dist_sq, _SQRT_BP,
        lambda x: math.sqrt(max(0, x)), name="wall_dist",
    )

    sentinel = create_literal_value(torch.tensor([99.0]), name="sentinel")
    sort_score = select(is_wall, wall_dist, sentinel)

    # Wall index one-hot (host-fed wall_index: 0, 1, 2, ...)
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Pack wall value: geometry + angular info + onehot
    wall_value_for_sort = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        w_dx, w_dy, wall_dist,
        position_onehot,
    ])
    d_sort_val = 8 + max_walls

    # =====================================================================
    # SORTED_WALL: attend_argmin_unmasked
    # =====================================================================

    selected_sort = attend_argmin_unmasked(
        pos_encoding=pos_encoding,
        score=sort_score,
        mask_vector=sort_mask,
        position_onehot=position_onehot,
        value=wall_value_for_sort,
    )

    sel_wall_data = _extract_from(selected_sort, d_sort_val, 0, 5, "sel_wall_data")
    sel_dx = _extract_from(selected_sort, d_sort_val, 5, 1, "sel_dx")
    sel_dy = _extract_from(selected_sort, d_sort_val, 6, 1, "sel_dy")
    sel_dist = _extract_from(selected_sort, d_sort_val, 7, 1, "sel_dist")
    sel_onehot = _extract_from(selected_sort, d_sort_val, 8, max_walls, "sel_onehot")

    # Gate sorted values: zero at non-sorted positions
    zeros_1 = create_literal_value(torch.zeros(1), name="z1")
    zeros_5 = create_literal_value(torch.zeros(5), name="z5")
    gated_dx = select(is_sorted, sel_dx, zeros_1)
    gated_dy = select(is_sorted, sel_dy, zeros_1)
    gated_dist = select(is_sorted, sel_dist, zeros_1)
    gated_wall_data = select(is_sorted, sel_wall_data, zeros_5)

    # =====================================================================
    # RENDER: attend to sorted walls, parametric intersection, pixels
    # =====================================================================

    # Ray angle from col_idx + player_angle
    col_times_fov = multiply_const(col_idx, float(fov))
    ao_raw = piecewise_linear(
        col_times_fov,
        [float(i) for i in range(0, fov * W + 1, max(1, W))],
        lambda x: float(int(x) // W),
        name="ao_raw",
    )
    angle_offset = add_const(ao_raw, float(-(fov // 2)))
    ray_angle_raw = add(player_angle, angle_offset)
    ray_angle_shifted = add_const(ray_angle_raw, 256.0)
    ray_angle = mod_const(ray_angle_shifted, 256, 512 + fov)
    ray_cos, ray_sin = trig_lookup(ray_angle)

    perp_shifted = add_const(angle_offset, 256.0)
    perp_angle = mod_const(perp_shifted, 256, 256 + fov)
    perp_cos, _perp_sin = trig_lookup(perp_angle)

    # --- Render attention: angular-similarity wall selection ---
    RENDER_GAIN = 80.0
    WALL_BIAS = 30.0
    DIST_SCALE = 1.0

    gated_ray_cos = select(is_render, ray_cos, zeros_1)
    gated_ray_sin = select(is_render, ray_sin, zeros_1)
    is_render_01 = multiply_const(add_const(is_render, 1.0), 0.5)
    is_sorted_01 = multiply_const(add_const(is_sorted, 1.0), 0.5)

    render_attn_in = Concatenate([
        pos_encoding,
        gated_ray_cos, gated_ray_sin, is_render_01,
        gated_dx, gated_dy, gated_dist, is_sorted_01,
        gated_wall_data,
    ])

    d_pe = len(pos_encoding)
    s_ray_cos = d_pe
    s_ray_sin = d_pe + 1
    s_is_render = d_pe + 2
    s_dx = d_pe + 3
    s_dy = d_pe + 4
    s_dist = d_pe + 5
    s_is_sorted = d_pe + 6
    s_wall_data = d_pe + 7

    d_head_render = 3 + 5

    q_matrix = torch.zeros(len(render_attn_in), d_head_render)
    q_matrix[s_ray_cos, 0] = RENDER_GAIN
    q_matrix[s_ray_sin, 1] = RENDER_GAIN
    q_matrix[s_is_render, 2] = RENDER_GAIN

    k_matrix = torch.zeros(len(render_attn_in), d_head_render)
    k_matrix[s_dx, 0] = 1.0
    k_matrix[s_dy, 1] = 1.0
    k_matrix[s_is_sorted, 2] = WALL_BIAS
    k_matrix[s_dist, 2] = -DIST_SCALE

    v_matrix = torch.zeros(len(render_attn_in), d_head_render)
    for i in range(5):
        v_matrix[s_wall_data + i, 3 + i] = 1.0

    o_matrix = torch.zeros(d_head_render, 5)
    for i in range(5):
        o_matrix[3 + i, i] = 1.0

    render_attn = Attn(
        query_in=render_attn_in, key_in=render_attn_in,
        value_in=render_attn_in,
        query_matrix=q_matrix, key_matrix=k_matrix,
        value_matrix=v_matrix, output_matrix=o_matrix,
    )

    r_wall_ax = _extract_from(render_attn, 5, 0, 1, "r_ax")
    r_wall_ay = _extract_from(render_attn, 5, 1, 1, "r_ay")
    r_wall_bx = _extract_from(render_attn, 5, 2, 1, "r_bx")
    r_wall_by = _extract_from(render_attn, 5, 3, 1, "r_by")
    r_wall_tex = _extract_from(render_attn, 5, 4, 1, "r_tex")

    # --- Parametric intersection ---
    ex = subtract(r_wall_bx, r_wall_ax)
    ey = subtract(r_wall_by, r_wall_ay)
    dx_r = subtract(r_wall_ax, player_x)
    dy_r = subtract(player_y, r_wall_ay)

    ey_cos = piecewise_linear_2d(ey, ray_cos, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_ey_cos")
    ex_sin = piecewise_linear_2d(ex, ray_sin, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_ex_sin")
    den = subtract(ey_cos, ex_sin)

    ey_dx = piecewise_linear_2d(ey, dx_r, _DIFF_BP, _DIFF_BP, lambda a,b: a*b, name="r_ey_dx")
    ex_dy = piecewise_linear_2d(ex, dy_r, _DIFF_BP, _DIFF_BP, lambda a,b: a*b, name="r_ex_dy")
    num_t = add(ey_dx, ex_dy)

    dx_sin = piecewise_linear_2d(dx_r, ray_sin, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_dx_sin")
    dy_cos = piecewise_linear_2d(dy_r, ray_cos, _DIFF_BP, _TRIG_BP, lambda a,b: a*b, name="r_dy_cos")
    num_u = add(dx_sin, dy_cos)

    # Den → angle data
    sign_den = compare(den, 0.0)
    abs_den = abs(den)
    inv_abs_den = reciprocal(abs_den, min_value=0.01, max_value=2.0 * max_coord)
    signed_inv_den = select(sign_den, inv_abs_den, negate(inv_abs_den))

    # Distance + validity
    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))
    is_den_nz = compare(abs_den, 0.05)
    is_t_pos = compare(adj_num_t, 0.05)
    is_u_ge0 = compare(adj_num_u, -0.05)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -0.05)
    abs_inv = select(sign_den, signed_inv_den, negate(signed_inv_den))
    dist_r = signed_multiply(
        adj_num_t, abs_inv,
        max_abs1=2.0 * max_coord * max_coord,
        max_abs2=1.0 / 0.01,
        step=1.0, max_abs_output=BIG_DISTANCE,
    )
    is_valid = bool_all_true([is_den_nz, is_t_pos, is_u_ge0, is_u_le_den])
    big = create_literal_value(torch.tensor([BIG_DISTANCE]), name="big")
    dist_r = select(is_valid, dist_r, big)

    # Wall height
    wall_top, wall_bottom, wall_height = _wall_height_lookup(
        dist_r, perp_cos, config, max_coord,
    )

    # Texture column
    tex_col_idx = _u_norm_lookup(adj_num_u, abs_den, tex_w, max_coord)
    num_tex = len(textures)
    n_keys = num_tex * tex_w
    flat_key = add(multiply_const(r_wall_tex, float(tex_w)), tex_col_idx)

    def _tex_col_vals(flat_idx):
        k = int(round(flat_idx))
        if 0 <= k < n_keys:
            tid = k // tex_w
            col = k % tex_w
            return [float(v) for v in textures[tid][col].flatten()]
        return [0.0] * (tex_h * 3)

    tex_column_colors = piecewise_linear(
        flat_key, [float(k) for k in range(n_keys)],
        _tex_col_vals, name="tex_col_lookup",
    )

    # Column fill
    patch_row_start = multiply_const(patch_idx, float(rp))
    pixels = _textured_column_fill(
        wall_top, wall_bottom, wall_height,
        tex_column_colors, tex_h, config, max_coord=max_coord,
        patch_row_start=patch_row_start, rows_per_patch=rp,
    )

    # =====================================================================
    # Output: gated by token type
    # =====================================================================

    # SORTED_WALL output: type + wall data + onehot
    sort_output = Concatenate([
        create_literal_value(E8_SORTED_WALL, name="sort_type"),
        sel_wall_data,
        sel_onehot,
    ])

    # RENDER output: type + pixels
    render_output = Concatenate([
        create_literal_value(E8_RENDER, name="render_type"),
        pixels,
    ])

    # START output: type + new state
    start_output = Concatenate([
        create_literal_value(E8_START, name="start_type"),
        new_x, new_y, new_angle,
    ])

    # Pad all to same width and select
    d_sort_out = 8 + 5 + max_walls
    d_render_out = 8 + rp * 3
    d_start_out = 8 + 3
    d_out = max(d_sort_out, d_render_out, d_start_out)

    def _pad(node, cur_width):
        if cur_width >= d_out:
            return node
        return Concatenate([node, create_literal_value(
            torch.zeros(d_out - cur_width), name="pad")])

    sort_padded = _pad(sort_output, d_sort_out)
    render_padded = _pad(render_output, d_render_out)
    start_padded = _pad(start_output, d_start_out)

    # Two-level select: render vs (sort vs start)
    inner = select(is_sorted, sort_padded, start_padded)
    output = select(is_render, render_padded, inner)

    return output, pos_encoding
