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
    thermometer_floor_div,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, bool_any_true, equals_vector
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
# Collision intersection (runtime, per-WALL-token)
# ---------------------------------------------------------------------------

# Breakpoints for velocity-range products.
_VEL_BP = [-0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.7]


def _collision_validity(den: Node, num_t: Node, num_u: Node) -> Node:
    """Check ray-segment intersection validity from (den, num_t, num_u).

    Returns a boolean node: 1.0 if hit, -1.0 if miss.
    """
    epsilon = 0.05
    sign_den = compare(den, 0.0)
    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))
    abs_den = select(sign_den, den, negate(den))

    is_den_ok = compare(abs_den, epsilon)
    is_t_pos = compare(adj_num_t, epsilon)
    t_margin = subtract(abs_den, adj_num_t)
    is_t_le_den = compare(t_margin, -epsilon)
    is_u_ge_0 = compare(adj_num_u, -epsilon)
    u_margin = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_margin, -epsilon)

    return bool_all_true(
        [is_den_ok, is_t_pos, is_t_le_den, is_u_ge_0, is_u_le_den]
    )


def _build_wall_collision(
    vel_dx: Node,
    vel_dy: Node,
    player_x: Node,
    player_y: Node,
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    is_wall: Node,
) -> Tuple[Node, Node, Node]:
    """Compute per-wall collision hit flags for three rays.

    All inputs are runtime (host-fed).  Six ``piecewise_linear_2d``
    calls produce the shared products; three validity checks (full,
    x-only, y-only) derive den/num_t/num_u from those products.

    Returns ``(hit_full, hit_x, hit_y)`` — each 1.0 (hit) or -1.0
    (miss), gated to -1.0 at non-WALL positions.
    """
    # Wall edge vectors (subtract, free)
    ex = subtract(wall_bx, wall_ax)
    ey = subtract(wall_by, wall_ay)

    # Player-to-wall-start vectors (subtract, free)
    dax = subtract(wall_ax, player_x)
    day = subtract(wall_ay, player_y)

    # --- 6 shared products (each 1 MLP sublayer) ---
    p_dx_ey = piecewise_linear_2d(vel_dx, ey, _VEL_BP, _DIFF_BP,
                                   lambda a, b: a * b, name="c_dx_ey")
    p_dy_ex = piecewise_linear_2d(vel_dy, ex, _VEL_BP, _DIFF_BP,
                                   lambda a, b: a * b, name="c_dy_ex")
    p_dax_ey = piecewise_linear_2d(dax, ey, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="c_dax_ey")
    p_day_ex = piecewise_linear_2d(day, ex, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="c_day_ex")
    p_dax_dy = piecewise_linear_2d(dax, vel_dy, _DIFF_BP, _VEL_BP,
                                    lambda a, b: a * b, name="c_dax_dy")
    p_day_dx = piecewise_linear_2d(day, vel_dx, _DIFF_BP, _VEL_BP,
                                    lambda a, b: a * b, name="c_day_dx")

    # --- Shared num_t (same for all three rays) ---
    num_t = subtract(p_dax_ey, p_day_ex)

    # --- Full ray: (vel_dx, vel_dy) ---
    den_full = subtract(p_dx_ey, p_dy_ex)
    num_u_full = subtract(p_dax_dy, p_day_dx)
    hit_full_raw = _collision_validity(den_full, num_t, num_u_full)

    # --- X-only ray: (vel_dx, 0) ---
    den_x = p_dx_ey
    num_u_x = negate(p_day_dx)
    hit_x_raw = _collision_validity(den_x, num_t, num_u_x)

    # --- Y-only ray: (0, vel_dy) ---
    den_y = negate(p_dy_ex)
    num_u_y = p_dax_dy
    hit_y_raw = _collision_validity(den_y, num_t, num_u_y)

    # Gate: -1.0 (no hit) at non-WALL positions
    no_hit = create_literal_value(torch.tensor([-1.0]), name="no_hit")
    hit_full = select(is_wall, hit_full_raw, no_hit)
    hit_x = select(is_wall, hit_x_raw, no_hit)
    hit_y = select(is_wall, hit_y_raw, no_hit)

    return hit_full, hit_x, hit_y


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
    sort_mask = create_input("sort_mask", max_walls)
    col_idx = create_input("col_idx", 1)
    patch_idx = create_input("patch_idx", 1)

    # --- Token type detection ---
    is_start = equals_vector(token_type, E8_START)
    is_wall = equals_vector(token_type, E8_WALL)
    is_eos = equals_vector(token_type, E8_EOS)
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
    # =====================================================================
    # WALL: distance score + sort value + collision hit flags
    # =====================================================================

    # Runtime collision: test velocity ray against this wall's geometry.
    # The host feeds real player inputs at WALL positions so vel_dx/vel_dy
    # are correct here (same values as at START).
    hit_full, hit_x, hit_y = _build_wall_collision(
        vel_dx, vel_dy, player_x, player_y,
        wall_ax, wall_ay, wall_bx, wall_by, is_wall,
    )

    mid_x = multiply_const(add(wall_ax, wall_bx), 0.5)
    mid_y = multiply_const(add(wall_ay, wall_by), 0.5)
    w_dx = subtract(mid_x, player_x)
    w_dy = subtract(mid_y, player_y)

    # --- Central ray intersection distance for sort score ---
    # Compute intersection of the central ray (player's viewing direction)
    # with this wall segment's extended line.  This gives front-to-back
    # ordering that matches the reference renderer.
    #
    # Using the same intersection math as the RENDER phase:
    #   ex, ey = edge vector (B - A)
    #   fx = ax - px, gy = py - ay (sign convention matches RENDER)
    #   den = ey*cos - ex*sin
    #   num_t = ey*fx + ex*gy
    #   t = num_t / den  (positive = in front)
    move_cos, move_sin = trig_lookup(new_angle)
    w_ex = subtract(wall_bx, wall_ax)
    w_ey = subtract(wall_by, wall_ay)
    w_fx = subtract(wall_ax, player_x)
    w_gy = subtract(player_y, wall_ay)

    sort_ey_cos = piecewise_linear_2d(
        w_ey, move_cos, _DIFF_BP, _TRIG_BP,
        lambda a, b: a * b, name="sort_ey_cos",
    )
    sort_ex_sin = piecewise_linear_2d(
        w_ex, move_sin, _DIFF_BP, _TRIG_BP,
        lambda a, b: a * b, name="sort_ex_sin",
    )
    sort_den = subtract(sort_ey_cos, sort_ex_sin)

    sort_ey_fx = piecewise_linear_2d(
        w_ey, w_fx, _DIFF_BP, _DIFF_BP,
        lambda a, b: a * b, name="sort_ey_fx",
    )
    sort_ex_gy = piecewise_linear_2d(
        w_ex, w_gy, _DIFF_BP, _DIFF_BP,
        lambda a, b: a * b, name="sort_ex_gy",
    )
    sort_num_t = add(sort_ey_fx, sort_ex_gy)

    sort_sign_den = compare(sort_den, 0.0)
    sort_abs_den = abs(sort_den)
    sort_adj_num_t = select(sort_sign_den, sort_num_t, negate(sort_num_t))

    is_sort_den_nz = compare(sort_abs_den, 0.05)
    is_sort_t_pos = compare(sort_adj_num_t, 0.0)

    sort_inv_den = reciprocal(sort_abs_den, min_value=0.01, max_value=2.0 * max_coord)
    sort_t = signed_multiply(
        sort_adj_num_t, sort_inv_den,
        max_abs1=2.0 * max_coord * max_coord,
        max_abs2=1.0 / 0.01,
        step=1.0, max_abs_output=BIG_DISTANCE,
    )

    is_sort_valid = bool_all_true([is_sort_den_nz, is_sort_t_pos])
    sentinel = create_literal_value(torch.tensor([99.0]), name="sentinel")
    center_ray_dist = select(is_sort_valid, sort_t, sentinel)
    sort_score = select(is_wall, center_ray_dist, sentinel)

    # Wall index one-hot (host-fed wall_index: 0, 1, 2, ...)
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Pack wall value: geometry + angular info + onehot
    wall_value_for_sort = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        w_dx, w_dy, center_ray_dist,
        position_onehot,
    ])
    d_sort_val = 8 + max_walls

    # =====================================================================
    # EOS: attend to WALL positions, aggregate collision, resolve position
    # =====================================================================

    # Remap hit flags from {-1, +1} to {0, 1} for clean averaging
    hit_full_01 = multiply_const(add_const(hit_full, 1.0), 0.5)
    hit_x_01 = multiply_const(add_const(hit_x, 1.0), 0.5)
    hit_y_01 = multiply_const(add_const(hit_y, 1.0), 0.5)

    is_eos_01 = multiply_const(add_const(is_eos, 1.0), 0.5)
    is_wall_01 = multiply_const(add_const(is_wall, 1.0), 0.5)

    resolve_attn_in = Concatenate([
        pos_encoding,
        is_eos_01, is_wall_01,
        hit_full_01, hit_x_01, hit_y_01,
    ])

    d_pe_r = len(pos_encoding)
    s_eos_01 = d_pe_r
    s_wall_01 = d_pe_r + 1
    s_hf = d_pe_r + 2
    s_hx = d_pe_r + 3
    s_hy = d_pe_r + 4

    RESOLVE_GAIN = 80.0
    d_head_resolve = 1 + 3

    q_resolve = torch.zeros(len(resolve_attn_in), d_head_resolve)
    q_resolve[s_eos_01, 0] = RESOLVE_GAIN

    k_resolve = torch.zeros(len(resolve_attn_in), d_head_resolve)
    k_resolve[s_wall_01, 0] = 1.0

    v_resolve = torch.zeros(len(resolve_attn_in), d_head_resolve)
    v_resolve[s_hf, 1] = 1.0
    v_resolve[s_hx, 2] = 1.0
    v_resolve[s_hy, 3] = 1.0

    o_resolve = torch.zeros(d_head_resolve, 3)
    o_resolve[1, 0] = 1.0
    o_resolve[2, 1] = 1.0
    o_resolve[3, 2] = 1.0

    resolve_attn = Attn(
        query_in=resolve_attn_in, key_in=resolve_attn_in,
        value_in=resolve_attn_in,
        query_matrix=q_resolve, key_matrix=k_resolve,
        value_matrix=v_resolve, output_matrix=o_resolve,
    )

    # Threshold averaged flags: any value > 0 means at least one wall hit.
    # Worst case (1 hit out of max_walls=8) → avg = 0.125 >> 0.05.
    avg_hf = _extract_from(resolve_attn, 3, 0, 1, "avg_hf")
    avg_hx = _extract_from(resolve_attn, 3, 1, 1, "avg_hx")
    avg_hy = _extract_from(resolve_attn, 3, 2, 1, "avg_hy")

    any_hit_full = compare(avg_hf, 0.05)
    any_hit_x = compare(avg_hx, 0.05)
    any_hit_y = compare(avg_hy, 0.05)

    use_new_x = bool_any_true([negate(any_hit_full), negate(any_hit_x)])
    use_new_y = bool_any_true([negate(any_hit_full), negate(any_hit_y)])

    new_x = add(player_x, vel_dx)
    new_y = add(player_y, vel_dy)
    resolved_x = select(use_new_x, new_x, player_x)
    resolved_y = select(use_new_y, new_y, player_y)

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
    zeros_W = create_literal_value(torch.zeros(W), name="z_W")
    gated_wall_data = select(is_sorted, sel_wall_data, zeros_5)
    gated_dist = select(is_sorted, sel_dist, zeros_1)

    # --- Visibility mask: column range where this wall is visible ---
    # Compute atan2 of each wall endpoint relative to the player, then
    # convert to column indices.  in_range produces a W-wide ±1 mask.
    #
    # For endpoint Q at (qx, qy), relative to player at (px, py):
    #   dqx = qx - px,  dqy = qy - py
    #   angle_Q = atan2(dqy, dqx) in [0, 255]
    #   relative angle = angle_Q - player_angle  (centered on view dir)
    #   col_Q = (relative_angle + fov/2) * W / fov

    sel_ax = _extract_from(sel_wall_data, 5, 0, 1, "sel_ax")
    sel_ay = _extract_from(sel_wall_data, 5, 1, 1, "sel_ay")
    sel_bx = _extract_from(sel_wall_data, 5, 2, 1, "sel_bx")
    sel_by = _extract_from(sel_wall_data, 5, 3, 1, "sel_by")

    dax = subtract(sel_ax, player_x)
    day = subtract(sel_ay, player_y)
    dbx = subtract(sel_bx, player_x)
    dby = subtract(sel_by, player_y)

    # Compute column index where each endpoint projects, using the
    # decomposition: cross/dot with the viewing direction gives the
    # tangent of the relative angle, then 1D atan maps to column.
    #
    #   cross_A = cos(PA)*day - sin(PA)*dax  (perpendicular component)
    #   dot_A   = cos(PA)*dax + sin(PA)*day  (parallel component)
    #   tan_rel_A = cross_A / dot_A
    #   col_A = atan(tan_rel_A) * W/fov_rad + W/2
    #
    # For the FOV, fov_rad = fov * 2*pi/256.  Walls behind the player
    # (dot < 0) get column indices outside [0, W], which in_range
    # correctly excludes.
    sort_cos, sort_sin = trig_lookup(new_angle)

    cross_a = subtract(
        piecewise_linear_2d(sort_cos, day, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="cos_day_a"),
        piecewise_linear_2d(sort_sin, dax, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="sin_dax_a"),
    )
    dot_a = add(
        piecewise_linear_2d(sort_cos, dax, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="cos_dax_a"),
        piecewise_linear_2d(sort_sin, day, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="sin_day_a"),
    )
    cross_b = subtract(
        piecewise_linear_2d(sort_cos, dby, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="cos_dby_b"),
        piecewise_linear_2d(sort_sin, dbx, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="sin_dbx_b"),
    )
    dot_b = add(
        piecewise_linear_2d(sort_cos, dbx, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="cos_dbx_b"),
        piecewise_linear_2d(sort_sin, dby, _TRIG_BP, _DIFF_BP,
                            lambda a, b: a * b, name="sin_dby_b"),
    )

    # tan_rel = cross / dot.  Clamp dot away from 0 for stability.
    # Walls behind the player (dot < 0) get large |tan| → col far
    # outside [0, W], naturally excluded by in_range.
    dot_a_sign = compare(dot_a, 0.0)
    dot_a_abs = abs(dot_a)
    dot_a_clamped = select(
        compare(dot_a_abs, 0.1),
        dot_a_abs,
        create_literal_value(torch.tensor([0.1]), name="dot_min"),
    )
    inv_dot_a = reciprocal(dot_a_clamped, min_value=0.1, max_value=2.0 * max_coord)
    signed_inv_dot_a = select(dot_a_sign, inv_dot_a, negate(inv_dot_a))
    tan_a = signed_multiply(cross_a, signed_inv_dot_a,
                            max_abs1=max_coord, max_abs2=1.0 / 0.1,
                            step=0.5, max_abs_output=20.0)

    dot_b_sign = compare(dot_b, 0.0)
    dot_b_abs = abs(dot_b)
    dot_b_clamped = select(
        compare(dot_b_abs, 0.1),
        dot_b_abs,
        create_literal_value(torch.tensor([0.1]), name="dot_min_b"),
    )
    inv_dot_b = reciprocal(dot_b_clamped, min_value=0.1, max_value=2.0 * max_coord)
    signed_inv_dot_b = select(dot_b_sign, inv_dot_b, negate(inv_dot_b))
    tan_b = signed_multiply(cross_b, signed_inv_dot_b,
                            max_abs1=max_coord, max_abs2=1.0 / 0.1,
                            step=0.5, max_abs_output=20.0)

    # atan(tan_rel) → relative angle in discrete units, then → column
    # For small FOV, atan(x) ≈ x within the FOV range.  Use piecewise
    # linear for accuracy across [-20, 20] range.
    _ATAN_BP = [-20, -10, -5, -3, -2, -1.5, -1, -0.75, -0.5, -0.25,
                0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20]
    fov_rad = float(fov) * math.pi / 128.0
    col_from_tan_scale = float(W) / fov_rad

    col_a = piecewise_linear(
        tan_a, _ATAN_BP,
        lambda t: math.atan(t) * col_from_tan_scale + W / 2.0,
        name="col_a",
    )
    col_b = piecewise_linear(
        tan_b, _ATAN_BP,
        lambda t: math.atan(t) * col_from_tan_scale + W / 2.0,
        name="col_b",
    )

    # min/max of col_a, col_b for in_range.
    # Clamp to [-2, W+2] because in_range overflows with distant bounds.
    from torchwright.ops.arithmetic_ops import clamp
    col_a_c = clamp(col_a, -2.0, float(W + 2))
    col_b_c = clamp(col_b, -2.0, float(W + 2))

    a_lt_b = compare(subtract(col_b_c, col_a_c), 0.0)
    vis_lo = select(a_lt_b, col_a_c, col_b_c)
    vis_hi = select(a_lt_b, col_b_c, col_a_c)

    vis_mask = in_range(vis_lo, vis_hi, W)
    gated_vis_mask = select(is_sorted, vis_mask, zeros_W)

    is_sorted_01 = multiply_const(add_const(is_sorted, 1.0), 0.5)

    # =====================================================================
    # RENDER: visibility-masked wall selection + parametric intersection
    # =====================================================================

    # Ray angle from col_idx + player_angle
    col_times_fov = multiply_const(col_idx, float(fov))
    ao_raw = thermometer_floor_div(col_times_fov, W, fov * (W - 1))
    angle_offset = add_const(ao_raw, float(-(fov // 2)))
    ray_angle_raw = add(player_angle, angle_offset)
    ray_angle_shifted = add_const(ray_angle_raw, 256.0)
    ray_angle = mod_const(ray_angle_shifted, 256, 512 + fov)
    ray_cos, ray_sin = trig_lookup(ray_angle)

    perp_shifted = add_const(angle_offset, 256.0)
    perp_angle = mod_const(perp_shifted, 256, 256 + fov)
    perp_cos, _perp_sin = trig_lookup(perp_angle)

    # --- Render attention: visibility-masked wall selection ---
    # The attention score is dominated by the dot product of the render
    # token's column one-hot with the sorted wall's visibility mask.
    # Among visible walls, earlier sort order (lower position) wins.
    #
    # Query (at RENDER): col_onehot (W) + is_render (1)
    # Key (at SORTED_WALL): vis_mask (W) + is_sorted (1)
    # Value (at SORTED_WALL): wall_data (5)

    # Column one-hot: +1 at col_idx, -1 elsewhere → map to 0/1
    col_p1 = add_const(col_idx, 1.0)
    col_onehot = in_range(col_idx, col_p1, W)
    col_onehot_01 = multiply_const(add_const(col_onehot, 1.0), 0.5)
    gated_col_onehot = select(is_render, col_onehot_01, zeros_W)
    is_render_01 = multiply_const(add_const(is_render, 1.0), 0.5)

    # Render attention: visibility-masked nearest-wall selection.
    #
    # Score = VIS_GAIN * vis_mask[col]            — visible (+1) vs hidden (-1)
    #       + VIS_GAIN * SORTED_BIAS * is_sorted  — bonus for sorted tokens
    #       - VIS_GAIN * DIST_SCALE * dist         — prefer closer walls
    #
    # The visibility match (±1) dominates: a 2*VIS_GAIN swing between
    # visible and hidden walls.  Among visible walls, the distance
    # tiebreak selects the nearest.
    VIS_GAIN = 200.0
    SORTED_BIAS = 100.0
    DIST_SCALE = 5.0

    render_attn_in = Concatenate([
        pos_encoding,
        gated_col_onehot, is_render_01,
        gated_vis_mask, is_sorted_01, gated_dist,
        gated_wall_data,
    ])

    d_pe = len(pos_encoding)
    s_col_oh = d_pe
    s_is_render = d_pe + W
    s_vis_mask = d_pe + W + 1
    s_is_sorted = d_pe + 2 * W + 1
    s_dist = d_pe + 2 * W + 2
    s_wall_data = d_pe + 2 * W + 3

    d_head_render = W + 1 + 5

    q_matrix = torch.zeros(len(render_attn_in), d_head_render)
    for c in range(W):
        q_matrix[s_col_oh + c, c] = VIS_GAIN
    q_matrix[s_is_render, W] = VIS_GAIN

    k_matrix = torch.zeros(len(render_attn_in), d_head_render)
    for c in range(W):
        k_matrix[s_vis_mask + c, c] = 1.0
    k_matrix[s_is_sorted, W] = SORTED_BIAS
    k_matrix[s_dist, W] = -DIST_SCALE

    v_matrix = torch.zeros(len(render_attn_in), d_head_render)
    for i in range(5):
        v_matrix[s_wall_data + i, W + 1 + i] = 1.0

    o_matrix = torch.zeros(d_head_render, 5)
    for i in range(5):
        o_matrix[W + 1 + i, i] = 1.0

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

    # START output: type + padding (host ignores START output)
    start_output = Concatenate([
        create_literal_value(E8_START, name="start_type"),
        create_literal_value(torch.zeros(3), name="start_pad"),
    ])

    # EOS output: type + resolved state (collision-resolved position)
    eos_output = Concatenate([
        create_literal_value(E8_EOS, name="eos_type"),
        resolved_x, resolved_y, new_angle,
    ])

    # Pad all to same width and select
    d_sort_out = 8 + 5 + max_walls
    d_render_out = 8 + rp * 3
    d_start_out = 8 + 3
    d_eos_out = 8 + 3
    d_out = max(d_sort_out, d_render_out, d_start_out, d_eos_out)

    def _pad(node, cur_width):
        if cur_width >= d_out:
            return node
        return Concatenate([node, create_literal_value(
            torch.zeros(d_out - cur_width), name="pad")])

    sort_padded = _pad(sort_output, d_sort_out)
    render_padded = _pad(render_output, d_render_out)
    start_padded = _pad(start_output, d_start_out)
    eos_padded = _pad(eos_output, d_eos_out)

    # Three-level select: render vs sort vs eos vs default
    inner1 = select(is_eos, eos_padded, start_padded)
    inner2 = select(is_sorted, sort_padded, inner1)
    output = select(is_render, render_padded, inner2)

    return output, pos_encoding
