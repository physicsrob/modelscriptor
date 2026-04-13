"""Walls-as-tokens game graph: multi-phase renderer with runtime wall geometry.

Token sequence per frame:

    TEX_COL×(num_tex × tex_w) → INPUT → WALL×N → EOS → SORTED_WALL×N → RENDER×(dynamic)

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

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    bool_to_01,
    clamp,
    compare,
    floor_int,
    mod_const,
    multiply_2d,
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
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_argmin_unmasked,
    attend_mean_where,
)
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true, bool_any_true, bool_not, cond_gate, equals_vector
from torchwright.ops.map_select import in_range, select

from torchwright.doom.renderer import (
    _textured_column_fill,
    trig_lookup,
)
from torchwright.reference_renderer.types import RenderConfig


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

TOKEN_INPUT = 0
TOKEN_WALL = 1
TOKEN_EOS = 2
TOKEN_SORTED_WALL = 3
TOKEN_RENDER = 4

E8_INPUT = index_to_vector(TOKEN_INPUT)
E8_WALL = index_to_vector(TOKEN_WALL)
E8_EOS = index_to_vector(TOKEN_EOS)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_RENDER = index_to_vector(TOKEN_RENDER)

TOKEN_TEX_COL = 5
E8_TEX_COL = index_to_vector(TOKEN_TEX_COL)
TEX_E8_OFFSET = 6  # index_to_vector(6+i) = E8 code for texture i


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
    chunk_size: int = 20,
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
    cs = chunk_size

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
    d_sort_out = 8 + 5 + 3 + 2 * max_walls
    sort_feedback = create_input("sort_feedback", d_sort_out)
    prev_mask = _extract_from(
        sort_feedback, d_sort_out, 8 + 5 + 3 + max_walls, max_walls, "prev_mask",
    )
    d_render_fb = max_walls + 3
    render_feedback = create_input("render_feedback", d_render_fb)
    render_mask = _extract_from(render_feedback, d_render_fb, 0, max_walls, "render_mask")
    render_col = _extract_from(render_feedback, d_render_fb, max_walls, 1, "render_col")
    render_is_new_wall = _extract_from(
        render_feedback, d_render_fb, max_walls + 1, 1, "render_is_new_wall",
    )
    render_chunk_start = _extract_from(
        render_feedback, d_render_fb, max_walls + 2, 1, "render_chunk_start",
    )
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
    with annotate("tex_col"):
        tc_p1 = add_const(tex_col_input, 1.0)
        tc_onehot_01 = bool_to_01(in_range(tex_col_input, tc_p1, tex_w))

    # =====================================================================
    # INPUT: game logic (angle update + velocity, no collision)
    # =====================================================================
    with annotate("input"):

        with annotate("game_logic"):
            new_angle = _compute_new_angle(
                player_angle, input_turn_left, input_turn_right, turn_speed,
            )
            vel_dx, vel_dy = _compute_velocity(
                new_angle, input_forward, input_backward,
                input_strafe_left, input_strafe_right, move_speed,
            )
            move_cos, move_sin = trig_lookup(new_angle)

        # --- INPUT attention: distribute controls from INPUT to all positions ---
        # At INPUT, the graph computes correct new_angle, vel_dx/dy, move_cos/sin
        # from control inputs.  attend_mean_where averages over validity=is_input
        # positions — since there's exactly one, it passes through the values.
        with annotate("attention"):
            ctrl_attn = attend_mean_where(
                pos_encoding,
                validity=is_input,
                value=Concatenate([vel_dx, vel_dy, move_cos, move_sin, new_angle]),
            )

            attn_vel_dx   = _extract_from(ctrl_attn, 5, 0, 1, "a_vdx")
            attn_vel_dy   = _extract_from(ctrl_attn, 5, 1, 1, "a_vdy")
            attn_move_cos = _extract_from(ctrl_attn, 5, 2, 1, "a_mcos")
            attn_move_sin = _extract_from(ctrl_attn, 5, 3, 1, "a_msin")
            attn_new_angle = _extract_from(ctrl_attn, 5, 4, 1, "a_angle")

    # =====================================================================
    # WALL: distance score + sort value + collision hit flags
    # =====================================================================

    # Runtime collision: velocity comes from INPUT attention.
    with annotate("wall/collision"):
        hit_full, hit_x, hit_y = _build_wall_collision(
            attn_vel_dx, attn_vel_dy, player_x, player_y,
            wall_ax, wall_ay, wall_bx, wall_by, is_wall,
        )

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
    with annotate("wall/intersection"):
        w_ex = subtract(wall_bx, wall_ax)
        w_ey = subtract(wall_by, wall_ay)
        w_fx = subtract(wall_ax, player_x)
        w_gy = subtract(player_y, wall_ay)

        sort_ey_cos = piecewise_linear_2d(
            w_ey, attn_move_cos, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_ey_cos",
        )
        sort_ex_sin = piecewise_linear_2d(
            w_ex, attn_move_sin, _DIFF_BP, _TRIG_BP,
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

    # --- Precomputed render values (column-independent) ---
    # Rotate wall edge and player-to-A offset into the player's angular
    # frame so the render phase only needs per-column angle offsets
    # (perp_cos, perp_sin) instead of full ray angles (ray_cos, ray_sin).
    #
    #   sort_den = ey*cos_p - ex*sin_p   (already computed above)
    #   C = ey*sin_p + ex*cos_p
    #   D = fx*sin_p + gy*cos_p
    #   E = fx*cos_p - gy*sin_p
    #   H_inv_num_t = H / |num_t|        (wall-height scale factor)
    with annotate("wall/precompute"):
        sort_ey_sin = piecewise_linear_2d(
            w_ey, attn_move_sin, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_ey_sin",
        )
        sort_ex_cos = piecewise_linear_2d(
            w_ex, attn_move_cos, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_ex_cos",
        )
        precomp_C = add(sort_ey_sin, sort_ex_cos)

        sort_fx_sin = piecewise_linear_2d(
            w_fx, attn_move_sin, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_fx_sin",
        )
        sort_gy_cos = piecewise_linear_2d(
            w_gy, attn_move_cos, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_gy_cos",
        )
        precomp_D = add(sort_fx_sin, sort_gy_cos)

        sort_fx_cos = piecewise_linear_2d(
            w_fx, attn_move_cos, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_fx_cos",
        )
        sort_gy_sin = piecewise_linear_2d(
            w_gy, attn_move_sin, _DIFF_BP, _TRIG_BP,
            lambda a, b: a * b, name="sort_gy_sin",
        )
        precomp_E = subtract(sort_fx_cos, sort_gy_sin)

        abs_num_t = abs(sort_num_t)
        inv_abs_num_t = reciprocal(
            abs_num_t, min_value=0.3,
            max_value=2.0 * max_coord * max_coord, step=1.0,
        )
        precomp_H_inv = multiply_const(inv_abs_num_t, float(H))

    # --- Central ray distance for sort score ---
    with annotate("wall/sort_score"):
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
    with annotate("wall/onehot"):
        wall_index_p1 = add_const(wall_index, 1.0)
        onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
        ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
        position_onehot = add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)

    # Pack wall value: geometry + precomputed render data + onehot
    # Indices 0-4: ax,ay,bx,by,tex_id (vis mask needs 0-3)
    # Indices 5-9: sort_den, C, D, E, H_inv_num_t (render pipeline)
    # Index 10: center_ray_dist (render attention distance tiebreak)
    # Indices 11+: position_onehot (sort mask)
    wall_value_for_sort = Concatenate([
        wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id,
        sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv,
        center_ray_dist,
        position_onehot,
    ])
    d_sort_val = 11 + max_walls

    # =====================================================================
    # EOS: attend to WALL positions, aggregate collision, resolve position
    # =====================================================================

    with annotate("eos/collision_resolve"):
        # Remap hit flags from {-1, +1} to {0, 1} for clean averaging
        hit_full_01 = bool_to_01(hit_full)
        hit_x_01 = bool_to_01(hit_x)
        hit_y_01 = bool_to_01(hit_y)

        # Average hit flags across WALL positions.  If any wall was hit,
        # the mean is >= 1/max_walls = 0.125 >> threshold of 0.05.
        resolve_attn = attend_mean_where(
            pos_encoding,
            validity=is_wall,
            value=Concatenate([hit_full_01, hit_x_01, hit_y_01]),
        )

        avg_hf = _extract_from(resolve_attn, 3, 0, 1, "avg_hf")
        avg_hx = _extract_from(resolve_attn, 3, 1, 1, "avg_hx")
        avg_hy = _extract_from(resolve_attn, 3, 2, 1, "avg_hy")

        any_hit_full = compare(avg_hf, 0.05)
        any_hit_x = compare(avg_hx, 0.05)
        any_hit_y = compare(avg_hy, 0.05)

        use_new_x = bool_any_true([negate(any_hit_full), negate(any_hit_x)])
        use_new_y = bool_any_true([negate(any_hit_full), negate(any_hit_y)])

        new_x = add(player_x, attn_vel_dx)
        new_y = add(player_y, attn_vel_dy)
        resolved_x = select(use_new_x, new_x, player_x)
        resolved_y = select(use_new_y, new_y, player_y)

    # --- EOS state attention: distribute resolved state to SORTED ---
    # SORTED positions read resolved player state from EOS via attention
    # instead of host-fed inputs.  This lets the sort loop be purely
    # autoregressive (host just feeds output back as input).
    with annotate("eos/attention"):
        is_sorted_01 = bool_to_01(is_sorted)
        eos_state_attn = attend_mean_where(
            pos_encoding,
            validity=is_eos,
            value=Concatenate([resolved_x, resolved_y, attn_new_angle]),
        )

        eos_px    = _extract_from(eos_state_attn, 3, 0, 1, "eos_px")
        eos_py    = _extract_from(eos_state_attn, 3, 1, 1, "eos_py")
        eos_angle = _extract_from(eos_state_attn, 3, 2, 1, "eos_angle")

    # =====================================================================
    # SORTED_WALL: attend_argmin_unmasked
    # =====================================================================

    with annotate("sort/attention"):
        selected_sort = attend_argmin_unmasked(
            pos_encoding=pos_encoding,
            score=sort_score,
            mask_vector=prev_mask,
            position_onehot=position_onehot,
            value=wall_value_for_sort,
        )

        # Sort rank: number of walls already selected (= sum of mask entries).
        # At step k the mask has k bits set, so sort_rank = k.
        sort_rank = Linear(
            prev_mask, torch.ones(max_walls, 1), name="sort_rank",
        )

        # One-hot encoding of sort_rank for render-phase wall selection
        sort_rank_p1 = add_const(sort_rank, 1.0)
        sr_onehot_bool = in_range(sort_rank, sort_rank_p1, max_walls)
        sort_rank_onehot = add_scaled_nodes(0.5, sr_onehot_bool, 0.5, ones_oh)

        sel_wall_data = _extract_from(selected_sort, d_sort_val, 0, 5, "sel_wall_data")
        sel_render = _extract_from(selected_sort, d_sort_val, 5, 5, "sel_render")
        sel_tex_id = _extract_from(sel_wall_data, 5, 4, 1, "sel_tex_id")
        sel_onehot = _extract_from(selected_sort, d_sort_val, 11, max_walls, "sel_onehot")
        updated_mask = add(prev_mask, sel_onehot)

        # Gate sorted values: zero at non-sorted positions
        # Render data: [sort_den, C, D, E, H_inv_num_t, tex_id] = 6 values
        gated_render_data = cond_gate(is_sorted, Concatenate([sel_render, sel_tex_id]))

    # --- Visibility mask: column range where this wall is visible ---
    with annotate("sort/visibility"):
        sel_ax = _extract_from(sel_wall_data, 5, 0, 1, "sel_ax")
        sel_ay = _extract_from(sel_wall_data, 5, 1, 1, "sel_ay")
        sel_bx = _extract_from(sel_wall_data, 5, 2, 1, "sel_bx")
        sel_by = _extract_from(sel_wall_data, 5, 3, 1, "sel_by")

        dax = subtract(sel_ax, eos_px)
        day = subtract(sel_ay, eos_py)
        dbx = subtract(sel_bx, eos_px)
        dby = subtract(sel_by, eos_py)

        sort_cos, sort_sin = trig_lookup(eos_angle)

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

        # Tangent computation: cross/dot → column index.
        #
        # Optimized pipeline (6 sequential MLP sublayers per endpoint,
        # down from 10):
        #   abs(dot) → clamp(0.1,20) → reciprocal → multiply_2d(cross,inv)
        #   → atan_clamped → select(sign, col, W-col)
        #
        # Key changes vs the original:
        # - clamp(abs) replaces compare+select (1 sublayer vs 2)
        # - multiply_2d replaces signed_multiply (1 sublayer vs 3)
        # - atan bakes in column clamping (1 sublayer vs 2)
        # - sign handling moved after atan so compare(dot,0) runs
        #   in parallel with the main chain

        _ATAN_BP = [-100, -20, -10, -5, -3, -2, -1.5, -1, -0.75, -0.5,
                    -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10,
                    20, 100]
        fov_rad = float(fov) * math.pi / 128.0
        col_from_tan_scale = float(W) / fov_rad
        half_W = float(W) / 2.0
        col_lo, col_hi = -2.0, float(W + 2)

        def _atan_clamped(t):
            return max(col_lo, min(col_hi, math.atan(t) * col_from_tan_scale + half_W))

        max_inv = 1.0 / 0.1  # = 10

        # --- Endpoint A ---
        dot_a_sign = compare(dot_a, 0.0)           # parallel with chain below
        dot_a_clamped = clamp(abs(dot_a), 0.1, 2.0 * max_coord)
        inv_dot_a = reciprocal(dot_a_clamped, min_value=0.1, max_value=2.0 * max_coord)
        tan_a = multiply_2d(cross_a, inv_dot_a,
                            max_abs1=max_coord, max_abs2=max_inv,
                            step1=0.5, step2=0.5, min2=0.0)
        col_a_pos = piecewise_linear(tan_a, _ATAN_BP, _atan_clamped, name="col_a")
        col_a_neg = Linear(col_a_pos, torch.tensor([[-1.0]]),
                           torch.tensor([float(W)]), name="col_a_neg")
        col_a_c = select(dot_a_sign, col_a_pos, col_a_neg)

        # --- Endpoint B ---
        dot_b_sign = compare(dot_b, 0.0)
        dot_b_clamped = clamp(abs(dot_b), 0.1, 2.0 * max_coord)
        inv_dot_b = reciprocal(dot_b_clamped, min_value=0.1, max_value=2.0 * max_coord)
        tan_b = multiply_2d(cross_b, inv_dot_b,
                            max_abs1=max_coord, max_abs2=max_inv,
                            step1=0.5, step2=0.5, min2=0.0)
        col_b_pos = piecewise_linear(tan_b, _ATAN_BP, _atan_clamped, name="col_b")
        col_b_neg = Linear(col_b_pos, torch.tensor([[-1.0]]),
                           torch.tensor([float(W)]), name="col_b_neg")
        col_b_c = select(dot_b_sign, col_b_pos, col_b_neg)

        a_lt_b = compare(subtract(col_b_c, col_a_c), 0.0)
        vis_lo = select(a_lt_b, col_a_c, col_b_c)
        vis_hi = select(a_lt_b, col_b_c, col_a_c)

    # =====================================================================
    # RENDER: autoregressive wall-column render with chunked pixel output
    # =====================================================================

    with annotate("render/wall_attention"):
        # Build score + position_onehot for attend_argmin_unmasked
        render_sentinel = create_literal_value(
            torch.tensor([99.0]), name="render_sentinel",
        )
        render_score = select(is_sorted, sort_rank, render_sentinel)
        z_mw = create_literal_value(
            torch.zeros(max_walls), name="z_mw",
        )
        render_position_onehot = select(is_sorted, sort_rank_onehot, z_mw)

        # Value: render data + col bounds + onehot for mask update
        render_value = Concatenate([
            gated_render_data,   # 6: [sort_den, C, D, E, H_inv, tex_id]
            vis_lo,              # 1: col_lo
            vis_hi,              # 1: col_hi
            sort_rank_onehot,    # max_walls: for mask update
        ])
        render_value_gated = cond_gate(is_sorted, render_value)

        selected_render = attend_argmin_unmasked(
            pos_encoding=pos_encoding,
            score=render_score,
            mask_vector=render_mask,
            position_onehot=render_position_onehot,
            value=render_value_gated,
        )

        d_rv = 8 + max_walls
        r_sort_den    = _extract_from(selected_render, d_rv, 0, 1, "r_sort_den")
        r_C           = _extract_from(selected_render, d_rv, 1, 1, "r_C")
        r_D           = _extract_from(selected_render, d_rv, 2, 1, "r_D")
        r_E           = _extract_from(selected_render, d_rv, 3, 1, "r_E")
        r_H_inv_num_t = _extract_from(selected_render, d_rv, 4, 1, "r_H_inv")
        r_wall_tex    = _extract_from(selected_render, d_rv, 5, 1, "r_tex")
        r_col_lo      = _extract_from(selected_render, d_rv, 6, 1, "r_col_lo")
        r_col_hi      = _extract_from(selected_render, d_rv, 7, 1, "r_col_hi")
        r_onehot      = _extract_from(selected_render, d_rv, 8, max_walls, "r_onehot")

    with annotate("render/state_machine"):
        # Active column: new wall → use col_lo from attention; else feedback col
        active_col = select(render_is_new_wall, r_col_lo, render_col)

        # Per-column angle offset from player's forward direction
        col_times_fov = multiply_const(active_col, float(fov))
        ao_raw = thermometer_floor_div(col_times_fov, W, fov * (W - 1))
        angle_offset = add_const(ao_raw, float(-(fov // 2)))

    # --- Wall height from precomputed values ---
    with annotate("render/wall_height"):
        half_fov = fov // 2
        tan_bp = [float(i) for i in range(-half_fov, half_fov + 1)]
        tan_o = piecewise_linear(
            angle_offset, tan_bp,
            lambda x: math.tan(x * 2.0 * math.pi / 256.0),
            name="tan_offset",
        )

        max_tan = math.tan(half_fov * 2.0 * math.pi / 256.0) * 1.1
        tan_val_bp = [-max_tan + i * (2 * max_tan / 10) for i in range(11)]
        C_tan = piecewise_linear_2d(
            r_C, tan_o, _DIFF_BP, tan_val_bp,
            lambda a, b: a * b, name="C_tan_o",
        )
        den_over_cos = subtract(r_sort_den, C_tan)
        abs_den_over_cos = abs(den_over_cos)

        max_h_inv = float(H) / 0.3
        h_inv_n = 16
        h_inv_ratio = (max_h_inv / 0.01) ** (1.0 / (h_inv_n - 1))
        height_inv_bp = [0.01 * (h_inv_ratio ** k) for k in range(h_inv_n)]
        height_inv_bp[0] = 0.0
        height_inv_bp[-1] = max_h_inv

        doc_max = 2.5 * max_coord
        doc_bp = [doc_max * i / 15 for i in range(16)]

        wall_height_raw = piecewise_linear_2d(
            r_H_inv_num_t, abs_den_over_cos,
            height_inv_bp, doc_bp,
            lambda a, b: a * b, name="wall_height_raw",
        )
        wall_height = clamp(wall_height_raw, 0.0, float(H))

        center = float(H) / 2.0
        half_height = multiply_const(wall_height, 0.5)
        wall_top = Linear(
            half_height, torch.tensor([[-1.0]]),
            torch.tensor([center]), name="wall_top",
        )
        wall_bottom = Linear(
            half_height, torch.tensor([[1.0]]),
            torch.tensor([center]), name="wall_bottom",
        )

    # --- Texture u-coordinate via tan(offset) ---
    with annotate("render/tex_coord"):
        E_tan = piecewise_linear_2d(
            r_E, tan_o, _DIFF_BP, tan_val_bp,
            lambda a, b: a * b, name="E_tan_o",
        )
        num_u_over_cos = add(r_D, E_tan)
        abs_nuc = abs(num_u_over_cos)

        u_raw = piecewise_linear_2d(
            abs_nuc, abs_den_over_cos,
            doc_bp, doc_bp,
            lambda n, d: n / d if d > 0.01 else 0.0,
            name="u_ratio",
        )
        tex_col_float = multiply_const(u_raw, float(tex_w))
        tex_col_clamped = clamp(tex_col_float, 0.0, float(tex_w) - 0.5)
        tex_col_idx = floor_int(tex_col_clamped, 0, tex_w - 1)

    # --- TEX_COL attention: match texture ID + column via dot product ---
    with annotate("render/tex_attention"):
        num_tex = len(textures)
        tex_e8_query = piecewise_linear(
            r_wall_tex,
            [float(i) for i in range(num_tex)],
            lambda tid: [float(v) for v in
                         index_to_vector(int(round(tid)) + TEX_E8_OFFSET)],
            name="tex_id_to_e8",
        )

        tex_col_p1 = add_const(tex_col_idx, 1.0)
        rc_onehot_01 = bool_to_01(in_range(tex_col_idx, tex_col_p1, tex_w))

        COL_SCALE = 10.0
        TEX_MATCH_GAIN = 1000.0
        scaled_rc = multiply_const(rc_onehot_01, COL_SCALE)
        scaled_tc = multiply_const(tc_onehot_01, COL_SCALE)
        tex_col_attn = attend_argmax_dot(
            pos_encoding,
            query_vector=cond_gate(
                is_render, Concatenate([tex_e8_query, scaled_rc])),
            key_vector=cond_gate(
                is_tex_col, Concatenate([texture_id_e8, scaled_tc])),
            value=cond_gate(is_tex_col, tex_pixels),
            match_gain=TEX_MATCH_GAIN,
        )

        tex_column_colors = tex_col_attn

    # Column fill — chunk-based
    with annotate("render/column_fill"):
        vis_top_render = clamp(wall_top, 0.0, float(H))
        vis_bottom_render = clamp(wall_bottom, 0.0, float(H))

        # New column detection (chunk_start < -0.5 = sentinel)
        neg_half = create_literal_value(torch.tensor([-0.5]), name="neg_half")
        is_new_col = compare(subtract(neg_half, render_chunk_start), 0.0)
        active_start = select(is_new_col, vis_top_render, render_chunk_start)

        chunk_length = clamp(
            subtract(vis_bottom_render, active_start), 0.0, float(cs),
        )

        pixels = _textured_column_fill(
            wall_top, wall_bottom, wall_height,
            tex_column_colors, tex_h, config, max_coord=max_coord,
            patch_row_start=active_start, rows_per_patch=cs,
        )

    # State transitions (3-way: more chunks / advance col / advance wall)
    with annotate("render/state_transitions"):
        next_chunk_start_val = add_const(active_start, float(cs))
        has_more_chunks = compare(
            subtract(vis_bottom_render, next_chunk_start_val), 0.5,
        )

        col_p1 = add_const(active_col, 1.0)
        not_more_chunks = bool_not(has_more_chunks)
        has_more_cols = compare(subtract(r_col_hi, col_p1), 0.5)
        advance_col = bool_all_true([not_more_chunks, has_more_cols])
        advance_wall = bool_all_true([not_more_chunks, bool_not(has_more_cols)])

        # Mask update
        mask_with_new = add(render_mask, r_onehot)
        next_render_mask = select(advance_wall, mask_with_new, render_mask)

        # Done detection
        mask_sum = Linear(
            mask_with_new, torch.ones(max_walls, 1), name="render_mask_sum",
        )
        all_walls_done = compare(mask_sum, max_walls - 0.5)
        done_flag = bool_all_true([advance_wall, all_walls_done])

        # Next feedback fields
        zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
        next_col_output = select(
            has_more_chunks, active_col,
            select(advance_col, col_p1, zero_col),
        )
        chunk_sentinel = create_literal_value(
            torch.tensor([-1.0]), name="chunk_sentinel",
        )
        next_chunk = select(has_more_chunks, next_chunk_start_val, chunk_sentinel)
        pos_one = create_literal_value(torch.tensor([1.0]), name="pos_one")
        neg_one = create_literal_value(torch.tensor([-1.0]), name="neg_one")
        next_is_new_wall = select(advance_wall, pos_one, neg_one)

        next_feedback = Concatenate([
            next_render_mask, next_col_output, next_is_new_wall, next_chunk,
        ])

    # =====================================================================
    # Output: gated by token type
    # =====================================================================
    with annotate("output"):

        # SORTED_WALL output: type + wall data + sort_rank + col_lo + col_hi
        #                     + onehot + updated mask
        sort_output = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="sort_type"),
            sel_wall_data,
            sort_rank,
            vis_lo,
            vis_hi,
            sel_onehot,
            updated_mask,
        ])

        # RENDER output: type + col + start_y + length + done + pixels + feedback
        render_output = Concatenate([
            create_literal_value(E8_RENDER, name="render_type"),
            active_col,
            active_start,
            chunk_length,
            done_flag,
            pixels,
            next_feedback,
        ])

        # INPUT output: type + padding (host ignores INPUT output)
        input_output = Concatenate([
            create_literal_value(E8_INPUT, name="input_type"),
            create_literal_value(torch.zeros(3), name="input_pad"),
        ])

        # EOS output: seeds the sort loop with E8_SORTED_WALL type + resolved
        # state at offsets 8-10 + zeros for sort mask.
        eos_output = Concatenate([
            create_literal_value(E8_SORTED_WALL, name="eos_sort_seed"),
            resolved_x, resolved_y, attn_new_angle,
            create_literal_value(
                torch.zeros(2 + 3 + 2 * max_walls), name="eos_sort_pad"),
        ])

        # TEX_COL output: type + padding (host ignores TEX_COL output)
        tex_col_output = Concatenate([
            create_literal_value(E8_TEX_COL, name="tex_col_type"),
            create_literal_value(torch.zeros(3), name="tc_pad"),
        ])

        # Pad all to same width and select
        d_sort_out = 8 + 5 + 3 + 2 * max_walls
        d_render_out = 12 + cs * 3 + max_walls + 3
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
        output = select(is_render, render_padded, inner3)

    return output, pos_encoding
