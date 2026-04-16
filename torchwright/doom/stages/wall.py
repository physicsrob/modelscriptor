"""WALL stage: per-wall collision + BSP-rank sort score + render precomputation.

At every WALL token the graph computes, for one wall segment:

* **Collision flags** for three rays (full velocity, x-only, y-only)
  against the player's movement ray.  Consumed by EOS for wall-sliding
  resolution.
* **BSP rank** ``rank(W) = dot(coeffs_W, side_P_vec) + const_W`` — a
  front-to-back sort key derived from the BSP tree's spatial structure.
  A clean integer permutation of ``0..N-1``; used as the score by
  SORTED's ``attend_argmin_valid_unmasked``.
* **Renderability flag** ``is_renderable`` — ±1 boolean, true iff this
  is a real wall token whose central ray is not parallel to the wall
  and whose intersection is in front of the player.  SORTED's argmin
  treats non-renderable walls as *invalid keys* rather than folding the
  concern into the sort score.
* **Render precomputations** ``sort_den, C, D, E, H_inv`` — the wall
  geometry rotated into the player's angular frame so RENDER only
  needs per-column angle offsets.
* **Visibility column range** ``(vis_lo, vis_hi)`` — screen-column
  endpoints of the wall's visible arc, computed by rotating the wall
  segment into the player's frame and clipping against the FOV cone.
  Gated to 0 for non-renderable walls to prevent softmax leakage of
  sentinel values in SORTED's attention.
* **Position one-hot** so SORTED can mask out already-picked walls.

These items are packed (via ``wall_payload.pack_wall_payload``)
into a single value node consumed by SORTED's attention.
"""

import math
from dataclasses import dataclass

import torch

from torchwright.graph import Linear, Node, annotate
from torchwright.graph.asserts import assert_integer, assert_onehot
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    clamp,
    compare,
    low_rank_2d,
    max as max_node,
    min as min_node,
    multiply_2d,
    multiply_const,
    negate,
    piecewise_linear_2d,
    reciprocal,
    subtract,
    sum_nodes,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true, cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import DIFF_BP, TRIG_BP, VEL_BP
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.wall_payload import pack_wall_payload


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class CollisionFlags:
    """Three ray-segment hit flags, each 1.0 (hit) or -1.0 (miss).

    Gated to -1.0 at non-WALL positions so EOS's mean-over-WALLs aggregation
    only sees real hit/miss decisions.
    """

    hit_full: Node     # full (vel_dx, vel_dy) ray
    hit_x: Node        # x-only (vel_dx, 0) ray
    hit_y: Node        # y-only (0, vel_dy) ray


@dataclass
class WallInputs:
    # Host-fed wall geometry (meaningful only at WALL positions).
    wall_ax: Node
    wall_ay: Node
    wall_bx: Node
    wall_by: Node
    wall_tex_id: Node
    wall_index: Node

    # Host-fed player state (pre-collision-resolution).
    player_x: Node
    player_y: Node

    # Token-type flag (1.0 at WALL positions).
    is_wall: Node

    # Broadcast values from the INPUT stage.
    vel_dx: Node
    vel_dy: Node
    move_cos: Node
    move_sin: Node

    # BSP rank precomputation: ``rank = dot(coeffs, side_P_vec) + const``.
    # Host precomputes coefficients from the BSP tree structure; side_P_vec
    # comes from the BSP stage's broadcast.
    wall_bsp_coeffs: Node   # max_bsp_nodes-wide (meaningful at WALL positions)
    wall_bsp_const: Node    # 1-wide (meaningful at WALL positions)
    side_P_vec: Node        # max_bsp_nodes-wide (broadcast from BSP stage)


@dataclass
class WallOutputs:
    collision: CollisionFlags
    sort_score: Node        # per-position, fed as ``score`` to SORTED's argmin
    sort_value: Node        # packed payload, fed as ``value`` to SORTED's argmin
    position_onehot: Node   # per-wall one-hot + 0.5 bias, fed to SORTED argmin
    is_renderable: Node     # ±1 validity signal fed to SORTED's argmin


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_wall(
    inputs: WallInputs,
    config: RenderConfig,
    max_walls: int,
    max_coord: float,
    max_bsp_nodes: int,
) -> WallOutputs:
    H = config.screen_height

    with annotate("wall/collision"):
        collision = _compute_collision_flags(inputs)

    with annotate("wall/intersection"):
        sort_den, sort_num_t = _compute_central_ray_intersection(inputs)

    with annotate("wall/precompute"):
        precomp_C, precomp_D, precomp_E, precomp_H_inv = (
            _compute_render_precomputation(inputs, sort_num_t, H, max_coord)
        )

    with annotate("bsp/rank"):
        bsp_rank, is_renderable = _compute_bsp_rank(
            inputs, sort_den, sort_num_t, max_bsp_nodes,
        )

    with annotate("wall/visibility"):
        vis_lo, vis_hi = _compute_visibility_columns(
            inputs.wall_ax, inputs.wall_ay, inputs.wall_bx, inputs.wall_by,
            inputs.player_x, inputs.player_y,
            inputs.move_cos, inputs.move_sin,
            is_renderable,
            config=config, max_coord=max_coord,
        )

    with annotate("wall/onehot"):
        position_onehot = _compute_position_onehot(inputs.wall_index, max_walls)

    sort_value = pack_wall_payload(
        inputs.wall_ax, inputs.wall_ay, inputs.wall_bx, inputs.wall_by,
        inputs.wall_tex_id,
        sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv,
        bsp_rank,
        vis_lo, vis_hi,
        position_onehot,
    )

    return WallOutputs(
        collision=collision,
        sort_score=bsp_rank,
        sort_value=sort_value,
        position_onehot=position_onehot,
        is_renderable=is_renderable,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _collision_validity(den: Node, num_t: Node, num_u: Node) -> Node:
    """Ray-segment intersection validity from (den, num_t, num_u).

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


def _compute_collision_flags(inputs: WallInputs) -> CollisionFlags:
    """Per-wall hit flags for three movement rays.

    Six shared ``piecewise_linear_2d`` products drive three validity checks
    (full, x-only, y-only) via different (den, num_t, num_u) assignments.
    """
    # Wall edge vectors and player-to-wall-start vectors (cheap subtracts).
    ex = subtract(inputs.wall_bx, inputs.wall_ax)
    ey = subtract(inputs.wall_by, inputs.wall_ay)
    dax = subtract(inputs.wall_ax, inputs.player_x)
    day = subtract(inputs.wall_ay, inputs.player_y)

    # Shared products (6 MLP sublayers).
    p_dx_ey = piecewise_linear_2d(inputs.vel_dx, ey, VEL_BP, DIFF_BP,
                                   lambda a, b: a * b, name="c_dx_ey")
    p_dy_ex = piecewise_linear_2d(inputs.vel_dy, ex, VEL_BP, DIFF_BP,
                                   lambda a, b: a * b, name="c_dy_ex")
    p_dax_ey = piecewise_linear_2d(dax, ey, DIFF_BP, DIFF_BP,
                                    lambda a, b: a * b, name="c_dax_ey")
    p_day_ex = piecewise_linear_2d(day, ex, DIFF_BP, DIFF_BP,
                                    lambda a, b: a * b, name="c_day_ex")
    p_dax_dy = piecewise_linear_2d(dax, inputs.vel_dy, DIFF_BP, VEL_BP,
                                    lambda a, b: a * b, name="c_dax_dy")
    p_day_dx = piecewise_linear_2d(day, inputs.vel_dx, DIFF_BP, VEL_BP,
                                    lambda a, b: a * b, name="c_day_dx")

    # Shared num_t (same for all three rays).
    num_t = subtract(p_dax_ey, p_day_ex)

    # Full ray: (vel_dx, vel_dy).
    den_full = subtract(p_dx_ey, p_dy_ex)
    num_u_full = subtract(p_dax_dy, p_day_dx)
    hit_full_raw = _collision_validity(den_full, num_t, num_u_full)

    # X-only ray: (vel_dx, 0).
    den_x = p_dx_ey
    num_u_x = negate(p_day_dx)
    hit_x_raw = _collision_validity(den_x, num_t, num_u_x)

    # Y-only ray: (0, vel_dy).
    den_y = negate(p_dy_ex)
    num_u_y = p_dax_dy
    hit_y_raw = _collision_validity(den_y, num_t, num_u_y)

    # Gate to -1.0 at non-WALL positions.
    no_hit = create_literal_value(torch.tensor([-1.0]), name="no_hit")
    return CollisionFlags(
        hit_full=select(inputs.is_wall, hit_full_raw, no_hit),
        hit_x=select(inputs.is_wall, hit_x_raw, no_hit),
        hit_y=select(inputs.is_wall, hit_y_raw, no_hit),
    )


def _compute_central_ray_intersection(inputs: WallInputs):
    """Intersect the player's central viewing ray with this wall's line.

    Shared with the RENDER phase's parametric-intersection math:
        ex, ey = edge vector (B - A)
        fx = ax - px, gy = py - ay
        den = ey*cos - ex*sin
        num_t = ey*fx + ex*gy
        t = num_t / den  (positive = in front)

    Returns ``(sort_den, sort_num_t)`` — downstream divides them to get the
    front-to-back distance used as the sort score.
    """
    w_ex = subtract(inputs.wall_bx, inputs.wall_ax)
    w_ey = subtract(inputs.wall_by, inputs.wall_ay)
    w_fx = subtract(inputs.wall_ax, inputs.player_x)
    w_gy = subtract(inputs.player_y, inputs.wall_ay)

    sort_ey_cos = piecewise_linear_2d(
        w_ey, inputs.move_cos, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_ey_cos",
    )
    sort_ex_sin = piecewise_linear_2d(
        w_ex, inputs.move_sin, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_ex_sin",
    )
    sort_den = subtract(sort_ey_cos, sort_ex_sin)

    sort_ey_fx = piecewise_linear_2d(
        w_ey, w_fx, DIFF_BP, DIFF_BP,
        lambda a, b: a * b, name="sort_ey_fx",
    )
    sort_ex_gy = piecewise_linear_2d(
        w_ex, w_gy, DIFF_BP, DIFF_BP,
        lambda a, b: a * b, name="sort_ex_gy",
    )
    sort_num_t = add(sort_ey_fx, sort_ex_gy)

    return sort_den, sort_num_t


def _compute_render_precomputation(
    inputs: WallInputs,
    sort_num_t: Node,
    H: int,
    max_coord: float,
):
    """Rotate wall edge + player-to-A into the player's angular frame.

    RENDER only needs per-column angle offsets (perp_cos, perp_sin) rather
    than full ray angles.  The precomputed triple (C, D, E) captures the
    column-independent part; H_inv is the wall-height scale factor.

        C = ey*sin_p + ex*cos_p
        D = fx*sin_p + gy*cos_p
        E = fx*cos_p - gy*sin_p
        H_inv = H / |num_t|
    """
    w_ex = subtract(inputs.wall_bx, inputs.wall_ax)
    w_ey = subtract(inputs.wall_by, inputs.wall_ay)
    w_fx = subtract(inputs.wall_ax, inputs.player_x)
    w_gy = subtract(inputs.player_y, inputs.wall_ay)

    sort_ey_sin = piecewise_linear_2d(
        w_ey, inputs.move_sin, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_ey_sin",
    )
    sort_ex_cos = piecewise_linear_2d(
        w_ex, inputs.move_cos, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_ex_cos",
    )
    precomp_C = add(sort_ey_sin, sort_ex_cos)

    sort_fx_sin = piecewise_linear_2d(
        w_fx, inputs.move_sin, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_fx_sin",
    )
    sort_gy_cos = piecewise_linear_2d(
        w_gy, inputs.move_cos, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_gy_cos",
    )
    precomp_D = add(sort_fx_sin, sort_gy_cos)

    sort_fx_cos = piecewise_linear_2d(
        w_fx, inputs.move_cos, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_fx_cos",
    )
    sort_gy_sin = piecewise_linear_2d(
        w_gy, inputs.move_sin, DIFF_BP, TRIG_BP,
        lambda a, b: a * b, name="sort_gy_sin",
    )
    precomp_E = subtract(sort_fx_cos, sort_gy_sin)

    abs_num_t = abs(sort_num_t)
    inv_abs_num_t = reciprocal(
        abs_num_t, min_value=0.3,
        max_value=2.0 * max_coord * max_coord, step=1.0,
    )
    precomp_H_inv = multiply_const(inv_abs_num_t, float(H))

    return precomp_C, precomp_D, precomp_E, precomp_H_inv


def _compute_bsp_rank(
    inputs: WallInputs,
    sort_den: Node,
    sort_num_t: Node,
    max_bsp_nodes: int,
):
    """BSP-derived front-to-back sort key + per-wall renderability flag.

        rank(W) = dot(coeffs_W, side_P_vec) + const_W

    Since ``side_P_vec[i] ∈ {0, 1}``, the dot product simplifies to
    "keep ``coeffs[i]`` where ``side_P[i]=1``, else 0" — implemented per
    element with ``compare + cond_gate``.

    Renderability: a wall is renderable if it is a real wall token, the
    central ray is not parallel to it (``|sort_den| > ε``), and the wall
    is in front of the player (``num_t`` agrees in sign with ``den``).
    Renderability is returned as a separate ±1 boolean so SORTED's
    argmin can treat non-renderable walls as *invalid keys* via
    ``attend_argmin_valid_unmasked`` rather than having to encode them
    into the sort score with a sentinel. The sort score itself is thus a
    clean integer rank (BSP-permutation in ``0..N-1``); no tiebreak, no
    sentinel, no gating.

    Returns:
        ``(bsp_rank, is_renderable)`` — both per-position ``Node``s.
    """
    # Per-element product: keep coeffs[i] where side_P[i]=1, else 0.
    bsp_products = []
    for i in range(max_bsp_nodes):
        c_i = extract_from(
            inputs.wall_bsp_coeffs, max_bsp_nodes, i, 1, f"bsp_c_{i}",
        )
        s_i = extract_from(
            inputs.side_P_vec, max_bsp_nodes, i, 1, f"bsp_s_{i}",
        )
        # Compare against 0.5 yields a stable ±1 bool even against small
        # interpolation noise in side_P_vec's 0/1 values.
        s_bool = compare(s_i, 0.5)
        bsp_products.append(cond_gate(s_bool, c_i))
    bsp_dot = sum_nodes(bsp_products)
    bsp_rank = assert_integer(add(bsp_dot, inputs.wall_bsp_const))

    # Renderability: is_wall AND |sort_den| > ε AND num_t × sign(den) > 0.
    abs_sort_den = abs(sort_den)
    is_den_ok = compare(abs_sort_den, 0.05)
    den_sign = compare(sort_den, 0.0)
    adj_num_t = select(den_sign, sort_num_t, negate(sort_num_t))
    is_t_pos = compare(adj_num_t, 0.0)
    is_renderable = bool_all_true([inputs.is_wall, is_den_ok, is_t_pos])

    return bsp_rank, is_renderable


def _compute_position_onehot(wall_index: Node, max_walls: int) -> Node:
    """One-hot(wall_index) biased by +0.5 so SORTED attention can pick by dot product."""
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    return assert_onehot(add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh))


# ---------------------------------------------------------------------------
# Visibility column range
# ---------------------------------------------------------------------------

# Breakpoint grid for the front-facing ``(cross, |dot|) → col_front``
# lookup in ``_endpoint_to_column``.  Focused tightly on the *active*
# region — (cross, dot) pairs whose projected column lands inside
# ``[-2, W+2]``.  Outside that region the output saturates against the
# clamp inside the fn, so the piecewise_linear_2d's built-in edge
# extension (hold grid-edge value for out-of-grid inputs) gives the
# right answer with zero additional hyperplanes.  The dot axis is
# positive-only; the behind-flip is a single select on ``sign(dot)``
# after the lookup.
_COL_FOLD_BP_CROSS = [
    -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1,
    0.0,
    0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
]
_COL_FOLD_BP_DOT_ABS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]


# Saturation scale for compares whose input is a t-parameter difference
# naturally clustering near 0 (t ∈ [0, 1]).  The piecewise compare's built-in
# ramp width is ~1/step_sharpness ≈ 0.1 in input units, too wide when real
# t-differences are ~0.01.  Scaling by 100 shrinks the ramp to ~0.001 in
# t-space — tight enough that real differences always saturate cleanly.
_T_COMPARE_SCALE = 100.0


def _compute_visibility_columns(
    wall_ax: Node, wall_ay: Node, wall_bx: Node, wall_by: Node,
    player_x: Node, player_y: Node,
    move_cos: Node, move_sin: Node,
    is_renderable: Node,
    *,
    config: RenderConfig,
    max_coord: float,
):
    """Screen-column range subtended by this wall from the player's pose.

    Implements DOOM-style view-frustum clipping: the wall segment
    ``AB`` is clipped in the player's (dot, cross) frame against the
    two FOV-boundary half-planes

        f_L = sin(½·fov)·dot − cos(½·fov)·cross ≥ 0  (left boundary)
        f_R = sin(½·fov)·dot + cos(½·fov)·cross ≥ 0  (right boundary)

    Their intersection is the FOV cone (and automatically requires
    ``dot > 0``, since ``f_L + f_R = 2·sin(½·fov)·dot``).  The clipped
    segment's endpoints are then projected to screen columns via the
    standard ``atan(cross/dot)·(W/fov) + W/2`` mapping.  If the
    segment misses the cone entirely, ``vis_lo == vis_hi`` marks an
    empty range.

    Results are gated by ``is_renderable`` so non-renderable walls
    contribute 0 (not the W+2 sentinel) to any downstream softmax
    leakage through SORTED's attention.

    Uses pre-collision player state (``player_x/y``, ``move_cos/sin``)
    rather than post-collision resolved pose.  The angle is identical
    (collision resolution does not change facing direction); position
    differs by at most one velocity step on collision frames —
    negligible for column-level clip.
    """
    W = config.screen_width
    fov = config.fov_columns

    dax = subtract(wall_ax, player_x)
    day = subtract(wall_ay, player_y)
    dbx = subtract(wall_bx, player_x)
    dby = subtract(wall_by, player_y)

    # Rotate both endpoints into the player's (cross, dot) frame.
    # These are computed independently of precomp_D/E (which rotate the
    # wall edge + player-to-A for RENDER) — reusing precomp_D/E would
    # entangle residual-stream columns and amplify cross-coupling
    # through the compiled softmax.
    cross_a, dot_a = _rotate_into_player_frame(
        move_cos, move_sin, dax, day, "va",
    )
    cross_b, dot_b = _rotate_into_player_frame(
        move_cos, move_sin, dbx, dby, "vb",
    )

    fov_rad = float(fov) * math.pi / 128.0
    half_fov_rad = fov_rad / 2.0
    sin_hf = math.sin(half_fov_rad)
    cos_hf = math.cos(half_fov_rad)

    # FOV-boundary evaluations at the two endpoints.
    f_L_a = add_scaled_nodes(sin_hf, dot_a, -cos_hf, cross_a)
    f_L_b = add_scaled_nodes(sin_hf, dot_b, -cos_hf, cross_b)
    f_R_a = add_scaled_nodes(sin_hf, dot_a,  cos_hf, cross_a)
    f_R_b = add_scaled_nodes(sin_hf, dot_b,  cos_hf, cross_b)

    max_f_mag = (sin_hf + cos_hf) * max_coord    # bound on |f_*|
    max_denom = 2.0 * max_f_mag                  # bound on |f_A − f_B|

    t_lo_L, t_hi_L = _plane_clip_contribs(
        f_L_a, f_L_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="L",
    )
    t_lo_R, t_hi_R = _plane_clip_contribs(
        f_R_a, f_R_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="R",
    )

    zero_lit = create_literal_value(torch.tensor([0.0]), name="t_zero")
    one_lit  = create_literal_value(torch.tensor([1.0]), name="t_one")

    # t_lo = max(0, t_lo_L, t_lo_R), t_hi = min(1, t_hi_L, t_hi_R).
    # Use the ``(a±b ± |a-b|)/2`` max/min so we don't compound
    # compare+select transition-width errors across nested ops.
    t_lo = max_node(max_node(zero_lit, t_lo_L), t_lo_R)
    t_hi = min_node(min_node(one_lit,  t_hi_L), t_hi_R)

    # Segment fully outside FOV iff t_lo > t_hi.  Scale so the compare
    # saturates even when the range is "just barely" empty/non-empty.
    is_empty = compare(
        multiply_const(subtract(t_lo, t_hi), _T_COMPARE_SCALE), 0.0,
    )

    # ---- Project the clipped endpoints to screen cols ------------------
    # We don't need to interpolate A + t·(B−A) in (dot, cross) and then
    # project.  Observation: whenever an endpoint gets clipped, it ends
    # up on a known FOV boundary, whose screen col is known to be 0 or W
    # by construction (the boundary rays are at ±½·fov from forward,
    # which map to col=0 and col=W exactly).  So we only project the
    # original endpoints (for the case they're already inside the cone)
    # and for the clipped case just pick 0 or W.
    W_lit = create_literal_value(torch.tensor([float(W)]), name="col_W")

    col_A_interior = _endpoint_to_column(
        cross_a, dot_a, W=W, fov=fov, max_coord=max_coord, suffix="a_int",
    )
    col_B_interior = _endpoint_to_column(
        cross_b, dot_b, W=W, fov=fov, max_coord=max_coord, suffix="b_int",
    )

    # Which side of the cone clipped A?  L contribution won → col=W (left
    # screen edge).  R contribution won → col=0 (right edge).
    a_inside_L = compare(f_L_a, 0.0)
    a_inside_R = compare(f_R_a, 0.0)
    a_clipped_on_L = compare(
        multiply_const(subtract(t_lo_L, t_lo_R), _T_COMPARE_SCALE), 0.0,
    )
    col_A_boundary = select(a_clipped_on_L, W_lit, zero_lit)
    col_A = select(
        a_inside_L,
        select(a_inside_R, col_A_interior, col_A_boundary),
        col_A_boundary,
    )

    # B is clipped by whichever plane it *exits* first (smaller t_hi).
    b_inside_L = compare(f_L_b, 0.0)
    b_inside_R = compare(f_R_b, 0.0)
    b_clipped_on_L = compare(
        multiply_const(subtract(t_hi_R, t_hi_L), _T_COMPARE_SCALE), 0.0,
    )
    col_B_boundary = select(b_clipped_on_L, W_lit, zero_lit)
    col_B = select(
        b_inside_L,
        select(b_inside_R, col_B_interior, col_B_boundary),
        col_B_boundary,
    )

    vis_lo_visible = min_node(col_A, col_B)
    vis_hi_visible = max_node(col_A, col_B)

    # On empty segment, collapse to the right-edge sentinel so the
    # render stage reads zero-width cover and skips the wall.
    sentinel = create_literal_value(
        torch.tensor([float(W + 2)]), name="vis_empty_sentinel",
    )
    vis_lo_raw = select(is_empty, sentinel, vis_lo_visible)
    vis_hi_raw = select(is_empty, sentinel, vis_hi_visible)

    # Gate by is_renderable so non-renderable walls contribute 0 (not
    # the W+2 sentinel) to any downstream softmax leakage.
    vis_lo = cond_gate(is_renderable, vis_lo_raw)
    vis_hi = cond_gate(is_renderable, vis_hi_raw)

    return vis_lo, vis_hi


def _plane_clip_contribs(
    f_a: Node, f_b: Node, *, max_denom: float, max_f_mag: float, suffix: str,
) -> tuple[Node, Node]:
    """Per-plane contributions ``(t_lo_contrib, t_hi_contrib)`` for clipping
    a segment against a half-plane ``f(p) ≥ 0``.

    The crossing parameter is ``t* = f_A / (f_A − f_B)``.  The visible
    sub-range along the segment is::

        t_lo_contrib = 0   if A is inside (f_A ≥ 0) else t*
        t_hi_contrib = 1   if B is inside (f_B ≥ 0) else t*

    Combined with the ``max(0, …)`` / ``min(1, …)`` aggregation across
    both FOV planes, this correctly yields:

    * both endpoints inside this plane → ``[0, 1]`` (unconstrained).
    * both outside → ``[t*, t*]``; t* lies outside ``[0, 1]`` because a
      linear ``f`` is monotone along the segment, so aggregation forces
      ``t_lo > t_hi`` and the segment is rejected.
    * A inside, B outside → ``[0, t*]``.
    * A outside, B inside → ``[t*, 1]``.
    """
    denom = subtract(f_a, f_b)
    denom_pos = compare(denom, 0.0)
    denom_abs = clamp(abs(denom), 0.1, max_denom)
    inv_denom_abs = reciprocal(
        denom_abs, min_value=0.1, max_value=max_denom, step=0.1,
    )
    max_inv = 1.0 / 0.1  # upper bound on |inv_denom_abs|

    # t_star (signed) = f_a / denom.
    # Compute |t_star| via multiply_2d and flip sign based on denom.
    t_star_pos = multiply_2d(
        f_a, inv_denom_abs,
        max_abs1=max_f_mag, max_abs2=max_inv,
        step1=0.5, step2=0.5, min2=0.0,
        name=f"t_star_pos_{suffix}",
    )
    t_star_neg = multiply_const(t_star_pos, -1.0)
    t_star = select(denom_pos, t_star_pos, t_star_neg)

    zero_lit = create_literal_value(torch.tensor([0.0]), name=f"t_zero_{suffix}")
    one_lit  = create_literal_value(torch.tensor([1.0]), name=f"t_one_{suffix}")
    a_inside = compare(f_a, 0.0)
    b_inside = compare(f_b, 0.0)

    t_lo_contrib = select(a_inside, zero_lit, t_star)
    t_hi_contrib = select(b_inside, one_lit,  t_star)
    return t_lo_contrib, t_hi_contrib


def _rotate_into_player_frame(
    cos_p: Node, sin_p: Node, dx: Node, dy: Node, suffix: str,
):
    """Return ``(cross, dot)`` of a 2D offset with the player's facing vector.

    ``cross = cos*dy - sin*dx``  (signed perpendicular distance)
    ``dot   = cos*dx + sin*dy``  (forward projection)
    """
    cross = subtract(
        piecewise_linear_2d(cos_p, dy, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"cos_dy_{suffix}"),
        piecewise_linear_2d(sin_p, dx, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"sin_dx_{suffix}"),
    )
    dot = add(
        piecewise_linear_2d(cos_p, dx, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"cos_dx_{suffix}"),
        piecewise_linear_2d(sin_p, dy, TRIG_BP, DIFF_BP,
                            lambda a, b: a * b, name=f"sin_dy_{suffix}"),
    )
    return cross, dot


def _endpoint_to_column(
    cross: Node, dot: Node, *, W: int, fov: int, max_coord: float, suffix: str,
) -> Node:
    """Convert ``(cross, dot)`` to a signed screen column.

    Front case uses ``low_rank_2d`` to approximate ``atan(cross/|dot|)``
    as a rank-3 SVD-truncated separable sum, then applies the
    ``·col_scale + W/2`` affine as a free Linear.  The behind case
    (``dot < 0``) is a single select on ``sign(dot)`` afterwards,
    reflecting the column across screen centre.

    **Why low_rank_2d, not piecewise_linear_2d:** the latter's pinv
    min-norm solution oscillates in cell interiors on the non-uniform
    ``_COL_FOLD_BP_DOT_ABS`` grid — a single wall endpoint landing
    mid-cell produced ~4-col drift (see TODO.md).  ``low_rank_2d`` is
    the SVD-optimal rank-K fit with a deterministic worst-cell error
    bound of ``σ_{K+1}``.

    The fn is the raw ``atan`` (no scale, no clamp) because absorbing
    ``col_scale`` inflates the SVD spectrum and clamping at the grid
    corners inflates the effective rank.  Scale + bias are applied
    outside by a free Linear; saturation is handled by the outer
    ``clamp(col_final, col_lo, col_hi)`` below.

    **Precision:** rank-3 on this grid gives max error ≤ 0.0018 rad
    ≈ 0.28 col at ``col_scale ≈ 150``, below the ~0.5-col render
    tolerance.
    """
    col_lo, col_hi = -2.0, float(W + 2)
    fov_rad = float(fov) * math.pi / 128.0
    col_scale = float(W) / fov_rad
    half_W = float(W) / 2.0

    def _atan_of(cr: float, dt_abs: float) -> float:
        return math.atan(cr / dt_abs)

    bp_cross_lo = _COL_FOLD_BP_CROSS[0]
    bp_cross_hi = _COL_FOLD_BP_CROSS[-1]
    bp_dot_lo = _COL_FOLD_BP_DOT_ABS[0]
    bp_dot_hi = _COL_FOLD_BP_DOT_ABS[-1]

    dot_sign = compare(dot, 0.0)
    abs_dot = abs(dot)
    cross_clamped = clamp(cross, bp_cross_lo, bp_cross_hi)
    dot_pos = clamp(abs_dot, bp_dot_lo, bp_dot_hi)

    atan_val = low_rank_2d(
        cross_clamped, dot_pos,
        _COL_FOLD_BP_CROSS, _COL_FOLD_BP_DOT_ABS,
        _atan_of,
        rank=3,
        name=f"atan_front_{suffix}",
    )
    col_front = Linear(
        atan_val, torch.tensor([[col_scale]]),
        torch.tensor([half_W]), name=f"col_front_{suffix}",
    )

    # Behind-flip: ``col_back = W − col_front`` (reflection across
    # centre), matching the prior implementation.
    col_back = Linear(
        col_front, torch.tensor([[-1.0]]),
        torch.tensor([float(W)]), name=f"col_{suffix}_back",
    )
    col_final = select(dot_sign, col_front, col_back)

    return clamp(col_final, col_lo, col_hi)
