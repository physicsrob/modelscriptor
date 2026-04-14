"""WALL stage: per-wall collision + BSP-rank sort score + render precomputation.

At every WALL token the graph computes, for one wall segment:

* **Collision flags** for three rays (full velocity, x-only, y-only)
  against the player's movement ray.  Consumed by EOS for wall-sliding
  resolution.
* **BSP rank** ``rank(W) = dot(coeffs_W, side_P_vec) + const_W`` — a
  front-to-back sort key derived from the BSP tree's spatial structure.
  Walls parallel to the viewing ray (``|sort_den|`` ≈ 0) or behind the
  player (``num_t`` disagrees in sign with ``den``) get a sentinel so
  they sort last.  Used as the score by SORTED's
  ``attend_argmin_unmasked``.
* **Render precomputations** ``sort_den, C, D, E, H_inv`` — the wall
  geometry rotated into the player's angular frame so RENDER only
  needs per-column angle offsets.
* **Position one-hot** so SORTED can mask out already-picked walls.

The four last items are packed (via ``wall_payload.pack_wall_payload``)
into a single value node consumed by SORTED's attention.
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Node, annotate
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    compare,
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
        bsp_rank = _compute_bsp_rank(
            inputs, sort_den, sort_num_t, max_bsp_nodes,
        )

    with annotate("wall/onehot"):
        position_onehot = _compute_position_onehot(inputs.wall_index, max_walls)

    sort_value = pack_wall_payload(
        inputs.wall_ax, inputs.wall_ay, inputs.wall_bx, inputs.wall_by,
        inputs.wall_tex_id,
        sort_den, precomp_C, precomp_D, precomp_E, precomp_H_inv,
        bsp_rank,
        position_onehot,
    )

    return WallOutputs(
        collision=collision,
        sort_score=bsp_rank,
        sort_value=sort_value,
        position_onehot=position_onehot,
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
) -> Node:
    """BSP-derived front-to-back sort key.

        rank(W) = dot(coeffs_W, side_P_vec) + const_W

    Since ``side_P_vec[i] ∈ {0, 1}``, the dot product simplifies to
    "keep ``coeffs[i]`` where ``side_P[i]=1``, else 0" — implemented per
    element with ``compare + cond_gate``.

    Renderability gate: walls that are parallel to the viewing ray
    (``|sort_den|`` near zero) or behind the player (``num_t`` disagrees
    in sign with ``den``) would produce degenerate precomputed values.
    They get ``bsp_sentinel = 99.0`` so they sort after all renderable
    walls.  Tie-break among tied sentinels with ``wall_index * 0.1`` so
    the argmin softmax can concentrate on a single winner instead of
    averaging across ties.  Non-WALL positions get a slightly higher
    sentinel (``99.9``) so they always lose to any wall, even an
    unrenderable one.

    Constraints:
    * Real BSP ranks are a permutation of ``0..N-1`` with ``N ≤
      max_walls``; sentinels must stay above ``max_walls - 1``.
    * The tie-break offset ``0.1 * (max_walls-1)`` must stay below
      ``1.0`` so real-rank spacing (1.0) is preserved — limiting this
      scheme to ``max_walls ≤ 10``.
    * Sentinels must stay within ``|score| ≤ 100`` (the
      ``attend_argmin_unmasked`` bound) so the mask penalty can still
      override them.
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
    bsp_rank_raw = add(bsp_dot, inputs.wall_bsp_const)

    # Renderability gate: |sort_den| > ε AND num_t × sign(den) > 0.
    abs_sort_den = abs(sort_den)
    is_den_ok = compare(abs_sort_den, 0.05)
    den_sign = compare(sort_den, 0.0)
    adj_num_t = select(den_sign, sort_num_t, negate(sort_num_t))
    is_t_pos = compare(adj_num_t, 0.0)
    is_wall_renderable = bool_all_true([is_den_ok, is_t_pos])

    bsp_sentinel = create_literal_value(
        torch.tensor([99.0]), name="bsp_sentinel",
    )
    bsp_rank_filtered = select(
        is_wall_renderable, bsp_rank_raw, bsp_sentinel,
    )
    bsp_rank_tiebroken = add(
        bsp_rank_filtered, multiply_const(inputs.wall_index, 0.1),
    )

    # Non-wall positions get a strictly higher sentinel so they never
    # tie with wall_index=0's tiebroken sentinel.
    nonwall_sentinel = create_literal_value(
        torch.tensor([99.9]), name="nonwall_sentinel",
    )
    return select(inputs.is_wall, bsp_rank_tiebroken, nonwall_sentinel)


def _compute_position_onehot(wall_index: Node, max_walls: int) -> Node:
    """One-hot(wall_index) biased by +0.5 so SORTED attention can pick by dot product."""
    wall_index_p1 = add_const(wall_index, 1.0)
    onehot_bool = in_range(wall_index, wall_index_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="ones_oh")
    return add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh)
