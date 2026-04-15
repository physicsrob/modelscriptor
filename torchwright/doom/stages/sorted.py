"""SORTED stage: autoregressive argmin sort + per-wall visibility column range.

At each SORTED_WALL token the graph:

1. Runs ``attend_argmin_valid_unmasked`` over WALL positions using the
   BSP rank as the score, gated by the per-wall ``is_renderable`` flag
   and masked by the running ``prev_mask``.  Emits the selected wall's
   payload (geometry + render precomp + position one-hot) plus the
   updated mask.
2. Derives the selection's **sort rank** — literally the number of
   walls already picked (``sum(prev_mask)``) — which the THINKING
   stage uses as a second-order sort key (walls are picked in BSP
   order, so sort_rank == front-to-back index).
3. Computes the **visibility column range** ``(vis_lo, vis_hi)`` that
   the selected wall subtends on screen, by rotating endpoints into
   the player's frame and mapping ``tan(angle) → column`` via the FOV.

Downstream consumers:

* **THINKING** uses ``sort_rank`` as score, ``sort_rank_onehot`` as
  position key, and ``[gated_render_data, vis_lo, vis_hi,
  sort_rank_onehot]`` as value — one atomic attention pick per wall.
* **Orchestrator output** packs ``[E8_SORTED_WALL, sel_wall_data,
  sort_rank, vis_lo, vis_hi, sel_onehot, updated_mask]`` into the
  sort_feedback field that closes the autoregressive sort loop.
"""

import builtins
import math
from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.asserts import (
    assert_distinct_across,
    assert_picked_from,
    assert_score_gap_at_least,
)
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    clamp,
    compare,
    max as max_node,
    min as min_node,
    multiply_2d,
    multiply_const,
    piecewise_linear_2d,
    reciprocal,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_valid_unmasked
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.graph_constants import DIFF_BP, TRIG_BP
from torchwright.doom.renderer import trig_lookup
from torchwright.doom.wall_payload import (
    extract_geometry_field,
    unpack_wall_payload,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class SortedInputs:
    # WALL-stage outputs (per-WALL-position Nodes read via attention).
    sort_score: Node        # clean integer BSP rank at WALL positions
    is_renderable: Node     # ±1 validity: true iff wall is real + not parallel + in front
    position_onehot: Node
    sort_value: Node

    # Host-fed running mask of already-picked walls.
    prev_mask: Node

    # Resolved player state broadcast from EOS.
    eos_px: Node
    eos_py: Node
    eos_angle: Node

    # Token-type flags.  ``is_wall`` is consumed only by the attention
    # assertions (keys-validity) — the argmin itself already ignores
    # non-WALL positions via their sentinel scores.
    is_sorted: Node
    is_wall: Node

    pos_encoding: PosEncoding


@dataclass
class SortedOutputs:
    # Per-SORTED geometry + mask pieces.
    sel_wall_data: Node       # ax, ay, bx, by, tex_id  (5-wide)
    sel_onehot: Node          # per-position position_onehot of picked wall
    updated_mask: Node        # prev_mask + sel_onehot, fed back by host

    # Sort rank: number of walls already picked (== 0 on the first pick,
    # == max_walls-1 on the last).  sort_rank_onehot one-hot-encodes it
    # for downstream attention.
    sort_rank: Node
    sort_rank_onehot: Node

    # Visibility column range on screen (floats, already clamped to
    # ``[-2, W+2]`` by the atan piecewise).
    vis_lo: Node
    vis_hi: Node

    # Render data gated to zero at non-SORTED positions so THINKING's
    # attention value sums cleanly.  6-wide:
    # ``[sort_den, C, D, E, H_inv, tex_id]``.
    gated_render_data: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_sorted(
    inputs: SortedInputs,
    config: RenderConfig,
    max_walls: int,
    max_coord: float,
) -> SortedOutputs:
    with annotate("sort/attention"):
        (
            sel_wall_data,
            sel_onehot,
            updated_mask,
            gated_render_data,
            sort_rank,
            sort_rank_onehot,
        ) = _argmin_and_derive(inputs, max_walls)

    with annotate("sort/visibility"):
        vis_lo, vis_hi = _compute_visibility_columns(
            sel_wall_data,
            inputs.eos_px, inputs.eos_py, inputs.eos_angle,
            config=config,
            max_coord=max_coord,
        )

    return SortedOutputs(
        sel_wall_data=sel_wall_data,
        sel_onehot=sel_onehot,
        updated_mask=updated_mask,
        sort_rank=sort_rank,
        sort_rank_onehot=sort_rank_onehot,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        gated_render_data=gated_render_data,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _argmin_and_derive(inputs: SortedInputs, max_walls: int):
    """Pick the nearest unmasked wall + derive sort_rank + gate render data.

    Three invariants are asserted around the argmin:

    * ``sort_score`` values at WALL positions must be pairwise distinct
      (``assert_distinct_across``) — tied ranks would make the softmax
      blend walls.  Sort scores are clean integer BSP ranks (1.0 gaps),
      so the 0.8 margin has plenty of headroom.
    * The two smallest valid scores must differ by at least the
      softmax-resolvability margin (``assert_score_gap_at_least``).
      With unit-integer rank spacing the 0.5 margin is comfortably met.
    * The attention output must match exactly one ``sort_value`` row
      from a valid (WALL) position (``assert_picked_from``).  Reference
      math's exact softmax always picks; this assertion is for the
      compile-side probe, where the piecewise-linear softmax can blend
      near-ties.
    """
    # Pre-attention: scores at WALL positions must be pairwise distinct.
    checked_score = assert_distinct_across(
        inputs.sort_score, inputs.is_wall, margin=0.8,
    )
    checked_score = assert_score_gap_at_least(
        checked_score, inputs.is_wall, margin=0.5,
    )

    # TODO(end-of-sort): when ``N_renderable < max_walls``, after all
    # renderable walls are picked the masked-valid fallback re-picks the
    # last-picked wall.  RENDER then overdraws; the host's per-pixel
    # dedup (``filled[y, c]``) hides the waste.  A principled fix is a
    # compiled "done" signal that SORTED emits once all renderables are
    # exhausted, letting downstream stages short-circuit instead of
    # repeating work.
    selected_sort = attend_argmin_valid_unmasked(
        pos_encoding=inputs.pos_encoding,
        score=checked_score,
        validity=inputs.is_renderable,
        mask_vector=inputs.prev_mask,
        position_onehot=inputs.position_onehot,
        value=inputs.sort_value,
    )

    # Post-attention: result must match exactly one value row from a
    # WALL position (within atol).  This rarely fires at reference eval
    # but catches compile-side softmax blending via
    # ``check_asserts_on_compiled``.
    selected_sort = assert_picked_from(
        selected_sort, inputs.sort_value, inputs.is_wall, atol=0.01,
    )
    unpacked = unpack_wall_payload(selected_sort, max_walls)
    sel_wall_data = unpacked.wall_data
    sel_render = unpacked.render_data
    sel_onehot = unpacked.onehot
    sel_tex_id = extract_geometry_field(sel_wall_data, "tex_id")
    updated_mask = add(inputs.prev_mask, sel_onehot)

    # Gate so non-SORTED positions contribute 0 to THINKING's attention.
    gated_render_data = cond_gate(
        inputs.is_sorted, Concatenate([sel_render, sel_tex_id])
    )

    # sort_rank = sum of prev_mask bits.  At the k-th SORTED token in the
    # autoregressive loop, k walls are already masked, so sort_rank == k.
    # Implement with a constant-weight Linear (reduce-sum).
    sort_rank = Linear(
        inputs.prev_mask, torch.ones(max_walls, 1), name="sort_rank",
    )

    # One-hot encoding of sort_rank for THINKING's position key.
    sort_rank_p1 = add_const(sort_rank, 1.0)
    sr_onehot_bool = in_range(sort_rank, sort_rank_p1, max_walls)
    ones_oh = create_literal_value(torch.ones(max_walls), name="sr_ones")
    sort_rank_onehot = add_scaled_nodes(0.5, sr_onehot_bool, 0.5, ones_oh)

    return (
        sel_wall_data, sel_onehot, updated_mask, gated_render_data,
        sort_rank, sort_rank_onehot,
    )


def _compute_visibility_columns(
    sel_wall_data: Node,
    eos_px: Node,
    eos_py: Node,
    eos_angle: Node,
    *,
    config: RenderConfig,
    max_coord: float,
):
    """Columns subtended by the selected wall from the player's pose.

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

    **Why clip rather than project-endpoints-and-clamp:** endpoint
    projection cannot distinguish two geometrically different cases
    when the segment crosses the viewing plane (one endpoint
    ``dot > 0``, the other ``dot < 0``):

    * wall's visible portion actually crosses the FOV interior, OR
    * wall's visible portion skirts around the FOV off-screen.

    Both scenarios project the endpoints to the same clamp edges but
    should yield opposite vis-ranges (full-screen vs. empty).  Proper
    clipping — what DOOM's ``R_ClipWallSegment`` does — has the
    information to produce the right answer because it works with the
    segment's interior, not just its endpoints.
    """
    W = config.screen_width
    fov = config.fov_columns

    sel_ax = extract_geometry_field(sel_wall_data, "ax")
    sel_ay = extract_geometry_field(sel_wall_data, "ay")
    sel_bx = extract_geometry_field(sel_wall_data, "bx")
    sel_by = extract_geometry_field(sel_wall_data, "by")

    dax = subtract(sel_ax, eos_px)
    day = subtract(sel_ay, eos_py)
    dbx = subtract(sel_bx, eos_px)
    dby = subtract(sel_by, eos_py)

    sort_cos, sort_sin = trig_lookup(eos_angle)

    cross_a, dot_a = _rotate_into_player_frame(sort_cos, sort_sin, dax, day, "a")
    cross_b, dot_b = _rotate_into_player_frame(sort_cos, sort_sin, dbx, dby, "b")

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
    _COMPARE_SCALE_EMPTY = 100.0
    is_empty = compare(
        multiply_const(subtract(t_lo, t_hi), _COMPARE_SCALE_EMPTY), 0.0,
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
    #
    # Scale the t-difference before compare so the ±1 saturation is clean
    # when contributions are close (e.g., t_lo_L=0 vs t_lo_R=0.09).  The
    # compare's built-in ramp width is ~1/step_sharpness ≈ 0.1 in input
    # units, which is too wide when t values naturally cluster near
    # [0, 1] boundaries.  Scaling by 100 shrinks the ramp to ~0.001 in
    # t-space — tight enough that real differences always saturate.
    _COMPARE_SCALE = 100.0

    a_inside_L = compare(f_L_a, 0.0)
    a_inside_R = compare(f_R_a, 0.0)
    a_clipped_on_L = compare(
        multiply_const(subtract(t_lo_L, t_lo_R), _COMPARE_SCALE), 0.0,
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
        multiply_const(subtract(t_hi_R, t_hi_L), _COMPARE_SCALE), 0.0,
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
    vis_lo = select(is_empty, sentinel, vis_lo_visible)
    vis_hi = select(is_empty, sentinel, vis_hi_visible)

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


def _rotate_into_player_frame(cos_p: Node, sin_p: Node, dx: Node, dy: Node, suffix: str):
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


def _endpoint_to_column(
    cross: Node, dot: Node, *, W: int, fov: int, max_coord: float, suffix: str,
) -> Node:
    """Convert ``(cross, dot)`` to a signed screen column.

    Front case uses a fused ``piecewise_linear_2d(cross, |dot|)`` lookup
    that replaces the prior ``reciprocal → multiply_2d →
    piecewise_linear(atan)`` chain.  The behind case (``dot < 0``) is a
    single select on ``sign(dot)`` afterwards, reflecting the column
    across screen centre just like the prior code did — keeping it out
    of the piecewise avoids the steep atan discontinuity at ``dot = 0``
    that otherwise ruins the least-squares fit of the fused table.

    The fused front lookup is clamped inside the fn so the
    least-squares sees a bounded target function.  The grid is tight
    on the active region only; for (cross, |dot|) pairs outside the
    grid the piecewise holds edge values, which already sit at the
    clamp limits — no hyperplane budget spent on trivial constants.

    Motivation: the old chain compounded three discretisations; its
    ``multiply_2d`` at ``step=0.5`` drove ~0.06 absolute error in
    ``tan``, which atan's ~40 col/tan-unit slope near zero amplified
    into ~1-col drift at oblique angles.  One well-conditioned 2D
    table avoids the cascade.
    """
    col_lo, col_hi = -2.0, float(W + 2)
    fov_rad = float(fov) * math.pi / 128.0
    col_scale = float(W) / fov_rad
    half_W = float(W) / 2.0

    def _col_front_of(cr: float, dt_abs: float) -> float:
        # Clamp inside the fn so the least-squares sees a bounded
        # target — the alternative (unclamped atan, final clamp after
        # the select) drifts further from the clamped boundary because
        # the piecewise fit tries to track values that go well past the
        # screen edges.
        col = math.atan(cr / dt_abs) * col_scale + half_W
        return builtins.max(col_lo, builtins.min(col_hi, col))

    # Clamp the piecewise's inputs to the grid bounds — piecewise_linear_2d's
    # extrapolation outside the grid is ill-behaved, but any
    # ``(cross, |dot|)`` that would have projected to a column outside
    # ``[-2, W+2]`` already maps to a clamped boundary value inside the
    # grid, so clamping the inputs is lossless.
    bp_cross_lo = _COL_FOLD_BP_CROSS[0]
    bp_cross_hi = _COL_FOLD_BP_CROSS[-1]
    bp_dot_lo = _COL_FOLD_BP_DOT_ABS[0]
    bp_dot_hi = _COL_FOLD_BP_DOT_ABS[-1]

    dot_sign = compare(dot, 0.0)
    abs_dot = abs(dot)
    cross_clamped = clamp(cross, bp_cross_lo, bp_cross_hi)
    dot_pos = clamp(abs_dot, bp_dot_lo, bp_dot_hi)

    col_front = piecewise_linear_2d(
        cross_clamped, dot_pos,
        _COL_FOLD_BP_CROSS, _COL_FOLD_BP_DOT_ABS,
        _col_front_of,
        name=f"col_front_{suffix}",
    )

    # Behind-flip: ``col_back = W − col_front`` (reflection across
    # centre), matching the prior implementation.
    col_back = Linear(
        col_front, torch.tensor([[-1.0]]),
        torch.tensor([float(W)]), name=f"col_{suffix}_back",
    )
    col_final = select(dot_sign, col_front, col_back)

    return clamp(col_final, col_lo, col_hi)
