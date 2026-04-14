"""SORTED stage: autoregressive argmin sort + per-wall visibility column range.

At each SORTED_WALL token the graph:

1. Runs ``attend_argmin_unmasked`` over WALL positions using the BSP
   rank as the score, masked by the running ``prev_mask``.  Emits the
   selected wall's payload (geometry + render precomp + position
   one-hot) plus the updated mask.
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

import math
from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.asserts import assert_distinct_across, assert_picked_from
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    add_scaled_nodes,
    clamp,
    compare,
    multiply_2d,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
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
    sort_score: Node        # bsp_rank at WALL positions, sentinel elsewhere
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

    Two invariants are asserted around the argmin:

    * ``sort_score`` values at WALL positions must be pairwise distinct
      (``assert_distinct_across``) — tied ranks would make the softmax
      blend walls.  This is the angle-192 class of bug; the check fires
      at reference-eval time before the attention runs.
    * The attention output must match exactly one ``sort_value`` row
      from a valid (WALL) position (``assert_picked_from``).  Reference
      math's exact softmax always picks; this assertion is for the
      compile-side probe, where the piecewise-linear softmax can blend
      near-ties.
    """
    # Pre-attention: scores at WALL positions must be pairwise distinct.
    checked_score = assert_distinct_across(
        inputs.sort_score, inputs.is_wall, margin=0.5,
    )

    selected_sort = attend_argmin_unmasked(
        pos_encoding=inputs.pos_encoding,
        score=checked_score,
        mask_vector=inputs.prev_mask,
        position_onehot=inputs.position_onehot,
        value=inputs.sort_value,
    )

    # Post-attention: result must match exactly one value row from a
    # WALL position (within atol).  This rarely fires at reference eval
    # but catches compile-side softmax blending via
    # ``check_asserts_on_compiled``.
    selected_sort = assert_picked_from(
        selected_sort, inputs.sort_value, inputs.is_wall, atol=1e-2,
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

    Wall endpoints A, B are translated into the player's frame, then
    ``tan(angle) = cross / dot`` is mapped to a screen column via the
    camera's FOV.  Returns ``(vis_lo, vis_hi)`` as floats already
    clamped to ``[-2, W+2]`` by the piecewise-linear atan step.

    Optimized tangent pipeline (6 sequential MLP sublayers per endpoint):
    ``abs(dot) → clamp(0.1, 2·max_coord) → reciprocal → multiply_2d(cross,
    inv) → atan_clamped → select(sign, col, W - col)``.  Key structural
    wins versus the naïve pipeline:

    * ``clamp(abs(.))`` subsumes compare+select (saves 1 sublayer).
    * ``multiply_2d`` subsumes signed_multiply for the cross*inv product.
    * The piecewise-linear atan fuses column scaling + column clamping
      into one step.
    * Sign handling is applied after atan so ``compare(dot, 0)`` runs in
      parallel with the main chain instead of serially.
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

    col_a_c = _endpoint_to_column(
        cross_a, dot_a, W=W, fov=fov, max_coord=max_coord, suffix="a",
    )
    col_b_c = _endpoint_to_column(
        cross_b, dot_b, W=W, fov=fov, max_coord=max_coord, suffix="b",
    )

    a_lt_b = compare(subtract(col_b_c, col_a_c), 0.0)
    vis_lo = select(a_lt_b, col_a_c, col_b_c)
    vis_hi = select(a_lt_b, col_b_c, col_a_c)
    return vis_lo, vis_hi


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


# Atan breakpoints with extended range outside ``±20`` so saturated dots
# don't produce column drift.  Matches main's tuned table.
_ATAN_BP_VIS = [
    -100, -20, -10, -5, -3, -2, -1.5, -1, -0.75, -0.5,
    -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 100,
]


def _endpoint_to_column(
    cross: Node, dot: Node, *, W: int, fov: int, max_coord: float, suffix: str,
) -> Node:
    """Convert ``(cross, dot)`` to a signed screen column.

    For ``dot > 0`` the point is in front: column is ``atan(cross/dot) *
    (W/fov_rad) + W/2``.  For ``dot < 0`` (behind the player) the
    naive tangent wraps, so we flip: the behind-column reflection is
    ``W - col_pos``.  The flip is applied via ``select(dot_sign, ...)``
    after the main chain so the sign compare parallelizes with it.
    """
    fov_rad = float(fov) * math.pi / 128.0
    col_from_tan_scale = float(W) / fov_rad
    half_W = float(W) / 2.0
    col_lo, col_hi = -2.0, float(W + 2)

    def _atan_clamped(t: float) -> float:
        return max(col_lo, min(col_hi, math.atan(t) * col_from_tan_scale + half_W))

    max_inv = 1.0 / 0.1  # reciprocal's max output when clamp floor is 0.1

    dot_sign = compare(dot, 0.0)                      # parallel with chain below
    dot_clamped = clamp(abs(dot), 0.1, 2.0 * max_coord)
    inv_dot = reciprocal(dot_clamped, min_value=0.1, max_value=2.0 * max_coord)
    tan_signed = multiply_2d(
        cross, inv_dot,
        max_abs1=max_coord, max_abs2=max_inv,
        step1=0.5, step2=0.5, min2=0.0,
    )
    col_pos = piecewise_linear(tan_signed, _ATAN_BP_VIS, _atan_clamped, name=f"col_{suffix}")
    col_neg = Linear(
        col_pos, torch.tensor([[-1.0]]),
        torch.tensor([float(W)]), name=f"col_{suffix}_neg",
    )
    return select(dot_sign, col_pos, col_neg)
