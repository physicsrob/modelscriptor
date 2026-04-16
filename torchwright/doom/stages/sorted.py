"""SORTED stage: autoregressive argmin sort over WALL positions.

At each SORTED_WALL token the graph runs
``attend_argmin_above_integer`` over WALL positions.  The score is
the per-wall BSP rank.  The threshold carried by each query is
``prev_bsp_rank`` (a scalar fed back from the previous SORTED step),
so the attention picks the smallest ``bsp_rank > prev_bsp_rank``
among renderable walls.  ``indicators_above`` (computed at WALL
time, gated by ``is_renderable``) is the key-side indicator basis:
slot ``c`` is 1 iff the key's ``bsp_rank > c - 1`` AND the wall is
renderable.

**Why thresholds not masks.**  The V1-style above-threshold pattern
carries one scalar (``prev_bsp_rank``) through the sort_feedback
rather than a ``max_walls``-wide running mask.  That narrows the
feedback, drops the mask/position-onehot rendezvous from the
attention, and turns exhaustion into a clean integer threshold that
goes out of range when all renderable walls are picked.

Visibility column computation (``vis_lo``, ``vis_hi``) lives in the
WALL stage and travels through the payload — SORTED just unpacks it.

Downstream consumers:

* **THINKING** uses ``sel_bsp_rank`` (from the payload) as score and
  ``sel_onehot`` (wall-index one-hot) as position key.
* **Orchestrator output** packs ``[E8_SORTED_WALL, sel_wall_data,
  sel_bsp_rank, vis_lo, vis_hi, sel_onehot]`` into the sort_feedback
  field that closes the autoregressive sort loop.  No ``updated_mask``
  — the threshold-based attention needs only the scalar
  ``prev_bsp_rank``, extracted from the ``sel_bsp_rank`` slot.

**Exhaustion.**  When the threshold exceeds every renderable wall's
BSP rank (``N_renderable`` walls picked after ``N_renderable`` steps),
``attend_argmin_above_integer`` returns a softmax-averaged garbage
value per its documented contract.  We detect the exhausted state
cheaply via ``sort_done = compare(prev_bsp_rank - sel_bsp_rank, -0.5)``
so downstream stages can skip garbage picks.  The threshold slot is
clamped to ``[0, max_walls - 1]`` so the attention always has a
valid ``threshold_onehot`` column, avoiding the all-zero degeneracy.
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.asserts import (
    assert_distinct_across,
    assert_integer,
    assert_onehot,
    assert_picked_from,
    assert_score_gap_at_least,
)
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_const, add_scaled_nodes, clamp, compare, subtract,
)
from torchwright.ops.attention_ops import attend_argmin_above_integer
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import in_range, select

from torchwright.doom.graph_utils import extract_from
from torchwright.doom.wall_payload import (
    VISIBILITY_WIDTH,
    extract_geometry_field,
    unpack_wall_payload,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class SortedInputs:
    # WALL-stage outputs (per-WALL-position Nodes read via attention).
    sort_score: Node         # clean integer BSP rank at WALL positions
    sort_value: Node         # packed wall payload (fed as attention value)
    indicators_above: Node   # max_walls-wide thermometer key-side indicator

    # BSP rank selected at the previous SORTED step (scalar, initialized
    # to -1 at the EOS seed).  Threshold for the next pick = prev_bsp_rank,
    # so we want keys with bsp_rank strictly greater than prev_bsp_rank.
    prev_bsp_rank: Node

    # Token-type flags used by the pre-attention score-distinctness
    # assertions.  ``is_wall`` restricts the valid-subset check to WALL
    # positions.  ``is_sorted`` gates downstream THINKING-visible nodes.
    is_sorted: Node
    is_wall: Node

    pos_encoding: PosEncoding


@dataclass
class SortedOutputs:
    # Per-SORTED geometry pieces.
    sel_wall_data: Node       # ax, ay, bx, by, tex_id  (5-wide)
    sel_onehot: Node          # per-position wall-index one-hot of picked wall

    # BSP rank of the selected wall — used by THINKING as the score
    # for its own argmin (picks walls in front-to-back order) and as the
    # ``prev_bsp_rank`` for the next SORTED step's threshold.
    sel_bsp_rank: Node

    # Sort exhaustion flag: +1 when sel_bsp_rank <= prev_bsp_rank
    # (no renderable wall strictly above the threshold — the attention
    # softmax-averaged garbage), -1 when the sort is still making progress.
    sort_done: Node

    # Visibility column range on screen (floats, already clamped to
    # ``[-2, W+2]`` by the atan piecewise, gated to 0 on non-renderable
    # walls at WALL time).
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
    max_walls: int,
) -> SortedOutputs:
    with annotate("sort/threshold"):
        threshold_onehot = _compute_threshold_onehot(
            inputs.prev_bsp_rank, max_walls,
        )

    with annotate("sort/attention"):
        (
            sel_wall_data,
            sel_onehot,
            gated_render_data,
            sel_bsp_rank,
            sort_done,
            vis_lo,
            vis_hi,
        ) = _argmin_above_and_derive(inputs, threshold_onehot, max_walls)

    return SortedOutputs(
        sel_wall_data=sel_wall_data,
        sel_onehot=sel_onehot,
        sel_bsp_rank=sel_bsp_rank,
        sort_done=sort_done,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        gated_render_data=gated_render_data,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _compute_threshold_onehot(prev_bsp_rank: Node, max_walls: int) -> Node:
    """Convert scalar ``prev_bsp_rank ∈ {-1, 0, …, max_walls - 1}`` to a
    width-``max_walls`` ``{0, 1}`` one-hot.

    Slot ``c`` is 1 iff ``prev_bsp_rank + 1 == c`` (clamped to
    ``[0, max_walls - 1]``).  The attention's rendezvous with
    ``indicators_above[c] = I(bsp_rank > c - 1 AND is_renderable)`` then
    picks the smallest ``bsp_rank`` strictly greater than
    ``prev_bsp_rank`` among renderable walls.

    **Why clamp.**  After all renderable walls are picked,
    ``prev_bsp_rank`` reaches ``N_renderable - 1``, so the threshold
    slot would be ``N_renderable``.  If ``N_renderable < max_walls``
    the remaining slots are still valid one-hot positions, but if the
    threshold drifts past ``max_walls - 1`` the one-hot becomes
    all-zero and the attention degenerates (softmax over uniform
    logits → garbage).  Clamping to ``max_walls - 1`` keeps the
    attention in a well-defined state on exhausted steps — the pick
    is still garbage (no valid keys above the clamped threshold) but
    it's consistent-shape garbage that ``sort_done`` catches.

    Mirrors ``examples.sort_digits_v1._threshold_onehot`` but with
    a clamp for the exhaustion case.
    """
    # threshold = prev_bsp_rank + 1, clamped to [0, max_walls - 1].
    raw = add_const(prev_bsp_rank, 1.0)
    clamped = clamp(raw, 0.0, float(max_walls - 1))
    clamped_p1 = add_const(clamped, 1.0)
    onehot_bool = in_range(clamped, clamped_p1, max_walls)  # ±1
    ones = create_literal_value(
        torch.ones(max_walls), name="threshold_ones",
    )
    return assert_onehot(
        add_scaled_nodes(0.5, onehot_bool, 0.5, ones),
    )


def _argmin_above_and_derive(
    inputs: SortedInputs, threshold_onehot: Node, max_walls: int,
):
    """Pick the smallest-BSP-rank renderable wall strictly above the
    threshold, and derive payload fields + exhaustion signal.

    Three invariants are asserted around the argmin:

    * ``sort_score`` values at WALL positions must be pairwise distinct
      (``assert_distinct_across``) — tied ranks would make the softmax
      blend walls.  Sort scores are clean integer BSP ranks (1.0 gaps),
      so the 0.8 margin has plenty of headroom.
    * The two smallest valid scores must differ by at least the
      softmax-resolvability margin (``assert_score_gap_at_least``).
      With unit-integer rank spacing the 1.0 margin is comfortably met.
    * The attention output must match exactly one ``sort_value`` row
      from a valid (WALL) position (``assert_picked_from``).  Reference
      math's exact softmax always picks; this assertion is for the
      compile-side probe, where the piecewise-linear softmax can blend
      near-ties.
    """
    checked_score = assert_distinct_across(
        inputs.sort_score, inputs.is_wall, margin=0.8,
    )
    checked_score = assert_score_gap_at_least(
        checked_score, inputs.is_wall, margin=1.0,
    )

    selected_sort = attend_argmin_above_integer(
        pos_encoding=inputs.pos_encoding,
        score=checked_score,
        indicators_above=inputs.indicators_above,
        threshold_onehot=threshold_onehot,
        value=inputs.sort_value,
    )

    # Post-attention: result must match exactly one value row from a
    # WALL position (within atol).  Catches compile-side softmax
    # blending when keys tie.
    selected_sort = assert_picked_from(
        selected_sort, inputs.sort_value, inputs.is_wall, atol=0.01,
    )

    unpacked = unpack_wall_payload(selected_sort, max_walls)
    sel_wall_data = unpacked.wall_data
    sel_render = unpacked.render_data
    sel_bsp_rank = unpacked.bsp_rank
    sel_onehot = unpacked.onehot
    sel_tex_id = extract_geometry_field(sel_wall_data, "tex_id")

    # Extract vis_lo/vis_hi from the payload (computed at WALL stage).
    vis_lo = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 0, 1, "vis_lo")
    vis_hi = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 1, 1, "vis_hi")

    # Sort exhaustion: during normal progress the attention picks a
    # wall with rank strictly greater than prev_bsp_rank, so
    # prev - sel <= -1.  Once exhausted, the softmax averages values
    # from below-threshold keys (column-0 domination by lowest-score
    # wall), so prev - sel becomes non-negative.  The -0.5 threshold
    # cleanly separates the cases.
    raw_sel_bsp_rank = assert_integer(sel_bsp_rank)
    sort_done = compare(
        subtract(inputs.prev_bsp_rank, raw_sel_bsp_rank),
        -0.5,
    )

    # Sentinel-ise sel_bsp_rank on exhausted steps so THINKING's
    # argmin (which uses sel_bsp_rank as score) ignores these cached
    # positions.  Without this, the softmax-averaged garbage rank on
    # exhausted steps is typically lower than real picks' ranks and
    # would win THINKING's argmin, pulling garbage wall data into
    # RENDER.  The sentinel also flows into the next SORTED step's
    # ``prev_bsp_rank`` feedback; the threshold clamp in
    # ``_compute_threshold_onehot`` handles that cleanly (threshold
    # pins at max_walls - 1 and successive steps stay exhausted).
    sort_done_sentinel = create_literal_value(
        torch.tensor([99.0]), name="sort_done_sentinel",
    )
    sel_bsp_rank_effective = select(
        sort_done, sort_done_sentinel, raw_sel_bsp_rank,
    )

    # Gate so non-SORTED positions contribute 0 to THINKING's attention.
    gated_render_data = cond_gate(
        inputs.is_sorted, Concatenate([sel_render, sel_tex_id]),
    )

    return (
        sel_wall_data, sel_onehot, gated_render_data,
        sel_bsp_rank_effective, sort_done, vis_lo, vis_hi,
    )
