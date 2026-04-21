"""SORTED stage: autoregressive argmin sort over WALL positions.

At each SORTED_WALL token the graph runs
``attend_argmin_above_integer`` over WALL positions.  The score is
the per-wall BSP rank.  Each SORTED token's threshold is derived
from its ``position_index`` (host-fed, 0-indexed): the token at
position i uses threshold slot i, picking the smallest
``bsp_rank > i - 1`` among renderable walls.  No feedback from
previous SORTED tokens is needed.

``indicators_above`` (computed at WALL time, gated by
``is_renderable``) is the key-side indicator basis: slot ``c`` is
1 iff the key's ``bsp_rank > c - 1`` AND the wall is renderable.

Visibility column computation (``vis_lo``, ``vis_hi``) lives in the
WALL stage and travels through the payload — SORTED just unpacks it.

Downstream consumers:

* The **host** caches ``sel_onehot``, ``vis_lo``, ``vis_hi``, and
  ``sel_tex_id`` from the overlaid outputs at SORTED positions to
  feed wall identity into RENDER on wall transitions.

**Exhaustion.**  When the threshold exceeds every renderable wall's
BSP rank (``N_renderable`` walls picked after ``N_renderable`` steps),
``attend_argmin_above_integer`` returns a softmax-averaged garbage
value per its documented contract.  We detect the exhausted state
cheaply via ``sort_done = compare(position_index - sel_bsp_rank, 0.5)``
so downstream stages can skip garbage picks.  The threshold slot is
clamped to ``[0, max_walls - 1]`` so the attention always has a
valid ``threshold_onehot`` column, avoiding the all-zero degeneracy.
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Node, annotate
from torchwright.graph.asserts import (
    assert_distinct_across,
    assert_integer,
    assert_onehot,
    assert_picked_from,
    assert_score_gap_at_least,
)
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_const,
    add_scaled_nodes,
    clamp,
    compare,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_above_integer
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.map_select import in_range

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
class SortedToken:
    """Overlay fields at each SORTED_WALL position."""

    position_index: Node  # sort_position_index (0-indexed, host copies from overlay)


@dataclass
class SortedKVInput:
    """Per-WALL-position values read via attention."""

    sort_score: Node  # clean integer BSP rank at WALL positions
    sort_value: Node  # packed wall payload (fed as attention value)
    indicators_above: Node  # max_walls-wide thermometer key-side indicator


@dataclass
class SortedTokenOutput:
    """Overlay + overflow outputs at SORTED positions."""

    sel_onehot: Node   # overlay → render_wall_j_onehot
    vis_lo: Node       # overlay → render_vis_lo (also seeds render_col)
    vis_hi: Node       # overlay → render_vis_hi
    sel_tex_id: Node   # overlay → render_tex_id
    sort_done: Node    # overflow — host reads
    next_sort_position_index: Node  # overlay → sort_position_index


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_sorted(
    token: SortedToken,
    kv: SortedKVInput,
    *,
    is_sorted: Node,
    is_wall: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
) -> SortedTokenOutput:
    with annotate("sort/threshold"):
        threshold_onehot = _compute_threshold_onehot(
            token.position_index,
            max_walls,
        )

    with annotate("sort/attention"):
        (
            sel_onehot,
            sel_tex_id,
            sort_done,
            vis_lo,
            vis_hi,
        ) = _argmin_above_and_derive(
            token, kv, is_wall, pos_encoding, threshold_onehot, max_walls
        )

    next_sort_position_index = add_const(token.position_index, 1.0)

    return SortedTokenOutput(
        sel_onehot=sel_onehot,
        sort_done=sort_done,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        sel_tex_id=sel_tex_id,
        next_sort_position_index=next_sort_position_index,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _compute_threshold_onehot(position_index: Node, max_walls: int) -> Node:
    """Convert ``position_index ∈ {0, 1, …, max_walls - 1}`` to a
    width-``max_walls`` ``{0, 1}`` one-hot.

    Slot ``c`` is 1 iff ``position_index == c`` (clamped to
    ``[0, max_walls - 1]``).  The attention's rendezvous with
    ``indicators_above[c] = I(bsp_rank > c - 1 AND is_renderable)`` then
    picks the smallest ``bsp_rank`` strictly greater than
    ``position_index - 1`` among renderable walls.

    SORTED token at position i uses threshold slot i, which picks the
    wall with the i-th smallest BSP rank among renderable walls.
    """
    clamped = clamp(position_index, 0.0, float(max_walls - 1))
    clamped_p1 = add_const(clamped, 1.0)
    onehot_bool = in_range(clamped, clamped_p1, max_walls)  # ±1
    ones = create_literal_value(
        torch.ones(max_walls),
        name="threshold_ones",
    )
    return assert_onehot(
        add_scaled_nodes(0.5, onehot_bool, 0.5, ones),
    )


def _argmin_above_and_derive(
    token: SortedToken,
    kv: SortedKVInput,
    is_wall: Node,
    pos_encoding: PosEncoding,
    threshold_onehot: Node,
    max_walls: int,
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
        kv.sort_score,
        is_wall,
        margin=0.8,
    )
    checked_score = assert_score_gap_at_least(
        checked_score,
        is_wall,
        margin=1.0,
    )

    selected_sort = attend_argmin_above_integer(
        pos_encoding=pos_encoding,
        score=checked_score,
        indicators_above=kv.indicators_above,
        threshold_onehot=threshold_onehot,
        value=kv.sort_value,
        assert_hardness_gt=0.99,
    )

    # Post-attention: result must match exactly one value row from a
    # WALL position (within atol).  Catches compile-side softmax
    # blending when keys tie.
    selected_sort = assert_picked_from(
        selected_sort,
        kv.sort_value,
        is_wall,
        atol=0.01,
    )

    unpacked = unpack_wall_payload(selected_sort, max_walls)
    sel_bsp_rank = unpacked.bsp_rank
    sel_onehot = unpacked.onehot
    sel_tex_id = extract_geometry_field(unpacked.wall_data, "tex_id")

    vis_lo = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 0, 1, "vis_lo")
    vis_hi = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 1, 1, "vis_hi")

    raw_sel_bsp_rank = assert_integer(sel_bsp_rank)
    sort_done = compare(
        subtract(token.position_index, raw_sel_bsp_rank),
        0.5,
    )

    return (
        sel_onehot,
        sel_tex_id,
        sort_done,
        vis_lo,
        vis_hi,
    )
