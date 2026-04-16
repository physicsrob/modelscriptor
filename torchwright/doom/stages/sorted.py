"""SORTED stage: autoregressive argmin sort over WALL positions.

At each SORTED_WALL token the graph runs
``attend_argmin_valid_unmasked`` over WALL positions using the BSP rank
as the score, gated by the per-wall ``is_renderable`` flag and masked
by the running ``prev_mask``.  Emits the selected wall's payload
(geometry + render precomp + visibility columns + position one-hot)
plus the updated mask.

Visibility column computation (``vis_lo``, ``vis_hi``) lives in the
WALL stage and travels through the payload — SORTED just unpacks it.

Downstream consumers:

* **THINKING** uses ``sel_bsp_rank`` (from the payload) as score and
  ``sel_onehot`` (wall-index one-hot) as position key.
* **Orchestrator output** packs ``[E8_SORTED_WALL, sel_wall_data,
  sel_bsp_rank, vis_lo, vis_hi, sel_onehot, updated_mask]`` into the
  sort_feedback field that closes the autoregressive sort loop.
"""

from dataclasses import dataclass

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.asserts import (
    assert_01,
    assert_distinct_across,
    assert_onehot,
    assert_picked_from,
    assert_score_gap_at_least,
)
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import add, compare, subtract
from torchwright.ops.attention_ops import attend_argmin_valid_unmasked
from torchwright.ops.logic_ops import cond_gate

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
    sort_score: Node        # clean integer BSP rank at WALL positions
    is_renderable: Node     # ±1 validity: true iff wall is real + not parallel + in front
    position_onehot: Node
    sort_value: Node

    # Host-fed running mask of already-picked walls.
    prev_mask: Node

    # BSP rank selected at the previous SORTED step (scalar, initialized
    # to -1 at the EOS seed).  Used to detect sort exhaustion: when the
    # masked-valid fallback re-picks the same wall, sel_bsp_rank <=
    # prev_bsp_rank.
    prev_bsp_rank: Node

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

    # BSP rank of the selected wall — used by THINKING as the score
    # for its own argmin (picks walls in front-to-back order).
    sel_bsp_rank: Node

    # Sort exhaustion flag: +1 when sel_bsp_rank <= prev_bsp_rank
    # (the masked-valid fallback re-picked the same or earlier wall),
    # -1 when the sort is still making progress.
    sort_done: Node

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
    max_walls: int,
) -> SortedOutputs:
    with annotate("sort/attention"):
        (
            sel_wall_data,
            sel_onehot,
            updated_mask,
            gated_render_data,
            sel_bsp_rank,
            sort_done,
            vis_lo,
            vis_hi,
        ) = _argmin_and_derive(inputs, max_walls)

    return SortedOutputs(
        sel_wall_data=sel_wall_data,
        sel_onehot=sel_onehot,
        updated_mask=updated_mask,
        sel_bsp_rank=sel_bsp_rank,
        sort_done=sort_done,
        vis_lo=vis_lo,
        vis_hi=vis_hi,
        gated_render_data=gated_render_data,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _argmin_and_derive(inputs: SortedInputs, max_walls: int):
    """Pick the nearest unmasked wall + gate render data.

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
        checked_score, inputs.is_wall, margin=1.0,
    )

    selected_sort = attend_argmin_valid_unmasked(
        pos_encoding=inputs.pos_encoding,
        score=checked_score,
        validity=inputs.is_renderable,
        mask_vector=assert_01(inputs.prev_mask),
        position_onehot=assert_onehot(inputs.position_onehot),
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
    sel_bsp_rank = unpacked.bsp_rank
    sel_onehot = unpacked.onehot
    sel_tex_id = extract_geometry_field(sel_wall_data, "tex_id")
    updated_mask = add(inputs.prev_mask, sel_onehot)

    # Extract vis_lo/vis_hi from the payload (computed at WALL stage).
    vis_lo = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 0, 1, "vis_lo")
    vis_hi = extract_from(unpacked.vis_cols, VISIBILITY_WIDTH, 1, 1, "vis_hi")

    # Sort exhaustion: when the masked-valid fallback re-picks the same
    # wall (sel_bsp_rank <= prev_bsp_rank), the sort has exhausted all
    # renderable walls.  BSP ranks are unique integers with strictly
    # increasing picks, so equality means re-pick.  The -0.5 threshold
    # cleanly separates the integer cases (diff=0 → done, diff=-1 → active).
    sort_done = compare(subtract(inputs.prev_bsp_rank, sel_bsp_rank), -0.5)

    # Gate so non-SORTED positions contribute 0 to THINKING's attention.
    gated_render_data = cond_gate(
        inputs.is_sorted, Concatenate([sel_render, sel_tex_id])
    )

    return (
        sel_wall_data, sel_onehot, updated_mask, gated_render_data,
        sel_bsp_rank, sort_done, vis_lo, vis_hi,
    )
