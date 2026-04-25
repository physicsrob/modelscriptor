"""THINKING_WALL stage: per-wall thinking-phase state machine with a
full 20-identifier cascade (17 per-wall + 3 RESOLVED at the frame
boundary).

Phase B Part 1 — the per-wall cascade splits the deep T/VIS math into
four new intermediate slots (T_STAR_L/R and COL_A/B) whose values flow
back to T_LO/T_HI/VIS_LO/VIS_HI through the KV cache.  This shallows
the per-step critical path from ~58 ops (13-slot cascade) to ~34 ops
(17-slot cascade).  Wire format per wall (35 steps):

    [THINKING_WALL_N]
      [BSP_RANK]         v(bsp_rank)      # integer 0..7
      [IS_RENDERABLE]    v(is_renderable) # 0 / 1
      [CROSS_A]          v(cross_a)       # signed offset, [-40, 40]
      [DOT_A]            v(dot_a)         # forward projection, [-40, 40]
      [CROSS_B]          v(cross_b)
      [DOT_B]            v(dot_b)
      [T_STAR_L]         v(t_star_L)      # left-plane clip, [-2, 2]
      [T_STAR_R]         v(t_star_R)
      [T_LO]             v(t_lo)          # clip parameter, [0, 1]
      [T_HI]             v(t_hi)
      [COL_A]            v(col_a)         # endpoint-A column, [-2, 122]
      [COL_B]            v(col_b)
      [VIS_LO]           v(vis_lo)        # screen col, [-2, 122]
      [VIS_HI]           v(vis_hi)
      [HIT_FULL]         v(hit_full_running_or) # 0 / 1, running OR
      [HIT_X]            v(hit_x_running_or)
      [HIT_Y]            v(hit_y_running_or)

HIT_FULL/HIT_X/HIT_Y emit the OR of this wall's flag with the previous
HIT_* value read from the KV cache — by the time wall 7's HIT_* fires,
each value is the global OR across all walls.  Wall 0 reads zero from
the empty cache (OR identity).  Part 3 wires RESOLVED to consume the
global HIT_* from wall 7 via ``get_value_after_last``.

Two classes of identifier computation:

* **Base values** (BSP_RANK, IS_RENDERABLE, CROSS/DOT, HIT_*): compute
  from first principles using the attended wall geometry, the BSP
  side-P vector, and the player's pre-collision pose.  Math is ported
  from ``wall.py`` 1:1.
* **Derived values** (T_STAR_L/R, T_LO/T_HI, COL_A/B, VIS_LO/VIS_HI):
  read CROSS/DOT/T/COL from the KV cache via :class:`ThinkingReadback`
  and apply the ported clip / projection math.  VIS gates on a
  locally-computed is_renderable (recomputed here rather than read via
  readback — the recompute is shallower than the round-trip through
  an attention hop + dequantize).

Cross-position data reads (three primary hops):

1. **Current wall identity.**  Most-recent ``THINKING_WALL_N`` marker
   → ``current_wall_index``.
2. **Wall geometry.**  Single ``attend_argmax_dot`` matching
   ``wall_j_onehot`` against WALL-position ``wall_position_onehot``;
   value block concatenates ``(wall_ax, wall_ay, wall_bx, wall_by,
   wall_bsp_coeffs, wall_bsp_const)``.  Fired at every per-wall
   identifier step via the ``is_any_identifier`` gate.
3. **Previous identifier slot** + **prior VALUE readbacks.**  The
   Part-2 prev-id attention plus :class:`ThinkingReadback`'s per-name
   attention to recent matching VALUE positions.
"""

from dataclasses import dataclass
from typing import List

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.asserts import assert_in_range, assert_integer
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    abs as abs_node,
    add,
    add_const,
    add_scaled_nodes,
    bool_to_01,
    clamp,
    compare,
    low_rank_2d,
    max as max_node,
    min as min_node,
    multiply_2d,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    square,
    subtract,
    sum_nodes,
)
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_most_recent_matching,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true, bool_any_true, cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

import math

from torchwright.doom.embedding import (
    D_EMBED,
    D_SLOT_ONEHOT,
    IDENTIFIER_NAMES,
    VALUE_RANGE_BY_NAME,
    _SLOT_ONEHOT_START,
    embed_lookup,
)
from torchwright.doom.graph_constants import (
    DIFF_BP,
    TRIG_BP,
    VEL_BP,
)
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.thinking_readback import (
    ThinkingReadback,
    build_thinking_readback,
    encode_value_binary,
)
from torchwright.ops.quantization import DEFAULT_N_LEVELS

# ---------------------------------------------------------------------------
# Slot indices for per-identifier machinery.  One constant per slot so the
# per-value wiring reads as ``is_identifier_by_slot[_SLOT_CROSS_A]`` rather
# than repeated ``IDENTIFIER_NAMES.index("CROSS_A")`` calls.
# ---------------------------------------------------------------------------

_SLOT_BSP_RANK = IDENTIFIER_NAMES.index("BSP_RANK")
_SLOT_IS_RENDERABLE = IDENTIFIER_NAMES.index("IS_RENDERABLE")
_SLOT_CROSS_A = IDENTIFIER_NAMES.index("CROSS_A")
_SLOT_DOT_A = IDENTIFIER_NAMES.index("DOT_A")
_SLOT_CROSS_B = IDENTIFIER_NAMES.index("CROSS_B")
_SLOT_DOT_B = IDENTIFIER_NAMES.index("DOT_B")
_SLOT_T_STAR_L = IDENTIFIER_NAMES.index("T_STAR_L")
_SLOT_T_STAR_R = IDENTIFIER_NAMES.index("T_STAR_R")
_SLOT_T_LO = IDENTIFIER_NAMES.index("T_LO")
_SLOT_T_HI = IDENTIFIER_NAMES.index("T_HI")
_SLOT_COL_A = IDENTIFIER_NAMES.index("COL_A")
_SLOT_COL_B = IDENTIFIER_NAMES.index("COL_B")
_SLOT_VIS_LO = IDENTIFIER_NAMES.index("VIS_LO")
_SLOT_VIS_HI = IDENTIFIER_NAMES.index("VIS_HI")
_SLOT_HIT_FULL = IDENTIFIER_NAMES.index("HIT_FULL")
_SLOT_HIT_X = IDENTIFIER_NAMES.index("HIT_X")
_SLOT_HIT_Y = IDENTIFIER_NAMES.index("HIT_Y")
_SLOT_RESOLVED_X = IDENTIFIER_NAMES.index("RESOLVED_X")
_SLOT_RESOLVED_Y = IDENTIFIER_NAMES.index("RESOLVED_Y")
_SLOT_RESOLVED_ANGLE = IDENTIFIER_NAMES.index("RESOLVED_ANGLE")

# Slot i → name of the next token to emit at the VALUE step whose most
# recent identifier was at slot i.  Slot ``_SLOT_HIT_Y`` is deliberately
# absent — its successor is wall-index-dependent (next marker or
# RESOLVED_X if last wall) and is handled separately.
_VALUE_SUCCESSOR_BY_SLOT = {
    _SLOT_BSP_RANK: "IS_RENDERABLE",
    _SLOT_IS_RENDERABLE: "CROSS_A",
    _SLOT_CROSS_A: "DOT_A",
    _SLOT_DOT_A: "CROSS_B",
    _SLOT_CROSS_B: "DOT_B",
    _SLOT_DOT_B: "T_STAR_L",
    _SLOT_T_STAR_L: "T_STAR_R",
    _SLOT_T_STAR_R: "T_LO",
    _SLOT_T_LO: "T_HI",
    _SLOT_T_HI: "COL_A",
    _SLOT_COL_A: "COL_B",
    _SLOT_COL_B: "VIS_LO",
    _SLOT_VIS_LO: "VIS_HI",
    _SLOT_VIS_HI: "HIT_FULL",
    _SLOT_HIT_FULL: "HIT_X",
    _SLOT_HIT_X: "HIT_Y",
    _SLOT_RESOLVED_X: "RESOLVED_Y",
    _SLOT_RESOLVED_Y: "RESOLVED_ANGLE",
    _SLOT_RESOLVED_ANGLE: "SORTED_WALL",
}

# Widened cond tolerance for ``select`` / ``cond_gate`` inside the
# visibility-projection chain — matches ``wall._VIS_C_TOL`` since the
# math is a 1:1 port.  See the comment on that constant in ``wall.py``
# for the provenance.
_VIS_C_TOL = 0.01
_T_COMPARE_SCALE = 100.0

# Breakpoint grid for the ``atan`` lookup in ``_endpoint_to_column``.
# Ported 1:1 from ``wall.py`` — same tolerance envelope.
_COL_FOLD_BP_CROSS = [
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.1,
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
]
_COL_FOLD_BP_DOT_ABS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class ThinkingWallKVInput:
    """Cross-position values every per-wall identifier step consumes.

    Wall geometry + BSP-rank coefficients live at WALL prompt
    positions, read via ``attend_argmax_dot`` on a per-wall one-hot.
    INPUT velocities / move trig / BSP side-P-vec are per-position
    broadcasts available at every position.
    """

    # From WALL prompt positions (read via attend_argmax_dot).
    wall_ax: Node
    wall_ay: Node
    wall_bx: Node
    wall_by: Node
    wall_index_at_wall: Node  # host-fed wall_index at WALL positions; the
    # first K channel of wall_geom_attention's
    # quadratic-equality match (Phase C Part 3).
    wall_index_neg_sq_at_wall: Node  # ``-wall_index²`` at WALL positions; second
    # K channel.  From WallKVOutput
    # (computed in WALL via one ``square``
    # sublayer; Phase C Part 1 added it).
    wall_bsp_coeffs: Node  # max_bsp_nodes-wide
    wall_bsp_const: Node  # 1-wide

    # From INPUT broadcast.
    vel_dx: Node
    vel_dy: Node
    move_cos: Node
    move_sin: Node

    # From BSP broadcast (per-position BSP-side indicator).
    side_P_vec: Node  # max_bsp_nodes-wide, ≈{0, 1} per component

    # From PLAYER broadcasts (pre-collision; collision math needs the
    # player's *intended* movement origin, which is the pre-collision
    # position).  Post-Part-4: the PLAYER broadcast is pre-collision —
    # the host feeds pre-collision game_state to PLAYER_X / PLAYER_Y.
    player_x: Node
    player_y: Node

    # From INPUT broadcast (post-turn, pre-collision).  Used by the
    # RESOLVED_ANGLE identifier — collision doesn't change angle, so
    # RESOLVED_ANGLE is a pass-through of the post-turn angle.
    new_angle: Node


@dataclass
class ThinkingWallOutput:
    """Outputs for the thinking-wall state machine.

    ``next_token_embedding`` is the 72-wide embedding of the next
    token the transformer should emit at this position.  Meaningful
    at thinking positions (marker / identifier / value); the
    orchestrator's ``is_thinking_active`` gate zeros it elsewhere.

    ``is_thinking_active`` is a ±1 flag marking "this position is
    marker, identifier, or value" — used by ``_assemble_output`` to
    decide whether to take this stage's next-token embedding or the
    SORTED/RENDER path.

    ``readback`` is the :class:`ThinkingReadback` instance that
    thinking-wall built internally.  Exposed so downstream stages
    (e.g. RENDER) can call ``get_value_after_last(...)`` to retrieve
    previously-emitted identifier values (e.g. RESOLVED_X/Y) from the
    KV cache — sharing the instance lets its per-name attention-head
    cache dedupe queries across stages.

    Phase B Part 2 adds two scalar KV channels specifically for the
    SORTED stage's quadratic-equality attention:

    * ``bsp_rank_scalar_for_sort`` — the wall's BSP rank as a float at
      BSP_RANK identifier positions of renderable walls; a sentinel
      (``-100``) at non-renderable walls and non-BSP_RANK positions.
    * ``bsp_rank_neg_sq_for_sort`` — ``-bsp_rank²`` at renderable
      BSP_RANK id positions; a sentinel (``-1000``) elsewhere.

    The sentinels are chosen small enough that the final select stays
    in a regime where ``select`` noise can't swamp the unit-step score
    gap between adjacent BSP ranks, yet large enough that the
    attention softmax (match_gain ≈ 20) puts effectively-zero weight
    on non-renderable / non-BSP_RANK keys.

    Phase B Part 2 also exposes the current-wall one-hot at thinking
    positions so downstream stages can do content-attention keyed by
    ``(identifier_name, wall_index)``.  Two variants are needed because
    consumers key different position types:

    * ``identifier_wall_index_onehot`` — gated by ``is_any_identifier``.
      The SORTED stage's quadratic-equality attention reads the V from
      BSP_RANK *identifier* positions, so its V-source channel needs
      the wall_index one-hot exposed at identifier positions.
    * ``value_wall_index_scalar`` / ``value_wall_index_neg_sq``
      (Phase C Part 3) — quadratic-equality K channels for content
      attentions keyed on ``(name-VALUE, wall_index)``: today
      ``render/vis_hi_content_attention`` and ``sort/vis_lo_attention``.
      Sentinel-gated to live at thinking-VALUE positions (sentinel
      -100 / -1000 elsewhere, same pattern as ``bsp_rank_*_for_sort``).

    All three are zero / sentinel outside the thinking phase.  At
    SORT_RESULT id / VALUE positions ``is_any_identifier`` /
    ``is_thinking_value`` technically fire (SORT_RESULT was added to
    ``IDENTIFIER_NAMES`` for uniform readback), but ``wall_j_onehot``
    there is stale thinking-phase state; sorted.py doesn't consume
    these fields at its own positions.
    """

    next_token_embedding: Node
    is_thinking_active: Node
    readback: "ThinkingReadback"
    bsp_rank_scalar_for_sort: Node
    bsp_rank_neg_sq_for_sort: Node
    identifier_wall_index_onehot: Node
    value_wall_index_scalar: Node
    value_wall_index_neg_sq: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_thinking_wall(
    kv: ThinkingWallKVInput,
    *,
    embedding: Node,
    is_wall: Node,
    is_thinking_wall_marker: Node,
    is_thinking_wall_n: list,
    is_any_identifier: Node,
    is_identifier_by_slot: List[Node],
    is_thinking_value: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
    max_coord: float,
    max_bsp_nodes: int,
    config: RenderConfig,
) -> ThinkingWallOutput:
    """Wire up the thinking-wall stage end to end."""
    assert max_walls <= 8, (
        "thinking_wall vocabulary defines 8 markers (THINKING_WALL_0..7); "
        f"max_walls={max_walls} would need additional vocabulary entries."
    )
    assert len(is_identifier_by_slot) == len(IDENTIFIER_NAMES) == 21, (
        f"expected 21 per-slot identifier detectors, got "
        f"{len(is_identifier_by_slot)}"
    )
    is_thinking_wall_n = is_thinking_wall_n[:max_walls]

    # ---------------------------------------------------------------------
    # Attention hop 1: current wall index from the most recent marker.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/marker_index"):
        wall_index_at_marker = sum_nodes(
            [
                multiply_const(bool_to_01(is_thinking_wall_n[i]), float(i))
                for i in range(max_walls)
            ]
        )

    with annotate("thinking_wall/find_current_wall"):
        query_const_1 = create_literal_value(
            torch.tensor([1.0]), name="tw_query_const_1"
        )
        gated_marker_wall_index = cond_gate(
            is_thinking_wall_marker, wall_index_at_marker
        )
        current_wall_index = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_const_1,
            key_vector=is_thinking_wall_marker,
            value=gated_marker_wall_index,
            match_gain=12000.0,
        )

    # ---------------------------------------------------------------------
    # Attention hop 2: most-recent-identifier slot one-hot.  Phase D
    # Part 1: V is the slot one-hot column block in W_EMBED — at each
    # IDENTIFIER row col ``_SLOT_ONEHOT_START + slot`` is +1 and the
    # other 20 cols are −1; non-identifier rows carry all −1.  The
    # extract folds into the V projection, so this attention has no
    # pre-computation other than the embedding leaf itself, dropping
    # the cond_gate + Concat + cond_gate chain that previously built
    # this V at depth 4.  ±1 storage matches the per-position
    # ``is_identifier_by_slot[i]`` extract semantics (same column),
    # so the same column does double duty as the per-position bool and
    # the cross-position prev-id V.  Downstream Linears that previously
    # took {0, 1} input have weights/bias adjusted to handle ±1 input
    # (see ``static_next_at_value`` in :func:`_compute_next_token_embedding`).
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/find_prev_identifier"):
        slot_onehot_raw = extract_from(
            embedding,
            D_EMBED,
            _SLOT_ONEHOT_START,
            D_SLOT_ONEHOT,
            "tw_slot_onehot",
        )
        # Tighten V's value_type to ±1 (the embedding leaf's value_type
        # is unknown; without this declaration downstream consumers see
        # an inflated bound and the cond_gate noise scales with that
        # bound, masking the actual ±1 V values).
        slot_onehot = assert_in_range(slot_onehot_raw, -1.0, 1.0)
        prev_slot_onehot_raw = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_const_1,
            key_vector=is_any_identifier,
            value=slot_onehot,
            match_gain=12000.0,
            assert_hardness_gt=0.99,
        )
        # Tighten the value_type to [-1, 1].  The V is ±1 cleanly from
        # W_EMBED and the attention is hard (assert_hardness_gt above),
        # but the compiler's value_range analyzer doesn't know that
        # without an explicit declaration — without this, downstream
        # cond_gate.M would inflate from 1 to whatever the embedding
        # leaf's broad value_type allows.
        prev_slot_onehot = assert_in_range(prev_slot_onehot_raw, -1.0, 1.0)

    # ---------------------------------------------------------------------
    # Phase C Part 3: derive the quadratic-equality K channels for
    # current_wall_index.  Used by this stage's wall_geom_attention
    # (Q-side) and exported as value_wall_index_scalar /
    # value_wall_index_neg_sq for downstream content attentions
    # (vis_hi at RENDER, vis_lo at SORTED).  ``square`` on a unit
    # grid is exact at integer wall_index ∈ [0, max_walls-1].
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/wall_index_quad"):
        wi_clamped = clamp(current_wall_index, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wi_clamped, 1.0)
        wall_j_onehot = bool_to_01(in_range(wi_clamped, wi_p1, max_walls))

        current_wall_index_sq = square(
            wi_clamped, max_value=float(max_walls - 1), step=1.0
        )
        current_wall_index_neg_sq = multiply_const(current_wall_index_sq, -1.0)

    # ---------------------------------------------------------------------
    # Attention hop 3: wall geometry for the current wall.
    #
    # Phase C Part 3: 2-wide quadratic-equality match on wall_index
    # (mirrors Phase C Part 1's render/wall_geom_attention).
    # Q at thinking-identifier positions: ``[2·current_wall_index, 1]``;
    # K at WALL positions: ``[wall_index, -wall_index²]`` (the
    # ``WallKVOutput`` channels Phase C Part 1 added); sentinels at
    # non-WALL positions keep softmax mass off them.
    #
    # V block packs (ax, ay, bx, by, bsp_coeffs..., bsp_const) so a
    # single attention head covers every per-wall identifier's
    # geometric need.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/wall_geom_attention"):
        # Q at identifier positions: [2·wall_j, 1].
        two_n = multiply_const(wi_clamped, 2.0)
        one_lit_q = create_literal_value(
            torch.tensor([1.0]), name="tw_wall_geom_q_one"
        )
        q_raw = Concatenate([two_n, one_lit_q])
        q_gated = cond_gate(is_any_identifier, q_raw)

        # K at WALL: [wall_index, -wall_index²]; sentinels elsewhere.
        # Same sentinel magnitudes (-100, -1000) as the rest of the
        # quadratic-equality system; approximate=False keeps the on-path
        # float-exact so the integer K survives without M·c_tol noise.
        sentinel_scalar = create_literal_value(
            torch.tensor([-100.0]), name="tw_wgeom_k_sentinel_scalar"
        )
        sentinel_neg_sq = create_literal_value(
            torch.tensor([-1000.0]), name="tw_wgeom_k_sentinel_neg_sq"
        )
        k_idx = select(
            is_wall, kv.wall_index_at_wall, sentinel_scalar, approximate=False
        )
        k_negsq = select(
            is_wall,
            kv.wall_index_neg_sq_at_wall,
            sentinel_neg_sq,
            approximate=False,
        )
        k = Concatenate([k_idx, k_negsq])

        wall_geom_value = Concatenate(
            [
                kv.wall_ax,
                kv.wall_ay,
                kv.wall_bx,
                kv.wall_by,
                kv.wall_bsp_coeffs,
                kv.wall_bsp_const,
            ]
        )
        wall_geom = attend_argmax_dot(
            query_vector=q_gated,
            key_vector=k,
            value=cond_gate(is_wall, wall_geom_value),
            match_gain=20.0,
            assert_hardness_gt=0.99,
        )
        sel_ax = extract_from(wall_geom, 4 + max_bsp_nodes + 1, 0, 1, "tw_sel_ax")
        sel_ay = extract_from(wall_geom, 4 + max_bsp_nodes + 1, 1, 1, "tw_sel_ay")
        sel_bx = extract_from(wall_geom, 4 + max_bsp_nodes + 1, 2, 1, "tw_sel_bx")
        sel_by = extract_from(wall_geom, 4 + max_bsp_nodes + 1, 3, 1, "tw_sel_by")
        sel_bsp_coeffs = extract_from(
            wall_geom, 4 + max_bsp_nodes + 1, 4, max_bsp_nodes, "tw_sel_bsp_coeffs"
        )
        sel_bsp_const = extract_from(
            wall_geom,
            4 + max_bsp_nodes + 1,
            4 + max_bsp_nodes,
            1,
            "tw_sel_bsp_const",
        )

    # ---------------------------------------------------------------------
    # Base-value computations.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/hit_compute"):
        hit_full, hit_x, hit_y = _compute_hit_flags(
            sel_ax,
            sel_ay,
            sel_bx,
            sel_by,
            kv.player_x,
            kv.player_y,
            kv.vel_dx,
            kv.vel_dy,
        )

    with annotate("thinking_wall/rotate_endpoints"):
        dax = subtract(sel_ax, kv.player_x)
        day = subtract(sel_ay, kv.player_y)
        dbx = subtract(sel_bx, kv.player_x)
        dby = subtract(sel_by, kv.player_y)
        cross_a, dot_a = _rotate_into_player_frame(
            kv.move_cos, kv.move_sin, dax, day, "va"
        )
        cross_b, dot_b = _rotate_into_player_frame(
            kv.move_cos, kv.move_sin, dbx, dby, "vb"
        )
        # Clamp into the wire-format CROSS/DOT range so the
        # downstream ``quantize_to_range`` has a tight input
        # value_type.  The natural range from ``piecewise_linear_2d``
        # subtract is ±80 (twice the grid max), which would make
        # ``quantize_to_range(..., -40, 40)`` declare a ~16× inflated
        # output and fire the affine-bounds soundness test.
        cross_dot_max = 40.0
        cross_a = clamp(cross_a, -cross_dot_max, cross_dot_max)
        dot_a = clamp(dot_a, -cross_dot_max, cross_dot_max)
        cross_b = clamp(cross_b, -cross_dot_max, cross_dot_max)
        dot_b = clamp(dot_b, -cross_dot_max, cross_dot_max)

    with annotate("thinking_wall/central_ray"):
        sort_den, sort_num_t = _compute_central_ray_intersection(
            sel_ax,
            sel_ay,
            sel_bx,
            sel_by,
            kv.player_x,
            kv.player_y,
            kv.move_cos,
            kv.move_sin,
        )

    with annotate("thinking_wall/bsp_rank"):
        bsp_rank, is_renderable = _compute_bsp_and_renderable(
            sel_bsp_coeffs,
            sel_bsp_const,
            kv.side_P_vec,
            sort_den,
            sort_num_t,
            max_bsp_nodes,
        )

    # ---------------------------------------------------------------------
    # Derived-value computations.  Build the readback handle first so
    # ``get_value_after_last(...)`` calls route through a shared
    # attention head per identifier name.
    # ---------------------------------------------------------------------
    readback = build_thinking_readback(
        embedding=embedding,
        prev_id_slots=prev_slot_onehot,
        is_value_category=is_thinking_value,
        pos_encoding=pos_encoding,
    )

    # ---------------------------------------------------------------------
    # RESOLVED computation: the running-OR HIT_* thinking tokens from
    # Part 1 carry the global any-hit aggregate by wall 7, so at
    # RESOLVED_X/Y/ANGLE positions ``get_value_after_last`` returns it
    # as a scalar in [0, 1].  The cross-position attend_mean_where
    # aggregation from Phase A Part 4 is gone; the sliding math is
    # unchanged.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/resolved_compute"):
        resolved_x, resolved_y, resolved_angle = _compute_resolved(
            readback=readback,
            player_x=kv.player_x,
            player_y=kv.player_y,
            vel_dx=kv.vel_dx,
            vel_dy=kv.vel_dy,
            new_angle=kv.new_angle,
        )

    with annotate("thinking_wall/bsp_rank_bound"):
        # BSP_RANK is an integer 0..7 per design doc.  The raw
        # ``assert_integer`` call in ``_compute_bsp_and_renderable``
        # preserves the natural value_type of the dot-product
        # accumulation (``max_bsp_nodes × max_coord``), which is too
        # wide for the downstream quantize affine (scale = 65535/7
        # amplifies).  Clamping to [0, 7] enforces a tight value_type
        # without adding runtime error on well-formed inputs.
        bsp_rank = clamp(bsp_rank, 0.0, 7.0)

    # ---------------------------------------------------------------------
    # Phase B Part 2: expose BSP_RANK scalars as a KV side-channel so the
    # SORTED stage's quadratic-equality attention at SORT_RESULT id
    # positions can match bsp_rank against wall_counter.  At every
    # position we emit ``[bsp_rank_scalar, bsp_rank_neg_sq]``:
    #
    #   * BSP_RANK id position of renderable wall: ``[r, -r²]``.  The
    #     quadratic-attention query ``[2N, 1]`` dots to
    #     ``2N·r - r²`` = ``-(r-N)² + N²``, peaking at ``r = N``.
    #   * Non-renderable (at BSP_RANK id) or non-BSP_RANK position: the
    #     sentinels ``[-100, -1000]``.  Query dot ≤ ``-200N - 1000``,
    #     which after match_gain ~20 contributes effectively-zero
    #     softmax weight.
    #
    # Sentinels are picked small enough that ``select`` with
    # ``approximate=False`` (float-exact on the winning branch) adds only
    # one extra sublayer per select, yet large enough that the
    # softmax margin swamps FP drift.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/bsp_rank_for_sort"):
        _BSP_SCALAR_SENTINEL = -100.0
        _BSP_NEG_SQ_SENTINEL = -1000.0

        # Integer square over bsp_rank ∈ {0..7}.  One piecewise-linear
        # sublayer; evaluation at integer breakpoints is float-exact.
        bsp_rank_sq = piecewise_linear(
            bsp_rank,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            lambda x: x * x,
            name="bsp_rank_sq",
        )
        bsp_rank_neg_sq_value = negate(bsp_rank_sq)

        # ``is_usable`` = this position is BSP_RANK id AND the attended
        # wall is renderable.  Non-usable positions emit the sentinels.
        is_usable = bool_all_true(
            [is_identifier_by_slot[_SLOT_BSP_RANK], is_renderable]
        )

        sentinel_scalar_lit = create_literal_value(
            torch.tensor([_BSP_SCALAR_SENTINEL]),
            name="tw_bsp_scalar_sentinel",
        )
        sentinel_neg_sq_lit = create_literal_value(
            torch.tensor([_BSP_NEG_SQ_SENTINEL]),
            name="tw_bsp_neg_sq_sentinel",
        )

        # approximate=False: the on-path is float-exact, so bsp_rank
        # (integer 0..7) and -r² (in [-49, 0]) pass through without
        # select's M·c_tol noise contaminating the attention dot.
        bsp_rank_scalar_for_sort = select(
            is_usable,
            bsp_rank,
            sentinel_scalar_lit,
            approximate=False,
        )
        bsp_rank_neg_sq_for_sort = select(
            is_usable,
            bsp_rank_neg_sq_value,
            sentinel_neg_sq_lit,
            approximate=False,
        )

    # ---------------------------------------------------------------------
    # Phase B Part 2: expose ``current_wall_index`` as a 1-hot at
    # thinking identifier and thinking VALUE positions.  Two gated
    # variants so downstream stages can key their content attentions
    # on identifier positions (e.g. the quadratic SORTED attention
    # reading BSP_RANK id V) or on VALUE positions (e.g. the VIS_LO /
    # VIS_HI content attentions reading thinking VALUE payloads).
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/wall_index_onehot_exports"):
        # identifier_wall_index_onehot is still consumed by SORTED's
        # quad_attention V (it returns the picked wall_index as an
        # 8-wide one-hot, then a Linear with arange weights converts
        # back to scalar).  Phase C Part 3 migrated the *value*-side
        # one-hot consumers to the new value_wall_index_scalar /
        # value_wall_index_neg_sq channels exposed below.
        identifier_wall_index_onehot = cond_gate(is_any_identifier, wall_j_onehot)

    # Phase C Part 3: quadratic-equality exports for content attentions
    # keyed by ``(name-VALUE, wall_index)``.  Sentinel pattern mirrors
    # ``bsp_rank_*_for_sort`` above: real values gated to thinking-VALUE
    # positions, large-negative sentinels elsewhere keep softmax mass off
    # non-VALUE keys.  ``approximate=False`` on the select keeps the
    # on-path float-exact for integer wall_index round-trip.
    with annotate("thinking_wall/wall_index_quad_exports"):
        sentinel_value_scalar = create_literal_value(
            torch.tensor([-100.0]), name="tw_value_widx_sentinel_scalar"
        )
        sentinel_value_neg_sq = create_literal_value(
            torch.tensor([-1000.0]), name="tw_value_widx_sentinel_neg_sq"
        )
        value_wall_index_scalar = select(
            is_thinking_value,
            wi_clamped,
            sentinel_value_scalar,
            approximate=False,
        )
        value_wall_index_neg_sq = select(
            is_thinking_value,
            current_wall_index_neg_sq,
            sentinel_value_neg_sq,
            approximate=False,
        )

    # ---------------------------------------------------------------------
    # Phase B Part 1 slot-split: the deep T/VIS chain gets carved into
    # four new intermediate slots (T_STAR_L/R for plane-crossings, COL_A/B
    # for atan-projected endpoint columns).  Each downstream slot reads
    # its upstream intermediates from the KV cache via ``get_value_after_last``
    # instead of recomputing them in-step.
    #
    # Shared FOV-edge evaluations: both T_STAR and T_LO/T_HI/COL paths
    # need ``f_L_a, f_L_b, f_R_a, f_R_b``.  Compute once and share (the
    # compiler will materialize them once per position).  CROSS_A/DOT_A/
    # CROSS_B/DOT_B come from the prior identifier VALUE positions in
    # the same wall; readback's per-name attention cache dedupes.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/fov_edges"):
        ca_kv = readback.get_value_after_last("CROSS_A")
        da_kv = readback.get_value_after_last("DOT_A")
        cb_kv = readback.get_value_after_last("CROSS_B")
        db_kv = readback.get_value_after_last("DOT_B")

        W = config.screen_width
        fov = config.fov_columns
        fov_rad = float(fov) * math.pi / 128.0
        half_fov_rad = fov_rad / 2.0
        sin_hf = math.sin(half_fov_rad)
        cos_hf = math.cos(half_fov_rad)

        # f_*(p) = sin(½·fov)·dot(p) ∓ cos(½·fov)·cross(p).  Inside cone
        # iff both f_L ≥ 0 and f_R ≥ 0.
        f_L_a = add_scaled_nodes(sin_hf, da_kv, -cos_hf, ca_kv)
        f_L_b = add_scaled_nodes(sin_hf, db_kv, -cos_hf, cb_kv)
        f_R_a = add_scaled_nodes(sin_hf, da_kv, cos_hf, ca_kv)
        f_R_b = add_scaled_nodes(sin_hf, db_kv, cos_hf, cb_kv)

        cross_dot_max = 40.0  # wire-format readback range, ±CROSS/DOT max
        max_f_mag = (sin_hf + cos_hf) * cross_dot_max
        max_denom = 2.0 * max_f_mag

    with annotate("thinking_wall/t_star"):
        # Raw plane-crossing parameter per plane.  ``t* = f_a / (f_a − f_b)``
        # computed as ``reciprocal → multiply_2d`` with sign fix-up.  The
        # emitted value gets clamped to the wire-format [-2, 2] range so the
        # quantize affine's declared output stays tight; for well-formed
        # geometry the in-range t* is in (0, 1).
        t_star_L = _compute_t_star(
            f_L_a, f_L_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="L"
        )
        t_star_R = _compute_t_star(
            f_R_a, f_R_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="R"
        )
        t_star_L = clamp(t_star_L, -2.0, 2.0)
        t_star_R = clamp(t_star_R, -2.0, 2.0)

    with annotate("thinking_wall/t_lo_hi"):
        # T_LO / T_HI become shallow consumers: read T_STAR_L/R from KV,
        # apply a_inside / b_inside select, aggregate with max(0, ...) /
        # min(1, ...).  ``a_inside`` is compare(f_L_a, 0) — a single op
        # on top of the shared f_L_a node.
        tsl_kv = readback.get_value_after_last("T_STAR_L")
        tsr_kv = readback.get_value_after_last("T_STAR_R")

        zero_lit = create_literal_value(torch.tensor([0.0]), name="tw_tlo_zero")
        one_lit = create_literal_value(torch.tensor([1.0]), name="tw_tlo_one")

        a_inside_L = compare(f_L_a, 0.0)
        a_inside_R = compare(f_R_a, 0.0)
        b_inside_L = compare(f_L_b, 0.0)
        b_inside_R = compare(f_R_b, 0.0)

        # Per-plane t_lo / t_hi contribs: if the relevant endpoint is inside
        # the plane, contribute the identity (0 for lo, 1 for hi); else the
        # KV-read t_star.
        t_lo_contrib_L = select(a_inside_L, zero_lit, tsl_kv, c_tol=_VIS_C_TOL)
        t_lo_contrib_R = select(a_inside_R, zero_lit, tsr_kv, c_tol=_VIS_C_TOL)
        t_hi_contrib_L = select(b_inside_L, one_lit, tsl_kv, c_tol=_VIS_C_TOL)
        t_hi_contrib_R = select(b_inside_R, one_lit, tsr_kv, c_tol=_VIS_C_TOL)

        t_lo = max_node(zero_lit, max_node(t_lo_contrib_L, t_lo_contrib_R))
        t_hi = min_node(one_lit, min_node(t_hi_contrib_L, t_hi_contrib_R))
        t_lo = clamp(t_lo, 0.0, 1.0)
        t_hi = clamp(t_hi, 0.0, 1.0)

    with annotate("thinking_wall/col"):
        # COL_A / COL_B: atan-projected interior column + FOV-boundary
        # column when the endpoint is outside the cone.  The boundary
        # side (col=0 vs col=W) is decided by which plane's clip wins —
        # derived from the same t_lo_contrib / t_hi_contrib nodes T_LO/T_HI
        # already computed (shared, single compiled materialisation).
        vis_lo_bound_hi = float(VALUE_RANGE_BY_NAME["VIS_LO"][1])
        W_lit = create_literal_value(torch.tensor([float(W)]), name="tw_col_W")

        col_A_interior = _endpoint_to_column(ca_kv, da_kv, W=W, fov=fov, suffix="tw_a")
        col_B_interior = _endpoint_to_column(cb_kv, db_kv, W=W, fov=fov, suffix="tw_b")

        a_clipped_on_L = compare(
            multiply_const(subtract(t_lo_contrib_L, t_lo_contrib_R), _T_COMPARE_SCALE),
            0.0,
        )
        b_clipped_on_L = compare(
            multiply_const(subtract(t_hi_contrib_R, t_hi_contrib_L), _T_COMPARE_SCALE),
            0.0,
        )
        col_A_boundary = select(a_clipped_on_L, W_lit, zero_lit, c_tol=_VIS_C_TOL)
        col_B_boundary = select(b_clipped_on_L, W_lit, zero_lit, c_tol=_VIS_C_TOL)
        col_A = select(
            a_inside_L,
            select(a_inside_R, col_A_interior, col_A_boundary, c_tol=_VIS_C_TOL),
            col_A_boundary,
            c_tol=_VIS_C_TOL,
        )
        col_B = select(
            b_inside_L,
            select(b_inside_R, col_B_interior, col_B_boundary, c_tol=_VIS_C_TOL),
            col_B_boundary,
            c_tol=_VIS_C_TOL,
        )
        col_A = clamp(col_A, -2.0, vis_lo_bound_hi)
        col_B = clamp(col_B, -2.0, vis_lo_bound_hi)

    with annotate("thinking_wall/vis"):
        # VIS_LO / VIS_HI become shallow: read COL_A/B and T_LO/T_HI from
        # KV; is_empty from comparing T_LO vs T_HI; apply sentinel + gate.
        ca_col_kv = readback.get_value_after_last("COL_A")
        cb_col_kv = readback.get_value_after_last("COL_B")
        tlo_kv = readback.get_value_after_last("T_LO")
        thi_kv = readback.get_value_after_last("T_HI")

        is_empty = compare(
            multiply_const(subtract(tlo_kv, thi_kv), _T_COMPARE_SCALE), 0.0
        )
        vis_lo_visible = min_node(ca_col_kv, cb_col_kv)
        vis_hi_visible = max_node(ca_col_kv, cb_col_kv)

        sentinel = create_literal_value(
            torch.tensor([float(W + 2)]), name="tw_vis_sentinel"
        )
        vis_lo_raw = select(is_empty, sentinel, vis_lo_visible, c_tol=_VIS_C_TOL)
        vis_hi_raw = select(is_empty, sentinel, vis_hi_visible, c_tol=_VIS_C_TOL)

        vis_lo = cond_gate(is_renderable, vis_lo_raw, c_tol=_VIS_C_TOL)
        vis_hi = cond_gate(is_renderable, vis_hi_raw, c_tol=_VIS_C_TOL)
        vis_lo = clamp(vis_lo, -2.0, vis_lo_bound_hi)
        vis_hi = clamp(vis_hi, -2.0, vis_lo_bound_hi)

    # ---------------------------------------------------------------------
    # Phase B Part 1: HIT_* running-OR accumulator.  Each wall's HIT_*
    # emits the OR of this wall's local flag with every prior wall's
    # HIT_* — by the time wall 7 fires, each of the three HIT_* values
    # is the global aggregate across all walls.  The previous value
    # comes from ``get_value_after_last("HIT_*")``; at wall 0 the cache
    # is empty, so we gate the readback to 0 using ``current_wall_index``.
    # OR over [0, 1] inputs is ``max`` (saturating).  Part 3 will wire
    # RESOLVED to consume the wall-7 global aggregate via a single
    # readback, replacing the current cross-position attend_mean_where
    # aggregation.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/hit_running_or"):
        # is_not_wall_zero: +1 at walls 1..7, −1 at wall 0.  ``current_wall_index``
        # is a clean integer in [0, max_walls-1] coming out of
        # ``find_current_wall``; compare at 0.5 gives a crisp ±1.
        is_not_wall_zero = compare(current_wall_index, 0.5)

        # Pass ``assert_hardness_gt=None`` so the runtime softmax-hardness
        # assert (used in ``debug=True`` forwards) is skipped — at wall 0
        # the cache is empty and the attention degrades to pure recency,
        # which may or may not concentrate above 0.99 at long context
        # lengths.  The cond_gate below zeroes the garbage out so
        # correctness doesn't depend on the empty-cache payload.
        hf_prev_raw = readback.get_value_after_last("HIT_FULL", assert_hardness_gt=None)
        hx_prev_raw = readback.get_value_after_last("HIT_X", assert_hardness_gt=None)
        hy_prev_raw = readback.get_value_after_last("HIT_Y", assert_hardness_gt=None)
        hf_prev = cond_gate(is_not_wall_zero, hf_prev_raw, c_tol=_VIS_C_TOL)
        hx_prev = cond_gate(is_not_wall_zero, hx_prev_raw, c_tol=_VIS_C_TOL)
        hy_prev = cond_gate(is_not_wall_zero, hy_prev_raw, c_tol=_VIS_C_TOL)

        # OR over {0, 1} inputs is ``max`` — and max of two ``[0, 1]``
        # values stays in ``[0, 1]``, so no clamp needed.  ``bool_to_01``
        # maps ±1 → 0/1 for the locally-computed flag.
        hit_full_or = max_node(bool_to_01(hit_full), hf_prev)
        hit_x_or = max_node(bool_to_01(hit_x), hx_prev)
        hit_y_or = max_node(bool_to_01(hit_y), hy_prev)
        hit_full_or = clamp(hit_full_or, 0.0, 1.0)
        hit_x_or = clamp(hit_x_or, 0.0, 1.0)
        hit_y_or = clamp(hit_y_or, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # Per-slot quantize → select-q → encode-once.
    #
    # Each identifier slot quantizes its computed float into a 1-wide
    # ``q ∈ [0, 65535]`` using the slot's design-doc (lo, hi) range.
    # We then sum-cond-gate to pick the active slot's q (exactly one is
    # +1 at any identifier position; sum collapses to the active one).
    # The triangle-wave cascade + 16 parallel compares in
    # :func:`encode_value_binary` runs **once** per position on the
    # selected q rather than 20 times across all slots.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/normalize_per_slot"):
        # Each per-wall value gets normalized to ``[0, 1]`` using its
        # design-doc (lo, hi) range.  Gating 1-wide ``[0, 1]`` signals
        # keeps ``cond_gate.M = 2``; if we gated on q ∈ ``[0, 65535]``
        # the per-gate ``M ≈ 131070`` would blow past
        # ``test_gate_M_bounded``'s 50 000 ratchet at every slot.  A
        # single ``multiply_const`` scales the selected normalized
        # value to ``[0, 65535]`` *after* the selection, where there
        # are no more gates to accumulate through.  Booleans pass
        # through ``bool_to_01`` first so the ±1 input lands as 0/1
        # (already in ``[0, 1]``; no normalization needed but a
        # uniform path keeps the helper simple).
        norm_by_slot: dict[int, Node] = {
            _SLOT_BSP_RANK: _norm(bsp_rank, "BSP_RANK"),
            _SLOT_IS_RENDERABLE: _norm(bool_to_01(is_renderable), "IS_RENDERABLE"),
            _SLOT_CROSS_A: _norm(cross_a, "CROSS_A"),
            _SLOT_DOT_A: _norm(dot_a, "DOT_A"),
            _SLOT_CROSS_B: _norm(cross_b, "CROSS_B"),
            _SLOT_DOT_B: _norm(dot_b, "DOT_B"),
            _SLOT_T_STAR_L: _norm(t_star_L, "T_STAR_L"),
            _SLOT_T_STAR_R: _norm(t_star_R, "T_STAR_R"),
            _SLOT_T_LO: _norm(t_lo, "T_LO"),
            _SLOT_T_HI: _norm(t_hi, "T_HI"),
            _SLOT_COL_A: _norm(col_A, "COL_A"),
            _SLOT_COL_B: _norm(col_B, "COL_B"),
            _SLOT_VIS_LO: _norm(vis_lo, "VIS_LO"),
            _SLOT_VIS_HI: _norm(vis_hi, "VIS_HI"),
            _SLOT_HIT_FULL: _norm(hit_full_or, "HIT_FULL"),
            _SLOT_HIT_X: _norm(hit_x_or, "HIT_X"),
            _SLOT_HIT_Y: _norm(hit_y_or, "HIT_Y"),
            _SLOT_RESOLVED_X: _norm(resolved_x, "RESOLVED_X"),
            _SLOT_RESOLVED_Y: _norm(resolved_y, "RESOLVED_Y"),
            _SLOT_RESOLVED_ANGLE: _norm(resolved_angle, "RESOLVED_ANGLE"),
        }

    with annotate("thinking_wall/select_and_factor"):
        # Select via sum-of-cond-gated-1-wide-normalized values.  At
        # most one slot's detector fires at any position, so exactly
        # one term is non-zero in the sum.  Each ``cond_gate`` has
        # ``M = 2`` (input range ``[0, 1]``) — small enough to pass
        # the ratchet.  ``sum_nodes`` is a pure Linear (no gate).
        gated_norm = [
            cond_gate(is_identifier_by_slot[slot], norm_by_slot[slot])
            for slot in sorted(norm_by_slot.keys())
        ]
        n_selected = sum_nodes(gated_norm)
        # Scale once from normalized [0, 1] back into q ∈ [0, 65535]
        # for the factor stage.  No more gates after this — the scale
        # is a plain Linear.
        q_selected = multiply_const(n_selected, float(DEFAULT_N_LEVELS - 1))
        emit = encode_value_binary(q_selected, suffix="_id_emit")
        # Gate to zero at non-identifier positions.  At non-firing
        # positions ``q_selected = 0`` and the factor produces
        # ``W_EMBED[VALUE_0]``; the gate cleanly suppresses that so
        # it doesn't leak into the marker or value paths.
        identifier_emit = cond_gate(is_any_identifier, emit)

    with annotate("thinking_wall/next_token_embedding"):
        next_token_embedding = _compute_next_token_embedding(
            is_thinking_wall_marker=is_thinking_wall_marker,
            is_thinking_value=is_thinking_value,
            prev_slot_onehot=prev_slot_onehot,
            identifier_contribution=identifier_emit,
            current_wall_index=current_wall_index,
            max_walls=max_walls,
        )

    # is_thinking_active: any of (marker, identifier, value).  Gates
    # the next-token-embedding select chain in _assemble_output.
    sum_active_01 = sum_nodes(
        [
            bool_to_01(is_thinking_wall_marker),
            bool_to_01(is_any_identifier),
            bool_to_01(is_thinking_value),
        ]
    )
    is_thinking_active = compare(sum_active_01, 0.5)

    return ThinkingWallOutput(
        next_token_embedding=next_token_embedding,
        is_thinking_active=is_thinking_active,
        readback=readback,
        bsp_rank_scalar_for_sort=bsp_rank_scalar_for_sort,
        bsp_rank_neg_sq_for_sort=bsp_rank_neg_sq_for_sort,
        identifier_wall_index_onehot=identifier_wall_index_onehot,
        value_wall_index_scalar=value_wall_index_scalar,
        value_wall_index_neg_sq=value_wall_index_neg_sq,
    )


# ---------------------------------------------------------------------------
# Base-value helpers
# ---------------------------------------------------------------------------


def _compute_hit_flags(
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    player_x: Node,
    player_y: Node,
    vel_dx: Node,
    vel_dy: Node,
):
    """Ray-segment hit flags for three rays — ported from ``wall.py``."""
    ex = subtract(wall_bx, wall_ax)
    ey = subtract(wall_by, wall_ay)
    dax = subtract(wall_ax, player_x)
    day = subtract(wall_ay, player_y)

    p_dx_ey = piecewise_linear_2d(
        vel_dx, ey, VEL_BP, DIFF_BP, lambda a, b: a * b, name="tw_dx_ey"
    )
    p_dy_ex = piecewise_linear_2d(
        vel_dy, ex, VEL_BP, DIFF_BP, lambda a, b: a * b, name="tw_dy_ex"
    )
    p_dax_ey = piecewise_linear_2d(
        dax, ey, DIFF_BP, DIFF_BP, lambda a, b: a * b, name="tw_dax_ey"
    )
    p_day_ex = piecewise_linear_2d(
        day, ex, DIFF_BP, DIFF_BP, lambda a, b: a * b, name="tw_day_ex"
    )
    p_dax_dy = piecewise_linear_2d(
        dax, vel_dy, DIFF_BP, VEL_BP, lambda a, b: a * b, name="tw_dax_dy"
    )
    p_day_dx = piecewise_linear_2d(
        day, vel_dx, DIFF_BP, VEL_BP, lambda a, b: a * b, name="tw_day_dx"
    )

    num_t = subtract(p_dax_ey, p_day_ex)

    den_full = subtract(p_dx_ey, p_dy_ex)
    num_u_full = subtract(p_dax_dy, p_day_dx)
    hit_full = _validity(den_full, num_t, num_u_full)

    den_x = p_dx_ey
    num_u_x = negate(p_day_dx)
    hit_x = _validity(den_x, num_t, num_u_x)

    den_y = negate(p_dy_ex)
    num_u_y = p_dax_dy
    hit_y = _validity(den_y, num_t, num_u_y)

    return hit_full, hit_x, hit_y


def _validity(den: Node, num_t: Node, num_u: Node) -> Node:
    """Same intersect-segment validity as wall._collision_validity."""
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

    return bool_all_true([is_den_ok, is_t_pos, is_t_le_den, is_u_ge_0, is_u_le_den])


def _rotate_into_player_frame(
    cos_p: Node,
    sin_p: Node,
    dx: Node,
    dy: Node,
    suffix: str,
):
    """``(cross, dot) = rotate((dx, dy), by -theta)`` — wall.py port."""
    cross = subtract(
        piecewise_linear_2d(
            cos_p, dy, TRIG_BP, DIFF_BP, lambda a, b: a * b, name=f"tw_cos_dy_{suffix}"
        ),
        piecewise_linear_2d(
            sin_p, dx, TRIG_BP, DIFF_BP, lambda a, b: a * b, name=f"tw_sin_dx_{suffix}"
        ),
    )
    dot = add(
        piecewise_linear_2d(
            cos_p, dx, TRIG_BP, DIFF_BP, lambda a, b: a * b, name=f"tw_cos_dx_{suffix}"
        ),
        piecewise_linear_2d(
            sin_p, dy, TRIG_BP, DIFF_BP, lambda a, b: a * b, name=f"tw_sin_dy_{suffix}"
        ),
    )
    return cross, dot


def _compute_central_ray_intersection(
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    player_x: Node,
    player_y: Node,
    move_cos: Node,
    move_sin: Node,
):
    """Intersect the player's central viewing ray with this wall's line.

    ``(sort_den, sort_num_t)`` port of ``wall._compute_central_ray_intersection``.
    """
    w_ex = subtract(wall_bx, wall_ax)
    w_ey = subtract(wall_by, wall_ay)
    w_fx = subtract(wall_ax, player_x)
    w_gy = subtract(player_y, wall_ay)

    sort_ey_cos = piecewise_linear_2d(
        w_ey, move_cos, DIFF_BP, TRIG_BP, lambda a, b: a * b, name="tw_sort_ey_cos"
    )
    sort_ex_sin = piecewise_linear_2d(
        w_ex, move_sin, DIFF_BP, TRIG_BP, lambda a, b: a * b, name="tw_sort_ex_sin"
    )
    sort_den = subtract(sort_ey_cos, sort_ex_sin)

    sort_ey_fx = piecewise_linear_2d(
        w_ey, w_fx, DIFF_BP, DIFF_BP, lambda a, b: a * b, name="tw_sort_ey_fx"
    )
    sort_ex_gy = piecewise_linear_2d(
        w_ex, w_gy, DIFF_BP, DIFF_BP, lambda a, b: a * b, name="tw_sort_ex_gy"
    )
    sort_num_t = add(sort_ey_fx, sort_ex_gy)
    return sort_den, sort_num_t


def _compute_bsp_and_renderable(
    wall_bsp_coeffs: Node,
    wall_bsp_const: Node,
    side_P_vec: Node,
    sort_den: Node,
    sort_num_t: Node,
    max_bsp_nodes: int,
):
    """Port of ``wall._compute_bsp_rank`` (minus the ``is_wall`` gate).

    Returns ``(bsp_rank, is_renderable)`` — integer 0..7 and ±1
    boolean.  The wall.py version ANDs with ``is_wall``; here we're at
    thinking-identifier positions (never WALL positions), so the
    ``is_wall`` check is dropped — renderability is purely a function
    of the (attended) wall geometry and the player pose.
    """
    bsp_products = []
    for i in range(max_bsp_nodes):
        c_i = extract_from(wall_bsp_coeffs, max_bsp_nodes, i, 1, f"tw_bsp_c_{i}")
        s_i = extract_from(side_P_vec, max_bsp_nodes, i, 1, f"tw_bsp_s_{i}")
        s_bool = compare(s_i, 0.5)
        bsp_products.append(cond_gate(s_bool, c_i))
    bsp_dot = sum_nodes(bsp_products)
    bsp_rank = assert_integer(add(bsp_dot, wall_bsp_const))

    abs_sort_den = abs_node(sort_den)
    is_den_ok = compare(abs_sort_den, 0.05)
    den_sign = compare(sort_den, 0.0)
    adj_num_t = select(den_sign, sort_num_t, negate(sort_num_t))
    is_t_pos = compare(adj_num_t, 0.0)
    is_renderable = bool_all_true([is_den_ok, is_t_pos])

    return bsp_rank, is_renderable


# ---------------------------------------------------------------------------
# Derived-value helpers (ported from wall._compute_visibility_columns).
# Phase B Part 1 splits the chain into four new intermediate slots
# (T_STAR_L/R, COL_A/B) so T_LO/T_HI/VIS_LO/VIS_HI become shallow
# KV-cache consumers.
# ---------------------------------------------------------------------------


def _compute_t_star(
    f_a: Node,
    f_b: Node,
    *,
    max_denom: float,
    max_f_mag: float,
    suffix: str,
) -> Node:
    """Raw plane-crossing parameter ``t* = f_a / (f_a − f_b)``.

    Computes ``reciprocal(|denom|) × f_a`` with a sign-fix select, the
    same decomposition used by the original ``_plane_clip_contribs`` —
    but without the ``a_inside`` / ``b_inside`` select applied on top.
    Callers (``T_LO`` / ``T_HI`` emit paths) apply that select
    themselves using ``f_L_a`` / ``f_R_a`` / ``f_L_b`` / ``f_R_b`` read
    from the KV cache alongside ``t_star``.
    """
    denom = subtract(f_a, f_b)
    denom_pos = compare(denom, 0.0)
    denom_abs = clamp(abs_node(denom), 0.1, max_denom)
    inv_denom_abs = reciprocal(denom_abs, min_value=0.1, max_value=max_denom, step=0.1)
    max_inv = 1.0 / 0.1

    t_star_pos = multiply_2d(
        f_a,
        inv_denom_abs,
        max_abs1=max_f_mag,
        max_abs2=max_inv,
        step1=0.5,
        step2=0.5,
        min2=0.0,
        name=f"tw_t_star_{suffix}",
    )
    t_star_neg = multiply_const(t_star_pos, -1.0)
    t_star = select(denom_pos, t_star_pos, t_star_neg, c_tol=_VIS_C_TOL)
    return t_star


def _endpoint_to_column(
    cross: Node,
    dot: Node,
    *,
    W: int,
    fov: int,
    suffix: str,
):
    """``(cross, dot) → screen column`` — ported from
    ``wall._endpoint_to_column``."""
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
    abs_dot = abs_node(dot)
    cross_clamped = clamp(cross, bp_cross_lo, bp_cross_hi)
    dot_pos = clamp(abs_dot, bp_dot_lo, bp_dot_hi)

    atan_val = low_rank_2d(
        cross_clamped,
        dot_pos,
        _COL_FOLD_BP_CROSS,
        _COL_FOLD_BP_DOT_ABS,
        _atan_of,
        rank=3,
        name=f"tw_atan_front_{suffix}",
    )
    col_front = Linear(
        atan_val,
        torch.tensor([[col_scale]]),
        torch.tensor([half_W]),
        name=f"tw_col_front_{suffix}",
    )
    col_back = Linear(
        col_front,
        torch.tensor([[-1.0]]),
        torch.tensor([float(W)]),
        name=f"tw_col_back_{suffix}",
    )
    col_final = select(dot_sign, col_front, col_back, c_tol=_VIS_C_TOL)
    return clamp(col_final, col_lo, col_hi)


# ---------------------------------------------------------------------------
# RESOLVED computation
# ---------------------------------------------------------------------------


def _compute_resolved(
    *,
    readback: ThinkingReadback,
    player_x: Node,
    player_y: Node,
    vel_dx: Node,
    vel_dy: Node,
    new_angle: Node,
):
    """Axis-separated wall sliding via running-OR HIT_* readbacks.

    Each HIT_FULL / HIT_X / HIT_Y thinking token emits the OR of its
    wall's local flag with every prior wall's (Phase B Part 1).  By
    wall 7 the emitted value is the global any-hit aggregate across all
    walls.  At RESOLVED_X/Y/ANGLE positions (which fire after wall 7's
    HIT_Y), ``get_value_after_last`` returns that global aggregate as a
    scalar in [0, 1] — 0 if no wall was hit on that ray, 1 otherwise.

    The resolved position on each axis uses the velocity component if
    *neither* the full ray nor that axis's lone ray hit anything;
    otherwise the player stays put on that axis.  ``new_angle`` passes
    through unchanged — collision doesn't rotate the player.
    """
    any_hit_full = compare(readback.get_value_after_last("HIT_FULL"), 0.5)
    any_hit_x = compare(readback.get_value_after_last("HIT_X"), 0.5)
    any_hit_y = compare(readback.get_value_after_last("HIT_Y"), 0.5)

    use_new_x = bool_any_true([negate(any_hit_full), negate(any_hit_x)])
    use_new_y = bool_any_true([negate(any_hit_full), negate(any_hit_y)])

    new_x = add(player_x, vel_dx)
    new_y = add(player_y, vel_dy)
    resolved_x = select(use_new_x, new_x, player_x)
    resolved_y = select(use_new_y, new_y, player_y)
    return resolved_x, resolved_y, new_angle


# ---------------------------------------------------------------------------
# Next-token embedding state machine
# ---------------------------------------------------------------------------


def _norm(value: Node, name: str) -> Node:
    """Normalize ``value`` to ``[0, 1]`` using the slot's ``(lo, hi)``.

    ``n = (value - lo) / (hi - lo)``.  Pure affine.  The caller pipes
    this through a ``cond_gate`` before scaling up to ``[0, 65535]``
    for the factor stage — gating in normalized space keeps
    ``cond_gate.M = 2`` instead of ``131070``.
    """
    lo, hi = VALUE_RANGE_BY_NAME[name]
    width = hi - lo
    shifted = add_const(value, -lo)
    return multiply_const(shifted, 1.0 / width)


def _compute_next_token_embedding(
    *,
    is_thinking_wall_marker: Node,
    is_thinking_value: Node,
    prev_slot_onehot: Node,
    identifier_contribution: Node,
    current_wall_index: Node,
    max_walls: int,
) -> Node:
    """Build the 72-wide next-token embedding at every position.

    Three mutually-exclusive contributions sum into the output:

    * marker step emits the BSP_RANK identifier embedding.
    * identifier step emits the pre-gated per-slot VALUE embedding
      (``identifier_contribution`` passed in — already zero at
      non-identifier positions).
    * value step emits the successor identifier / marker / RESOLVED_X
      / SORTED_WALL, selected via ``prev_slot_onehot``.

    Non-thinking positions zero out in all three contributions; the
    orchestrator further masks via ``is_thinking_active``.
    """
    e_bsp_rank = create_literal_value(embed_lookup("BSP_RANK"), name="e_bsp_rank")
    e_resolved_x = create_literal_value(embed_lookup("RESOLVED_X"), name="e_resolved_x")

    # ---- Marker step: emit BSP_RANK_ID. ----
    marker_contribution = cond_gate(is_thinking_wall_marker, e_bsp_rank)

    # ---- Value step: emit successor token. ----
    # Phase D Part 1: prev_slot_onehot is now ±1 (the V from the
    # readback attention is the slot one-hot column block in W_EMBED,
    # which carries +1 at the active slot and −1 elsewhere).  The
    # original Linear was designed for {0, 1} input
    # (output = ``next_after_slot[active_slot]``).  Substituting
    # ``y = (x + 1) / 2`` bakes the {0, 1} ↔ ±1 conversion into the
    # Linear's weights/bias — pure parameter shift, no extra ops, no
    # extra layers.
    next_after_slot = torch.zeros(len(IDENTIFIER_NAMES), D_EMBED)
    for slot, name in _VALUE_SUCCESSOR_BY_SLOT.items():
        next_after_slot[slot] = embed_lookup(name)
    static_next_at_value = Linear(
        prev_slot_onehot,
        next_after_slot / 2.0,
        next_after_slot.sum(dim=0) / 2.0,
        name="next_at_value_static",
    )

    # HIT_Y successor: next wall's THINKING_WALL marker, or RESOLVED_X
    # if we've just finished the last wall.
    wi_p1 = clamp(add_const(current_wall_index, 1.0), 0.0, float(max_walls - 1))
    wi_p2 = add_const(wi_p1, 1.0)
    wi_p1_onehot = bool_to_01(in_range(wi_p1, wi_p2, max_walls))
    marker_codes = torch.stack(
        [embed_lookup(f"THINKING_WALL_{i}") for i in range(max_walls)], dim=0
    )
    next_marker_embed = Linear(
        wi_p1_onehot,
        marker_codes,
        torch.zeros(D_EMBED),
        name="next_marker_embed",
    )
    is_last_wall = compare(current_wall_index, float(max_walls) - 1.5)
    hy_successor = select(is_last_wall, e_resolved_x, next_marker_embed)

    # Phase D Part 1: prev_slot_onehot is ±1, so the extract at the
    # HIT_Y slot is already the ±1 bool — no compare(0.5) needed.
    prev_was_hy_bool = extract_from(
        prev_slot_onehot, len(IDENTIFIER_NAMES), _SLOT_HIT_Y, 1, "prev_was_hy"
    )
    hy_contribution = cond_gate(prev_was_hy_bool, hy_successor)

    next_at_value_total = sum_nodes([static_next_at_value, hy_contribution])
    value_contribution = cond_gate(is_thinking_value, next_at_value_total)

    return sum_nodes([marker_contribution, identifier_contribution, value_contribution])
