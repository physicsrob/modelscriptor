"""THINKING_WALL stage: per-wall thinking-phase state machine with a
full 16-identifier cascade plus 3 RESOLVED identifiers at the frame
boundary.

Phase A Part 2 — wire format per wall (27 steps):

    [THINKING_WALL_N]
      [BSP_RANK]         v(0)          # stubbed, emits VALUE_0
      [IS_RENDERABLE]    v(0)          # stubbed
      [CROSS_A]          v(0)          # stubbed
      [DOT_A]            v(0)          # stubbed
      [CROSS_B]          v(0)          # stubbed
      [DOT_B]            v(0)          # stubbed
      [T_LO]             v(0)          # stubbed
      [T_HI]             v(0)          # stubbed
      [VIS_LO]           v(0)          # stubbed
      [VIS_HI]           v(0)          # stubbed
      [HIT_FULL]         v(hit_full)   # real
      [HIT_X]            v(hit_x)      # real
      [HIT_Y]            v(hit_y)      # real

After the last wall's HIT_Y value step (frame boundary):

    [RESOLVED_X]        v(0)           # stubbed (real math lands in Part 4)
    [RESOLVED_Y]        v(0)           # stubbed
    [RESOLVED_ANGLE]    v(0)           # stubbed
    [SORTED_WALL]                      # hand-off out of the thinking phase

Semantic contract: each autoregressive step takes a single ``token_id``
in and emits the next ``token_id`` as a 72-wide embedding the host
argmaxes against ``W_EMBED.T``.

Under the embedding-carrier architecture, an identifier step computes
the value and emits the VALUE-ID embedding encoding it; a value step
uses the most-recent-identifier lookup to decide which identifier /
marker / RESOLVED-chain / SORTED hand-off to emit next.

Stub identifier steps emit the fixed ``embed_lookup("VALUE_0")`` row.
HIT_FULL / HIT_X / HIT_Y identifier steps compute the ray-segment hit
flag (identical math to ``wall.py:_compute_collision_flags``) and emit
``embed_lookup("VALUE_0")`` or ``embed_lookup("VALUE_1")`` depending
on the flag.

Attention reads (three hops):

1. **Current wall identity.**  ``attend_most_recent_matching`` against
   ``is_thinking_wall_marker``: value carries the marker's wall_index
   as a 1-wide scalar.
2. **Previous identifier slot.**  ``attend_most_recent_matching``
   against ``is_any_identifier``: value is a 16-wide one-hot, slot
   ``i`` active iff the most recent identifier was
   ``IDENTIFIER_NAMES[i]``.
3. **Wall geometry from prompt.**  ``attend_argmax_dot`` using
   ``wall_j_onehot`` derived from ``current_wall_index``; restricted
   to HIT_FULL/X/Y identifier steps via a query gate so the read only
   fires where the hit math needs it.
"""

from dataclasses import dataclass
from typing import List

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_const,
    bool_to_01,
    clamp,
    compare,
    multiply_const,
    negate,
    piecewise_linear_2d,
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

from torchwright.doom.embedding import (
    D_EMBED,
    IDENTIFIER_NAMES,
    embed_lookup,
)
from torchwright.doom.graph_constants import (
    DIFF_BP,
    VEL_BP,
)
from torchwright.doom.graph_utils import extract_from

# ---------------------------------------------------------------------------
# Slot indices for the three HIT_* identifiers the state machine
# special-cases for real value computation.  Other slots are stubs whose
# identifier step emits VALUE_0 unconditionally.
# ---------------------------------------------------------------------------

_SLOT_HIT_FULL = IDENTIFIER_NAMES.index("HIT_FULL")
_SLOT_HIT_X = IDENTIFIER_NAMES.index("HIT_X")
_SLOT_HIT_Y = IDENTIFIER_NAMES.index("HIT_Y")

# Slot i → name of the next token to emit at the VALUE step whose most
# recent identifier was at slot i.  Slot ``_SLOT_HIT_Y`` is deliberately
# absent — its successor is wall-index-dependent (next marker or
# RESOLVED_X if last wall) and is handled separately.
_VALUE_SUCCESSOR_BY_SLOT = {
    IDENTIFIER_NAMES.index("BSP_RANK"): "IS_RENDERABLE",
    IDENTIFIER_NAMES.index("IS_RENDERABLE"): "CROSS_A",
    IDENTIFIER_NAMES.index("CROSS_A"): "DOT_A",
    IDENTIFIER_NAMES.index("DOT_A"): "CROSS_B",
    IDENTIFIER_NAMES.index("CROSS_B"): "DOT_B",
    IDENTIFIER_NAMES.index("DOT_B"): "T_LO",
    IDENTIFIER_NAMES.index("T_LO"): "T_HI",
    IDENTIFIER_NAMES.index("T_HI"): "VIS_LO",
    IDENTIFIER_NAMES.index("VIS_LO"): "VIS_HI",
    IDENTIFIER_NAMES.index("VIS_HI"): "HIT_FULL",
    IDENTIFIER_NAMES.index("HIT_FULL"): "HIT_X",
    IDENTIFIER_NAMES.index("HIT_X"): "HIT_Y",
    IDENTIFIER_NAMES.index("RESOLVED_X"): "RESOLVED_Y",
    IDENTIFIER_NAMES.index("RESOLVED_Y"): "RESOLVED_ANGLE",
    IDENTIFIER_NAMES.index("RESOLVED_ANGLE"): "SORTED_WALL",
}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class ThinkingWallKVInput:
    """Cross-position values the hit computation needs.

    Wall geometry, position one-hot, INPUT velocities, PLAYER
    positions — all already broadcast or readable from the prompt by
    other stages.
    """

    # From WALL prompt positions (read via attend_argmax_dot).
    wall_ax: Node
    wall_ay: Node
    wall_bx: Node
    wall_by: Node
    wall_position_onehot: Node

    # From INPUT broadcast.
    vel_dx: Node
    vel_dy: Node

    # From PLAYER broadcasts (pre-collision; collision needs the player's
    # *intended* movement origin, which is the pre-collision position).
    player_x: Node
    player_y: Node


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
    """

    next_token_embedding: Node  # 72-wide W_EMBED row
    is_thinking_active: Node  # 1-wide ±1


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_thinking_wall(
    kv: ThinkingWallKVInput,
    *,
    is_wall: Node,
    is_thinking_wall_marker: Node,
    is_thinking_wall_n: list,
    is_any_identifier: Node,
    is_identifier_by_slot: List[Node],
    is_thinking_value: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
) -> ThinkingWallOutput:
    """Wire up the thinking-wall stage end to end."""
    assert max_walls <= 8, (
        "thinking_wall vocabulary defines 8 markers (THINKING_WALL_0..7); "
        f"max_walls={max_walls} would need additional vocabulary entries."
    )
    assert len(is_identifier_by_slot) == len(IDENTIFIER_NAMES) == 16, (
        f"expected 16 per-slot identifier detectors, got "
        f"{len(is_identifier_by_slot)}"
    )
    is_thinking_wall_n = is_thinking_wall_n[:max_walls]

    is_hit_full_id = is_identifier_by_slot[_SLOT_HIT_FULL]
    is_hit_x_id = is_identifier_by_slot[_SLOT_HIT_X]
    is_hit_y_id = is_identifier_by_slot[_SLOT_HIT_Y]

    with annotate("thinking_wall/marker_index"):
        # At marker positions, decode wall_index from which of the 8
        # THINKING_WALL_i detectors matched.  Off-marker positions get
        # 0 (don't-care: only used at marker positions, validated via
        # attention below).
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

    with annotate("thinking_wall/find_prev_identifier"):
        # Store a 16-wide slot one-hot at every identifier position;
        # ``attend_most_recent_matching`` against ``is_any_identifier``
        # reads back "which identifier was most recent" as that same
        # one-hot.  At non-identifier positions the concatenation is
        # zero-valued (so even without the outer ``is_any_identifier``
        # gate the key-side would filter cleanly), but the explicit
        # gate keeps intent visible and bounds the value more tightly.
        slot_onehot_at_id = Concatenate(
            [
                cond_gate(is_identifier_by_slot[i], query_const_1)
                for i in range(len(IDENTIFIER_NAMES))
            ]
        )
        slot_onehot_gated = cond_gate(is_any_identifier, slot_onehot_at_id)
        prev_slot_onehot = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_const_1,
            key_vector=is_any_identifier,
            value=slot_onehot_gated,
            match_gain=12000.0,
            assert_hardness_gt=0.99,
        )

    with annotate("thinking_wall/wall_geom_attention"):
        wi_clamped = clamp(current_wall_index, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wi_clamped, 1.0)
        wall_j_onehot = bool_to_01(in_range(wi_clamped, wi_p1, max_walls))

        # Only HIT_FULL/X/Y identifier steps read wall geometry — stub
        # identifier steps emit VALUE_0 unconditionally and don't need
        # the geometry read.  Gating the query keeps the attention head
        # off the stub positions' work path.
        is_hit_any = bool_any_true([is_hit_full_id, is_hit_x_id, is_hit_y_id])
        wall_geom = attend_argmax_dot(
            query_vector=cond_gate(is_hit_any, wall_j_onehot),
            key_vector=cond_gate(is_wall, kv.wall_position_onehot),
            value=cond_gate(
                is_wall,
                Concatenate([kv.wall_ax, kv.wall_ay, kv.wall_bx, kv.wall_by]),
            ),
            match_gain=1000.0,
            assert_hardness_gt=0.99,
        )
        sel_ax = extract_from(wall_geom, 4, 0, 1, "tw_sel_ax")
        sel_ay = extract_from(wall_geom, 4, 1, 1, "tw_sel_ay")
        sel_bx = extract_from(wall_geom, 4, 2, 1, "tw_sel_bx")
        sel_by = extract_from(wall_geom, 4, 3, 1, "tw_sel_by")

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

    with annotate("thinking_wall/next_token_embedding"):
        next_token_embedding = _compute_next_token_embedding(
            is_thinking_wall_marker=is_thinking_wall_marker,
            is_any_identifier=is_any_identifier,
            is_thinking_value=is_thinking_value,
            is_hit_full_id=is_hit_full_id,
            is_hit_x_id=is_hit_x_id,
            is_hit_y_id=is_hit_y_id,
            prev_slot_onehot=prev_slot_onehot,
            hit_full=hit_full,
            hit_x=hit_x,
            hit_y=hit_y,
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
    )


# ---------------------------------------------------------------------------
# Hit-flag computation (same math as wall.py:_compute_collision_flags,
# but using attended-in wall geometry rather than host-fed at-position
# fields, and without the is_wall outer gate — these values are only
# consumed at HIT_*_ID thinking identifier positions).
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


# ---------------------------------------------------------------------------
# Next-token embedding state machine
# ---------------------------------------------------------------------------


def _compute_next_token_embedding(
    *,
    is_thinking_wall_marker: Node,
    is_any_identifier: Node,
    is_thinking_value: Node,
    is_hit_full_id: Node,
    is_hit_x_id: Node,
    is_hit_y_id: Node,
    prev_slot_onehot: Node,
    hit_full: Node,
    hit_x: Node,
    hit_y: Node,
    current_wall_index: Node,
    max_walls: int,
) -> Node:
    """Build the 72-wide next-token embedding at every position.

    Three mutually-exclusive contributions sum into the output:

    * marker step emits the BSP_RANK identifier embedding.
    * identifier step emits VALUE_0 (stubs) or VALUE_{hit} (HIT_*).
    * value step emits the successor identifier / marker / RESOLVED_X
      / SORTED_WALL, selected via ``prev_slot_onehot``.

    Non-thinking positions zero out in all three contributions; the
    orchestrator further masks via ``is_thinking_active``.
    """
    e_bsp_rank = create_literal_value(embed_lookup("BSP_RANK"), name="e_bsp_rank")
    e_value_0 = create_literal_value(embed_lookup("VALUE_0"), name="e_value_0")
    e_value_1 = create_literal_value(embed_lookup("VALUE_1"), name="e_value_1")
    e_resolved_x = create_literal_value(embed_lookup("RESOLVED_X"), name="e_resolved_x")

    # ---- Marker step: emit BSP_RANK_ID. ----
    marker_contribution = cond_gate(is_thinking_wall_marker, e_bsp_rank)

    # ---- Identifier step: emit VALUE_0 baseline; override with
    # (VALUE_1 - VALUE_0) at HIT_*_ID positions whose hit flag is +1.
    # The inner cond_gate zeros the delta when the flag is negative;
    # the outer cond_gate zeros the delta at non-HIT_* positions.  At
    # stub identifier positions, only the VALUE_0 baseline survives.
    delta_value_01 = subtract(e_value_1, e_value_0)
    hf_override = cond_gate(is_hit_full_id, cond_gate(hit_full, delta_value_01))
    hx_override = cond_gate(is_hit_x_id, cond_gate(hit_x, delta_value_01))
    hy_override = cond_gate(is_hit_y_id, cond_gate(hit_y, delta_value_01))
    base_at_id = cond_gate(is_any_identifier, e_value_0)
    identifier_contribution = sum_nodes(
        [base_at_id, hf_override, hx_override, hy_override]
    )

    # ---- Value step: emit successor token. ----
    # Fifteen of the 16 slot outcomes are static: a Linear indexes a
    # fixed (16, 72) matrix with ``prev_slot_onehot`` to look up the
    # successor embedding.  The HIT_Y row is zero in that matrix and
    # is filled in by a dedicated contribution below that branches on
    # wall_index (next marker vs RESOLVED_X).
    next_after_slot = torch.zeros(len(IDENTIFIER_NAMES), D_EMBED)
    for slot, name in _VALUE_SUCCESSOR_BY_SLOT.items():
        next_after_slot[slot] = embed_lookup(name)
    static_next_at_value = Linear(
        prev_slot_onehot,
        next_after_slot,
        torch.zeros(D_EMBED),
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

    # Gate the HIT_Y-specific successor by the slot-12 component of
    # ``prev_slot_onehot``.  ``prev_slot_onehot`` is 0/1 per slot, so
    # compare to 0.5 gives a clean ±1 bool for cond_gate.
    prev_was_hy_01 = extract_from(
        prev_slot_onehot, len(IDENTIFIER_NAMES), _SLOT_HIT_Y, 1, "prev_was_hy"
    )
    prev_was_hy_bool = compare(prev_was_hy_01, 0.5)
    hy_contribution = cond_gate(prev_was_hy_bool, hy_successor)

    next_at_value_total = sum_nodes([static_next_at_value, hy_contribution])
    value_contribution = cond_gate(is_thinking_value, next_at_value_total)

    return sum_nodes([marker_contribution, identifier_contribution, value_contribution])
