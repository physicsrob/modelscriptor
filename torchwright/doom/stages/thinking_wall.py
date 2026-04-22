"""THINKING_WALL stage: per-wall hit_full/hit_x/hit_y as autoregressive
tokens.

Phase A M4 — first migration of WALL-stage outputs to thinking tokens.
After PLAYER tokens, the autoregressive loop emits, for each wall:

    [THINKING_WALL_N]  (E8 code carrying wall_index N)
      [HIT_FULL_ID] v(hit_full)
      [HIT_X_ID]    v(hit_x)
      [HIT_Y_ID]    v(hit_y)

All three values are 0 or 1 ("did the player's velocity-ray hit this
wall along axis A?"), so quantization is exact.  After all 8 walls'
hits are emitted, the last value step emits ``next_token_type =
E8_SORTED_WALL`` to hand off to the existing sort+render loop.

The state machine lives entirely inside the graph:

* **Marker step** emits ``next_token_type = E8_HIT_FULL_ID`` (always —
  every wall starts with hit_full).
* **Identifier step** (HIT_FULL_ID / HIT_X_ID / HIT_Y_ID) emits
  ``next_token_type = E8_THINKING_VALUE``.
* **Value step** (THINKING_VALUE) finds the most-recent identifier in
  the KV cache to know (a) which hit value to compute and (b) what
  identifier or marker to emit next.

Three attention reads at value positions:

1. **Current wall identity.**  ``attend_most_recent_matching`` against
   ``is_thinking_wall_marker``: the value carries the marker's
   wall_index (1-wide).
2. **Previous identifier type.**  ``attend_most_recent_matching``
   against ``is_any_identifier``: the value carries a 3-wide
   one-hot ``[is_HF_id, is_HX_id, is_HY_id]``.
3. **Wall geometry from prompt.**  ``attend_argmax_dot`` with
   ``query = wall_j_onehot`` and ``key = wall_position_onehot``:
   reads ``(ax, ay, bx, by)`` from the matching WALL token.

Player position and velocity come from the existing PLAYER and INPUT
broadcasts respectively.

The hit-flag math is the same as ``stages.wall._compute_collision_flags``
(six shared piecewise products → three validity checks).
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    abs,
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
from torchwright.ops.logic_ops import bool_all_true, cond_gate
from torchwright.ops.map_select import in_range, select

from torchwright.doom.graph_constants import (
    DIFF_BP,
    E8_HIT_FULL_ID,
    E8_HIT_X_ID,
    E8_HIT_Y_ID,
    E8_SORTED_WALL,
    E8_THINKING_VALUE,
    E8_THINKING_WALL,
    VEL_BP,
)
from torchwright.doom.graph_utils import extract_from

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class ThinkingWallKVInput:
    """Cross-position values the value step needs.

    Wall geometry, position one-hot, INPUT velocities, PLAYER positions
    — all already broadcast or readable from the prompt by other stages.
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
    """Overlaid outputs the host copies forward.

    ``next_token_type`` drives the autoregressive sequence:
    marker→HIT_FULL_ID→VALUE→HIT_X_ID→VALUE→HIT_Y_ID→VALUE→
    (next marker or SORTED).

    ``thinking_value`` is the quantized integer the host re-feeds.
    Only meaningful at THINKING_VALUE positions; zero elsewhere.
    """

    next_token_type: Node  # 8-wide E8 code
    thinking_value: Node  # 1-wide
    is_thinking_active: Node  # 1-wide ±1; +1 at any thinking position


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
    is_hit_full_id: Node,
    is_hit_x_id: Node,
    is_hit_y_id: Node,
    is_thinking_value: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
) -> ThinkingWallOutput:
    """Wire up the thinking-wall stage end to end."""
    assert max_walls <= 8, (
        "thinking_wall vocabulary defines 8 markers (E8_THINKING_WALL[0..7]); "
        f"max_walls={max_walls} would need additional vocabulary entries in "
        "graph_constants.py."
    )
    # Use the first max_walls marker detectors / codes; the rest of the
    # 8-entry vocabulary is silently ignored.
    is_thinking_wall_n = is_thinking_wall_n[:max_walls]

    with annotate("thinking_wall/marker_index"):
        # At marker positions, decode wall_index from which of the 8
        # E8_THINKING_WALL[i] codes matched.  Off-marker positions get 0
        # (don't-care: we only use this value at marker positions, gated
        # via attention validity below).
        wall_index_at_marker = sum_nodes(
            [
                multiply_const(bool_to_01(is_thinking_wall_n[i]), float(i))
                for i in range(max_walls)
            ]
        )

    with annotate("thinking_wall/find_current_wall"):
        # At value (and identifier) positions, find the most recent
        # THINKING_WALL marker and read its wall_index.  Match-key is
        # the validity flag itself (±1) — a 1-wide content match —
        # combined with attend_most_recent_matching's PE-counter
        # recency tiebreak.
        query_const_1 = create_literal_value(
            torch.tensor([1.0]), name="tw_query_const_1"
        )
        # Gate the value: at non-marker positions the wall_index_at_marker
        # is 0 (not a real wall_index), and we want the attention to read
        # 0 there too — the recency-tiebreak math itself ignores them
        # (they don't outscore matches), but cond_gate makes the
        # contribution to a soft blend exactly 0 if any blending occurs.
        gated_marker_wall_index = cond_gate(
            is_thinking_wall_marker, wall_index_at_marker
        )
        # match_gain sized for max_n_pos ≈ 2048 with ±1 match swing of 2:
        # match_gain · 2 > _QUERY_GAIN · max_n_pos = 8 · 2048 ≈ 16400
        # → match_gain ≥ 8200.  Use 12000 for headroom.
        current_wall_index = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_const_1,
            key_vector=is_thinking_wall_marker,
            value=gated_marker_wall_index,
            match_gain=12000.0,
        )

    with annotate("thinking_wall/find_prev_identifier"):
        # 3-wide one-hot at identifier positions, 0 elsewhere.  At value
        # positions the most-recent-matching attention reads the most
        # recent identifier's one-hot back out.
        prev_id_value_at_id = Concatenate(
            [
                cond_gate(is_hit_full_id, query_const_1),
                cond_gate(is_hit_x_id, query_const_1),
                cond_gate(is_hit_y_id, query_const_1),
            ]
        )
        prev_id_value_gated = cond_gate(is_any_identifier, prev_id_value_at_id)
        prev_id_onehot = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_const_1,
            key_vector=is_any_identifier,
            value=prev_id_value_gated,
            match_gain=12000.0,
        )
        prev_was_hf_01 = extract_from(prev_id_onehot, 3, 0, 1, "prev_was_hf")
        prev_was_hx_01 = extract_from(prev_id_onehot, 3, 1, 1, "prev_was_hx")
        prev_was_hy_01 = extract_from(prev_id_onehot, 3, 2, 1, "prev_was_hy")

    with annotate("thinking_wall/wall_geom_attention"):
        # current_wall_index is in [0, 7] at value positions (after the
        # marker attention) — clamp defensively in case the soft attention
        # noise pushes it slightly outside.  Build a wall_j_onehot
        # (max_walls-wide {0,1}) and content-match against WALL prompt
        # positions to read (ax, ay, bx, by).
        wi_clamped = clamp(current_wall_index, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wi_clamped, 1.0)
        wall_j_onehot = bool_to_01(in_range(wi_clamped, wi_p1, max_walls))

        wall_geom = attend_argmax_dot(
            query_vector=cond_gate(is_thinking_value, wall_j_onehot),
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

    with annotate("thinking_wall/value_select"):
        # Convert ±1 hit booleans to {0, 1} (the integer the host reads)
        # and select the right one based on which identifier preceded
        # this VALUE token.
        hit_full_01 = bool_to_01(hit_full)
        hit_x_01 = bool_to_01(hit_x)
        hit_y_01 = bool_to_01(hit_y)

        # prev_was_*_01 is in {0, 1} (from cond_gate of ±1 onto a literal
        # 1.0).  sum-with-multiply computes a weighted select:
        # result = prev_HF·hit_full + prev_HX·hit_x + prev_HY·hit_y
        # Exactly one of the three is 1; the others are 0.
        selected_hit = sum_nodes(
            [
                cond_gate(compare(prev_was_hf_01, 0.5), hit_full_01),
                cond_gate(compare(prev_was_hx_01, 0.5), hit_x_01),
                cond_gate(compare(prev_was_hy_01, 0.5), hit_y_01),
            ]
        )

    with annotate("thinking_wall/next_token_type"):
        next_token_type = _compute_next_token_type(
            is_thinking_wall_marker=is_thinking_wall_marker,
            is_hit_full_id=is_hit_full_id,
            is_hit_x_id=is_hit_x_id,
            is_hit_y_id=is_hit_y_id,
            is_thinking_value=is_thinking_value,
            prev_was_hf_01=prev_was_hf_01,
            prev_was_hx_01=prev_was_hx_01,
            prev_was_hy_01=prev_was_hy_01,
            current_wall_index=current_wall_index,
            max_walls=max_walls,
        )

    # is_thinking_active: any of (marker, identifier, value).
    # Used by _assemble_output to gate the next_token_type select chain.
    sum_active_01 = sum_nodes(
        [
            bool_to_01(is_thinking_wall_marker),
            bool_to_01(is_any_identifier),
            bool_to_01(is_thinking_value),
        ]
    )
    is_thinking_active = compare(sum_active_01, 0.5)

    return ThinkingWallOutput(
        next_token_type=next_token_type,
        thinking_value=selected_hit,
        is_thinking_active=is_thinking_active,
    )


# ---------------------------------------------------------------------------
# Hit-flag computation (same math as wall.py:_compute_collision_flags,
# but using attended-in wall geometry rather than host-fed at-position
# fields, and without the is_wall gate — these values are only emitted
# at THINKING_VALUE positions).
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
    """Compute hit_full, hit_x, hit_y for the current wall + player ray.

    Mirrors ``wall._compute_collision_flags`` line for line, except:
    * geometry comes from the ``attend_argmax_dot`` read (not host-fed
      at-position),
    * no ``is_wall`` outer gate (the value step's own ``is_thinking_value``
      gate handles that at the output of the stage).
    """
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
# Next-token-type state machine
# ---------------------------------------------------------------------------


def _compute_next_token_type(
    *,
    is_thinking_wall_marker: Node,
    is_hit_full_id: Node,
    is_hit_x_id: Node,
    is_hit_y_id: Node,
    is_thinking_value: Node,
    prev_was_hf_01: Node,
    prev_was_hx_01: Node,
    prev_was_hy_01: Node,
    current_wall_index: Node,
    max_walls: int,
) -> Node:
    """Emit the next token's E8 code based on current step + history.

    State graph (only thinking positions; non-thinking positions get a
    don't-care zero output that the orchestrator will overwrite via the
    is_thinking_active gate):

    * marker         → HIT_FULL_ID
    * HIT_FULL_ID    → THINKING_VALUE
    * HIT_X_ID       → THINKING_VALUE
    * HIT_Y_ID       → THINKING_VALUE
    * THINKING_VALUE, prev=HF → HIT_X_ID
    * THINKING_VALUE, prev=HX → HIT_Y_ID
    * THINKING_VALUE, prev=HY:
          if wall_index < max_walls-1 → THINKING_WALL[wall_index+1]
          else                        → SORTED_WALL
    """
    e8_hf = create_literal_value(E8_HIT_FULL_ID, name="ntype_hf")
    e8_hx = create_literal_value(E8_HIT_X_ID, name="ntype_hx")
    e8_hy = create_literal_value(E8_HIT_Y_ID, name="ntype_hy")
    e8_value = create_literal_value(E8_THINKING_VALUE, name="ntype_value")
    e8_sorted = create_literal_value(E8_SORTED_WALL, name="ntype_sorted")
    zero_8 = create_literal_value(torch.zeros(8), name="ntype_zero")

    # next_marker_e8 = THINKING_WALL[wall_index + 1] when wall_index < 7.
    # Build via a Linear: stack the 8 E8 codes as rows of a (max_walls, 8)
    # matrix and project a one-hot of (wall_index + 1) through it.
    wi_p1 = clamp(add_const(current_wall_index, 1.0), 0.0, float(max_walls - 1))
    wi_p2 = add_const(wi_p1, 1.0)
    wi_p1_onehot = bool_to_01(in_range(wi_p1, wi_p2, max_walls))
    marker_codes = torch.stack(
        [E8_THINKING_WALL[i] for i in range(max_walls)], dim=0
    )  # (max_walls, 8)
    next_marker_e8 = Linear(
        wi_p1_onehot,
        marker_codes,
        torch.zeros(8),
        name="next_marker_e8",
    )

    # is_last_wall: wall_index >= max_walls - 1 (with 0.5 margin for
    # soft-attention noise on current_wall_index).
    is_last_wall = compare(current_wall_index, float(max_walls) - 1.5)
    next_after_hy = select(is_last_wall, e8_sorted, next_marker_e8)

    # Value-step branch: which identifier preceded?
    prev_was_hf = compare(prev_was_hf_01, 0.5)
    prev_was_hx = compare(prev_was_hx_01, 0.5)
    # Default for the innermost arm is next_after_hy (assumes prev=HY when
    # neither HF nor HX matched — the third arm of an exclusive 3-way
    # one-hot).
    next_at_value = select(
        prev_was_hf,
        e8_hx,
        select(prev_was_hx, e8_hy, next_after_hy),
    )

    # Identifier-step branch: always emit VALUE.
    # Marker step: always emit HIT_FULL_ID.
    # Compose via select cascade.
    out = select(
        is_thinking_wall_marker,
        e8_hf,
        select(
            is_hit_full_id,
            e8_value,
            select(
                is_hit_x_id,
                e8_value,
                select(
                    is_hit_y_id,
                    e8_value,
                    select(is_thinking_value, next_at_value, zero_8),
                ),
            ),
        ),
    )
    return out
