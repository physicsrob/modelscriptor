"""THINKING_WALL stage: per-wall hit_full/hit_x/hit_y as autoregressive
tokens.

Phase A Part 1 — wire format unchanged from M4 (7 tokens per wall):

    [THINKING_WALL_N]
      [HIT_FULL]  v(hit_full)
      [HIT_X]     v(hit_x)
      [HIT_Y]     v(hit_y)

The three hit values are 0 or 1 ("did the player's velocity-ray hit
this wall along axis A?"), so quantization is exact.  After all
walls' hits emit, the last value step hands off to SORTED.

Representation shifted from M4's overlaid ``(next_token_type,
thinking_value)`` pair to a single ``next_token_id`` emitted as the
72-wide embedding of the next token.  The semantic shift is forced
by the one-ID-per-step model: at an identifier step the graph
produces the next position's full token content (including its
payload), so the hit-value computation runs at the identifier step
rather than the value step.  The value step's job reduces to
emitting the next identifier (or the next marker / SORTED hand-off
after HIT_Y).

State graph (thinking positions only):

* marker           → HIT_FULL    (emit embed_lookup("HIT_FULL"))
* HIT_FULL         → VALUE of hit_full (embed W_EMBED[hit_full_01])
* HIT_X            → VALUE of hit_x
* HIT_Y            → VALUE of hit_y
* VALUE, prev=HF   → HIT_X
* VALUE, prev=HX   → HIT_Y
* VALUE, prev=HY:
       if wall_idx < max_walls - 1 → THINKING_WALL[wall_idx + 1]
       else                        → SORTED_WALL

Three attention reads (same as M4):

1. **Current wall identity.**  ``attend_most_recent_matching`` against
   ``is_thinking_wall_marker``: value carries the marker's wall_index.
2. **Previous identifier type.**  ``attend_most_recent_matching``
   against ``is_any_identifier``: carries 3-wide one-hot
   ``[is_HF, is_HX, is_HY]``.
3. **Wall geometry from prompt.**  ``attend_argmax_dot`` using
   ``wall_j_onehot`` derived from ``current_wall_index``.

Hit math (``_compute_hit_flags`` / ``_validity``) is identical to
``stages.wall._compute_collision_flags`` — six piecewise products +
three five-predicate validity checks.
"""

from dataclasses import dataclass

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
from torchwright.ops.logic_ops import bool_all_true, cond_gate
from torchwright.ops.map_select import in_range, select

from torchwright.doom.embedding import D_EMBED, embed_lookup
from torchwright.doom.graph_constants import (
    DIFF_BP,
    VEL_BP,
)
from torchwright.doom.graph_utils import extract_from

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
    is_hit_full_id: Node,
    is_hit_x_id: Node,
    is_hit_y_id: Node,
    is_thinking_value: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
) -> ThinkingWallOutput:
    """Wire up the thinking-wall stage end to end."""
    assert max_walls <= 8, (
        "thinking_wall vocabulary defines 8 markers (THINKING_WALL_0..7); "
        f"max_walls={max_walls} would need additional vocabulary entries."
    )
    is_thinking_wall_n = is_thinking_wall_n[:max_walls]

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
        wi_clamped = clamp(current_wall_index, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wi_clamped, 1.0)
        wall_j_onehot = bool_to_01(in_range(wi_clamped, wi_p1, max_walls))

        # Restrict the attend query to identifier positions — that's
        # where the hit value gets computed and emitted as the next
        # token.  Other thinking positions (marker, value) don't need
        # the wall-geometry read.
        wall_geom = attend_argmax_dot(
            query_vector=cond_gate(is_any_identifier, wall_j_onehot),
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
            is_hit_full_id=is_hit_full_id,
            is_hit_x_id=is_hit_x_id,
            is_hit_y_id=is_hit_y_id,
            is_thinking_value=is_thinking_value,
            prev_was_hf_01=prev_was_hf_01,
            prev_was_hx_01=prev_was_hx_01,
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
# consumed at THINKING identifier positions).
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
    is_hit_full_id: Node,
    is_hit_x_id: Node,
    is_hit_y_id: Node,
    is_thinking_value: Node,
    prev_was_hf_01: Node,
    prev_was_hx_01: Node,
    hit_full: Node,
    hit_x: Node,
    hit_y: Node,
    current_wall_index: Node,
    max_walls: int,
) -> Node:
    """Emit the 72-wide embedding of the next token.

    See module docstring for the state graph.  Non-thinking positions
    get a don't-care zero output that the orchestrator gates away via
    ``is_thinking_active``.
    """
    e_hf = create_literal_value(embed_lookup("HIT_FULL"), name="e_hf")
    e_hx = create_literal_value(embed_lookup("HIT_X"), name="e_hx")
    e_hy = create_literal_value(embed_lookup("HIT_Y"), name="e_hy")
    e_sorted = create_literal_value(embed_lookup("SORTED_WALL"), name="e_sorted")
    e_value_0 = create_literal_value(embed_lookup("VALUE_0"), name="e_value_0")
    e_value_1 = create_literal_value(embed_lookup("VALUE_1"), name="e_value_1")
    zero_embedding = create_literal_value(torch.zeros(D_EMBED), name="e_zero")

    # Hit-value emission at each HIT_*_ID step.  VALUE IDs 0 and 1
    # have distinct W_EMBED rows that ``select`` cleanly picks between
    # via the ±1 hit-flag predicate.
    e_hit_full_value = select(hit_full, e_value_1, e_value_0)
    e_hit_x_value = select(hit_x, e_value_1, e_value_0)
    e_hit_y_value = select(hit_y, e_value_1, e_value_0)

    # At a VALUE step, emit the next identifier / marker / SORTED.
    # next_marker_embed: one-hot(wall_index + 1) projected through the
    # stacked per-wall marker embeddings.
    wi_p1 = clamp(add_const(current_wall_index, 1.0), 0.0, float(max_walls - 1))
    wi_p2 = add_const(wi_p1, 1.0)
    wi_p1_onehot = bool_to_01(in_range(wi_p1, wi_p2, max_walls))
    marker_codes = torch.stack(
        [embed_lookup(f"THINKING_WALL_{i}") for i in range(max_walls)], dim=0
    )  # (max_walls, D_EMBED)
    next_marker_embed = Linear(
        wi_p1_onehot,
        marker_codes,
        torch.zeros(D_EMBED),
        name="next_marker_embed",
    )

    is_last_wall = compare(current_wall_index, float(max_walls) - 1.5)
    next_after_hy = select(is_last_wall, e_sorted, next_marker_embed)

    prev_was_hf = compare(prev_was_hf_01, 0.5)
    prev_was_hx = compare(prev_was_hx_01, 0.5)
    next_at_value = select(
        prev_was_hf,
        e_hx,
        select(prev_was_hx, e_hy, next_after_hy),
    )

    return select(
        is_thinking_wall_marker,
        e_hf,
        select(
            is_hit_full_id,
            e_hit_full_value,
            select(
                is_hit_x_id,
                e_hit_x_value,
                select(
                    is_hit_y_id,
                    e_hit_y_value,
                    select(is_thinking_value, next_at_value, zero_embedding),
                ),
            ),
        ),
    )
