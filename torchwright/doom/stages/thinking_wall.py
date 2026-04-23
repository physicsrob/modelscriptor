"""THINKING_WALL stage: per-wall thinking-phase state machine with a
full 16-identifier cascade plus 3 RESOLVED identifiers at the frame
boundary.

Phase A Part 3 — every per-wall identifier emits a real computed value.
Wire format per wall (27 steps, unchanged from Part 2):

    [THINKING_WALL_N]
      [BSP_RANK]         v(bsp_rank)      # integer 0..7
      [IS_RENDERABLE]    v(is_renderable) # 0 / 1
      [CROSS_A]          v(cross_a)       # signed offset, [-40, 40]
      [DOT_A]            v(dot_a)         # forward projection, [-40, 40]
      [CROSS_B]          v(cross_b)
      [DOT_B]            v(dot_b)
      [T_LO]             v(t_lo)          # clip parameter, [0, 1]
      [T_HI]             v(t_hi)
      [VIS_LO]           v(vis_lo)        # screen col, [-2, 122]
      [VIS_HI]           v(vis_hi)
      [HIT_FULL]         v(hit_full)      # 0 / 1
      [HIT_X]            v(hit_x)         # 0 / 1
      [HIT_Y]            v(hit_y)         # 0 / 1

RESOLVED chain (last wall's HIT_Y value → frame boundary) still stubs
as ``VALUE_0`` — real math lands in Part 4.

Two classes of identifier computation:

* **Base values** (BSP_RANK, IS_RENDERABLE, CROSS/DOT, HIT_*): compute
  from first principles using the attended wall geometry, the BSP
  side-P vector, and the player's pre-collision pose.  Math is ported
  from ``wall.py`` 1:1.
* **Derived values** (T_LO/T_HI, VIS_LO/VIS_HI): read CROSS/DOT/T from
  the KV cache via :class:`ThinkingReadback` at layer 0 of the own
  step, then apply the ported clip / projection math.  VIS gates on a
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
    piecewise_linear_2d,
    reciprocal,
    subtract,
    sum_nodes,
)
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_mean_where,
    attend_most_recent_matching,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true, bool_any_true, cond_gate
from torchwright.ops.map_select import in_range, select
from torchwright.reference_renderer.types import RenderConfig

import math

from torchwright.doom.embedding import (
    D_EMBED,
    IDENTIFIER_NAMES,
    VALUE_RANGE_BY_NAME,
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
    factor_q_to_embedding,
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
_SLOT_T_LO = IDENTIFIER_NAMES.index("T_LO")
_SLOT_T_HI = IDENTIFIER_NAMES.index("T_HI")
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
    _SLOT_DOT_B: "T_LO",
    _SLOT_T_LO: "T_HI",
    _SLOT_T_HI: "VIS_LO",
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
    wall_position_onehot: Node
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

    # From WALL positions (per-wall collision flags, aggregated via
    # attend_mean_where for the RESOLVED_X/RESOLVED_Y identifiers).
    hit_full: Node  # 1 if the full (vel_dx, vel_dy) ray hit this wall
    hit_x: Node  # 1 if the x-only ray hit this wall
    hit_y: Node  # 1 if the y-only ray hit this wall


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
    """

    next_token_embedding: Node
    is_thinking_active: Node
    readback: "ThinkingReadback"


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
    assert len(is_identifier_by_slot) == len(IDENTIFIER_NAMES) == 16, (
        f"expected 16 per-slot identifier detectors, got "
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
    # Attention hop 2 (Part 2 mechanism): most-recent-identifier slot
    # one-hot.  Stored as a 16-wide cond_gate(is_X_id, 1) at every
    # identifier position; read at VALUE positions via
    # attend_most_recent_matching against is_any_identifier.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/find_prev_identifier"):
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

    # ---------------------------------------------------------------------
    # Attention hop 3: wall geometry for the current wall.  Value block
    # packs (ax, ay, bx, by, bsp_coeffs..., bsp_const) so a single
    # attention head covers every per-wall identifier's geometric need.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/wall_geom_attention"):
        wi_clamped = clamp(current_wall_index, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wi_clamped, 1.0)
        wall_j_onehot = bool_to_01(in_range(wi_clamped, wi_p1, max_walls))

        # Gate on any per-wall identifier step (13 slots 0..12); the 3
        # RESOLVED slots don't need wall geom.  Using ``is_any_identifier``
        # subsumes this with no harm (the extra 3 positions waste a
        # compute cycle but don't affect correctness).
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
            query_vector=cond_gate(is_any_identifier, wall_j_onehot),
            key_vector=cond_gate(is_wall, kv.wall_position_onehot),
            value=cond_gate(is_wall, wall_geom_value),
            match_gain=1000.0,
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
    # RESOLVED computation: aggregate per-WALL collision flags across all
    # WALL prefill positions via attend_mean_where, then apply the
    # axis-separated wall-sliding logic from the former EOS stage.
    # ---------------------------------------------------------------------
    with annotate("thinking_wall/resolved_compute"):
        resolved_x, resolved_y, resolved_angle = _compute_resolved(
            hit_full=kv.hit_full,
            hit_x=kv.hit_x,
            hit_y=kv.hit_y,
            player_x=kv.player_x,
            player_y=kv.player_y,
            vel_dx=kv.vel_dx,
            vel_dy=kv.vel_dy,
            new_angle=kv.new_angle,
            is_wall=is_wall,
            pos_encoding=pos_encoding,
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

    with annotate("thinking_wall/bsp_rank_bound"):
        # BSP_RANK is an integer 0..7 per design doc.  The raw
        # ``assert_integer`` call in ``_compute_bsp_and_renderable``
        # preserves the natural value_type of the dot-product
        # accumulation (``max_bsp_nodes × max_coord``), which is too
        # wide for the downstream quantize affine (scale = 65535/7
        # amplifies).  Clamping to [0, 7] enforces a tight value_type
        # without adding runtime error on well-formed inputs.
        bsp_rank = clamp(bsp_rank, 0.0, 7.0)

    with annotate("thinking_wall/t_and_vis"):
        # T_LO/T_HI and VIS_LO/VIS_HI share FOV-cone clip intermediates
        # (f_L_a/b, f_R_a/b, per-plane t contribs).  Compute once and
        # emit each of the 4 values at its own identifier step.  All
        # four derived values read CROSS_A, DOT_A, CROSS_B, DOT_B from
        # the KV cache — the shared attention heads per-name, cached
        # inside ``ThinkingReadback``, make the 4-way readback cost
        # one attention per distinct name (not per consumer).
        ca_kv = readback.get_value_after_last("CROSS_A")
        da_kv = readback.get_value_after_last("DOT_A")
        cb_kv = readback.get_value_after_last("CROSS_B")
        db_kv = readback.get_value_after_last("DOT_B")
        t_lo, t_hi, vis_lo, vis_hi = _compute_clip_and_project(
            ca_kv,
            da_kv,
            cb_kv,
            db_kv,
            is_renderable,
            config=config,
            cross_dot_max=40.0,
        )
        # Clamp into the wire-format range so each quantize's input
        # value_type is tight.  Without the clamp, the natural
        # value_types from the select / cond_gate chain above inflate
        # by M·c_tol at every link, eventually overshooting the
        # quantize's declared output ``[0, 65535]`` assert.
        t_lo = clamp(t_lo, 0.0, 1.0)
        t_hi = clamp(t_hi, 0.0, 1.0)
        vis_lo_bound_hi = float(VALUE_RANGE_BY_NAME["VIS_LO"][1])
        vis_lo = clamp(vis_lo, -2.0, vis_lo_bound_hi)
        vis_hi = clamp(vis_hi, -2.0, vis_lo_bound_hi)

    # ---------------------------------------------------------------------
    # Per-slot quantize → select-q → factor-once.
    #
    # Each identifier slot quantizes its computed float into a 1-wide
    # ``q ∈ [0, 65535]`` using the slot's design-doc (lo, hi) range.
    # We then sum-cond-gate to pick the active slot's q (exactly one is
    # +1 at any identifier position; sum collapses to the active one).
    # The expensive ``thermometer_floor_div`` × 3 + ``in_range`` × 4
    # cascade in :func:`factor_q_to_embedding` runs **once** per
    # position on the selected q rather than 16 times across all slots
    # — drops the residual peak from ~1100 cols to ~80 cols.
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
            _SLOT_T_LO: _norm(t_lo, "T_LO"),
            _SLOT_T_HI: _norm(t_hi, "T_HI"),
            _SLOT_VIS_LO: _norm(vis_lo, "VIS_LO"),
            _SLOT_VIS_HI: _norm(vis_hi, "VIS_HI"),
            _SLOT_HIT_FULL: _norm(bool_to_01(hit_full), "HIT_FULL"),
            _SLOT_HIT_X: _norm(bool_to_01(hit_x), "HIT_X"),
            _SLOT_HIT_Y: _norm(bool_to_01(hit_y), "HIT_Y"),
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
        emit = factor_q_to_embedding(q_selected, suffix="_id_emit")
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
# Split into two stages so T_LO/T_HI can emit before VIS needs them.
# ---------------------------------------------------------------------------


def _compute_clip_and_project(
    cross_a: Node,
    dot_a: Node,
    cross_b: Node,
    dot_b: Node,
    is_renderable: Node,
    *,
    config: RenderConfig,
    cross_dot_max: float,
):
    """T_LO / T_HI / VIS_LO / VIS_HI in one pass.

    Shares FOV-cone-clip intermediates (``f_L_a/b``, ``f_R_a/b``,
    per-plane ``(t_lo_contrib, t_hi_contrib)``) between the
    T-parameter aggregation and the VIS projection's clip-side
    decision.  Matches ``wall._compute_visibility_columns`` 1:1 in
    structure — the split into separate emit steps happens at the
    caller, not here.

    T_LO / T_HI appear in the VIS-empty check (``t_lo > t_hi``) and
    the clip-side decisions (``a_clipped_on_L``, ``b_clipped_on_L``)
    derive from the *same* ``t_lo_L/R`` / ``t_hi_L/R`` nodes T_LO/T_HI
    do, so there is no round-trip through the KV cache — every shared
    intermediate is computed once.

    is_renderable is the locally-computed flag (not a KV-cache
    readback) — gating with the fresh value avoids routing a boolean
    through a readback round-trip.
    """
    W = config.screen_width
    fov = config.fov_columns
    fov_rad = float(fov) * math.pi / 128.0
    half_fov_rad = fov_rad / 2.0
    sin_hf = math.sin(half_fov_rad)
    cos_hf = math.cos(half_fov_rad)

    # FOV-boundary evaluations at both endpoints.
    f_L_a = add_scaled_nodes(sin_hf, dot_a, -cos_hf, cross_a)
    f_L_b = add_scaled_nodes(sin_hf, dot_b, -cos_hf, cross_b)
    f_R_a = add_scaled_nodes(sin_hf, dot_a, cos_hf, cross_a)
    f_R_b = add_scaled_nodes(sin_hf, dot_b, cos_hf, cross_b)

    # Size reciprocal / multiply_2d grids by the readback cross/dot
    # range (the design-doc CROSS_A/DOT_A nominal), not ``max_coord``.
    # In wall.py these were sized from ``max_coord`` because cross/dot
    # came from locally-rotated ``max_coord``-bounded geometry; here
    # they come from the KV-cache readback with the wider wire-format
    # range (±40 per design doc).
    max_f_mag = (sin_hf + cos_hf) * cross_dot_max
    max_denom = 2.0 * max_f_mag

    t_lo_L, t_hi_L = _plane_clip_contribs(
        f_L_a, f_L_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="tw_L"
    )
    t_lo_R, t_hi_R = _plane_clip_contribs(
        f_R_a, f_R_b, max_denom=max_denom, max_f_mag=max_f_mag, suffix="tw_R"
    )

    zero_lit = create_literal_value(torch.tensor([0.0]), name="tw_t_zero")
    one_lit = create_literal_value(torch.tensor([1.0]), name="tw_t_one")
    W_lit = create_literal_value(torch.tensor([float(W)]), name="tw_col_W")

    t_lo = max_node(max_node(zero_lit, t_lo_L), t_lo_R)
    t_hi = min_node(min_node(one_lit, t_hi_L), t_hi_R)

    is_empty = compare(multiply_const(subtract(t_lo, t_hi), _T_COMPARE_SCALE), 0.0)

    col_A_interior = _endpoint_to_column(cross_a, dot_a, W=W, fov=fov, suffix="tw_a")
    col_B_interior = _endpoint_to_column(cross_b, dot_b, W=W, fov=fov, suffix="tw_b")

    a_inside_L = compare(f_L_a, 0.0)
    a_inside_R = compare(f_R_a, 0.0)
    a_clipped_on_L = compare(
        multiply_const(subtract(t_lo_L, t_lo_R), _T_COMPARE_SCALE), 0.0
    )
    col_A_boundary = select(a_clipped_on_L, W_lit, zero_lit, c_tol=_VIS_C_TOL)
    col_A = select(
        a_inside_L,
        select(a_inside_R, col_A_interior, col_A_boundary, c_tol=_VIS_C_TOL),
        col_A_boundary,
        c_tol=_VIS_C_TOL,
    )

    b_inside_L = compare(f_L_b, 0.0)
    b_inside_R = compare(f_R_b, 0.0)
    b_clipped_on_L = compare(
        multiply_const(subtract(t_hi_R, t_hi_L), _T_COMPARE_SCALE), 0.0
    )
    col_B_boundary = select(b_clipped_on_L, W_lit, zero_lit, c_tol=_VIS_C_TOL)
    col_B = select(
        b_inside_L,
        select(b_inside_R, col_B_interior, col_B_boundary, c_tol=_VIS_C_TOL),
        col_B_boundary,
        c_tol=_VIS_C_TOL,
    )

    vis_lo_visible = min_node(col_A, col_B)
    vis_hi_visible = max_node(col_A, col_B)

    sentinel = create_literal_value(
        torch.tensor([float(W + 2)]), name="tw_vis_sentinel"
    )
    vis_lo_raw = select(is_empty, sentinel, vis_lo_visible, c_tol=_VIS_C_TOL)
    vis_hi_raw = select(is_empty, sentinel, vis_hi_visible, c_tol=_VIS_C_TOL)

    vis_lo = cond_gate(is_renderable, vis_lo_raw, c_tol=_VIS_C_TOL)
    vis_hi = cond_gate(is_renderable, vis_hi_raw, c_tol=_VIS_C_TOL)
    return t_lo, t_hi, vis_lo, vis_hi


def _plane_clip_contribs(
    f_a: Node,
    f_b: Node,
    *,
    max_denom: float,
    max_f_mag: float,
    suffix: str,
):
    """Per-plane clip contrib — ported from ``wall._plane_clip_contribs``."""
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

    zero_lit = create_literal_value(torch.tensor([0.0]), name=f"tw_tzero_{suffix}")
    one_lit = create_literal_value(torch.tensor([1.0]), name=f"tw_tone_{suffix}")
    a_inside = compare(f_a, 0.0)
    b_inside = compare(f_b, 0.0)

    t_lo_contrib = select(a_inside, zero_lit, t_star, c_tol=_VIS_C_TOL)
    t_hi_contrib = select(b_inside, one_lit, t_star, c_tol=_VIS_C_TOL)
    return t_lo_contrib, t_hi_contrib


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
    hit_full: Node,
    hit_x: Node,
    hit_y: Node,
    player_x: Node,
    player_y: Node,
    vel_dx: Node,
    vel_dy: Node,
    new_angle: Node,
    is_wall: Node,
    pos_encoding: PosEncoding,
):
    """Axis-separated wall sliding — ported from the former EOS stage.

    Aggregates per-WALL collision flags across WALL prefill positions
    via ``attend_mean_where``; any per-flag mean above ``1/max_walls``
    implies at least one wall was hit on that ray.  The resolved
    position on each axis uses the velocity component if *neither* the
    full ray nor that axis's lone ray hit anything; otherwise the
    player stays put on that axis.

    ``new_angle`` passes through unchanged — collision doesn't rotate
    the player.
    """
    hit_full_01 = bool_to_01(hit_full)
    hit_x_01 = bool_to_01(hit_x)
    hit_y_01 = bool_to_01(hit_y)

    resolve_attn = attend_mean_where(
        pos_encoding,
        validity=is_wall,
        value=Concatenate([hit_full_01, hit_x_01, hit_y_01]),
    )
    avg_hf = extract_from(resolve_attn, 3, 0, 1, "tw_avg_hf")
    avg_hx = extract_from(resolve_attn, 3, 1, 1, "tw_avg_hx")
    avg_hy = extract_from(resolve_attn, 3, 2, 1, "tw_avg_hy")

    any_hit_full = compare(avg_hf, 0.05)
    any_hit_x = compare(avg_hx, 0.05)
    any_hit_y = compare(avg_hy, 0.05)

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

    prev_was_hy_01 = extract_from(
        prev_slot_onehot, len(IDENTIFIER_NAMES), _SLOT_HIT_Y, 1, "prev_was_hy"
    )
    prev_was_hy_bool = compare(prev_was_hy_01, 0.5)
    hy_contribution = cond_gate(prev_was_hy_bool, hy_successor)

    next_at_value_total = sum_nodes([static_next_at_value, hy_contribution])
    value_contribution = cond_gate(is_thinking_value, next_at_value_total)

    return sum_nodes([marker_contribution, identifier_contribution, value_contribution])
