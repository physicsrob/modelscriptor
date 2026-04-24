"""SORTED stage: three-position pipeline over (SORTED_WALL marker →
SORT_RESULT id → SORT_RESULT VALUE) with a quadratic-equality attention
reading BSP ranks from thinking-phase KV.

Phase B Part 2 reshapes SORTED from a single-token stage reading prefill
WALL payloads into a three-token pipeline reading thinking-phase
intermediates.  Each wall transition now fires the three-token sequence:

    ...RENDER → SORTED_WALL → SORT_RESULT → VALUE(wall_index) → RENDER...

with ``wall_counter`` incrementing at the SORT_RESULT id position (the
single position at which the quadratic attention computes the picked
wall_index).  The VALUE position's host-echoed payload carries the
picked ``wall_index`` in a VALUE token (factored 4+4+4+4 one-hot), which
downstream stages (RENDER) read at layer 0 via content attention on
``(identifier=SORT_RESULT, wall_index=wall_index)`` against the KV cache.

Quadratic-equality mechanism
-----------------------------

Goal: pick the wall whose ``bsp_rank`` equals the current SORTED ordinal
``N`` (= ``wall_counter``).  Expanding the squared distance

    score(key) = −(bsp_rank − N)² = −bsp_rank² + 2 N bsp_rank − N²

drops the query-only ``−N²`` constant (it falls out of softmax), leaving
a standard ``query · key`` dot product with

    query at SORT_RESULT id       : [2N, 1]
    key   at BSP_RANK id (rend)   : [bsp_rank, −bsp_rank²]
    key   elsewhere (sentinel)    : [−100, −1000]

At ``match_gain=20`` the matching renderable key wins over its nearest
renderable neighbour (score gap 1) by ``e^20`` and over any sentinel
key (score gap ≥ 1050) by ``e^20000+``.

Exhaustion is detected post-hoc: the attention also returns the picked
wall's ``bsp_rank`` via a value-side scalar channel; if that rank
disagrees with ``N`` (specifically ``N > picked_bsp_rank``), every
renderable wall has been sorted already and the host terminates.

Interface to thinking_wall
--------------------------

``thinking_wall`` exposes two scalar KV channels on ``ThinkingWallOutput``:

* ``bsp_rank_scalar_for_sort``  — the K's first column
* ``bsp_rank_neg_sq_for_sort``  — the K's second column

Both carry the sentinel values at non-renderable or non-BSP_RANK-id
positions (see thinking_wall for the gating).
"""

from dataclasses import dataclass

import torch

from torchwright.graph import Concatenate, Linear, Node, annotate
from torchwright.graph.asserts import (
    assert_in_range,
)
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_const,
    bool_to_01,
    clamp,
    compare,
    multiply_const,
    subtract,
    sum_nodes,
)
from torchwright.ops.attention_ops import (
    attend_argmax_dot,
    attend_most_recent_matching,
)
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import cond_gate
from torchwright.ops.map_select import in_range, select

from torchwright.doom.embedding import (
    D_EMBED,
    VALUE_RANGE_BY_NAME,
    embed_lookup,
)
from torchwright.doom.graph_utils import extract_from
from torchwright.doom.thinking_readback import (
    ThinkingReadback,
    emit_integer_value_embedding,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Softmax gain for the quadratic-equality attention.  The matching key's
# score lead over its adjacent neighbour is 1 (in bsp_rank spacing), so
# match_gain 20 gives ``e^20 ≈ 4.85e8`` concentration — comfortably hard
# for ``assert_hardness_gt=0.99``.
_QUAD_MATCH_GAIN = 20.0

# Match-gain for the content attentions (wall_index + VIS_LO at the
# SORT_RESULT VALUE position).  Same regime as the thinking-phase
# prev-id attention at 20-wide: 12000 gives sharp softmax concentration
# for unit-dot matches at typical sequence lengths.
_CONTENT_MATCH_GAIN = 12000.0

# Width of the wall_index one-hot the attention returns — bounded by
# ``max_walls`` but we standardise on 8 (the vocabulary's
# ``THINKING_WALL_0..7`` count).
_WALL_ONEHOT_WIDTH = 8


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class SortedToken:
    """Per-position inputs.  ``wall_counter`` is the host-fed overlay
    field; the autoregressive loop increments it at the SORT_RESULT id
    position."""

    wall_counter: Node


@dataclass
class SortedKVInput:
    """Cross-position values read via attention.

    All values come from thinking-phase positions via ``ThinkingWallOutput``
    — Phase B Part 2 no longer consumes the prefill WALL payload that
    SORTED previously used.
    """

    # BSP_RANK scalar K/V channels exposed by thinking_wall.  Both are
    # sentinel-valued at non-renderable / non-BSP_RANK-id positions.
    bsp_rank_scalar: Node
    bsp_rank_neg_sq: Node

    # 1-hot of the wall_index at thinking IDENTIFIER positions — used
    # as the V-source channel for the quadratic-equality attention's
    # read of BSP_RANK id positions.  Zero outside the thinking phase.
    identifier_wall_index_onehot: Node

    # 1-hot of the wall_index at thinking VALUE positions — used as a
    # key-side channel for the VIS_LO content attention.
    value_wall_index_onehot: Node

    # BSP_RANK identifier detector (from game_graph._detect_token_types).
    is_bsp_rank_id: Node  # ±1 at BSP_RANK identifier positions

    # Readback handle shared across stages.  Used to retrieve per-wall
    # VIS_LO via ``(identifier=VIS_LO, wall_index)`` content match;
    # ``is_value_of("VIS_LO")`` supplies the key-side type indicator.
    readback: ThinkingReadback

    # Embedding leaf (host-echoed token IDs → 72-wide embedding).  Used
    # locally to decode the SORT_RESULT VALUE payload at the VALUE
    # position without going through readback.
    embedding: Node


@dataclass
class SortedTokenOutput:
    """Outputs across the 3-position SORT pipeline.

    Fields are meaningful only at their named position; elsewhere
    ``_assemble_output`` gates them to 0 or a don't-care literal.

    SORTED_WALL marker position
        * ``marker_next_embedding`` — emit the SORT_RESULT identifier
          as the next token.

    SORT_RESULT id position
        * ``sort_result_next_embedding`` — emit VALUE(wall_index) as
          the next token (via ``emit_integer_value_embedding``).
        * ``wall_index`` — scalar 0..max_walls-1, host-visible for
          the trace harness.
        * ``picked_bsp_rank`` — integer rank returned by the quadratic
          attention (for the exhaustion check).
        * ``sort_done`` — ±1, true iff the quadratic attention
          exhausted the renderable walls (``N > picked_bsp_rank``).
        * ``next_wall_counter`` — overlay output = wall_counter + 1.

    SORT_RESULT VALUE position
        * ``value_next_embedding`` — emit RENDER as the next token.
        * ``vis_lo`` — starting screen column for the picked wall,
          read via content attention on thinking VIS_LO positions
          keyed by ``(identifier=VIS_LO, wall_index)``.
    """

    marker_next_embedding: Node
    sort_result_next_embedding: Node
    value_next_embedding: Node
    wall_index: Node
    picked_bsp_rank: Node
    sort_done: Node
    next_wall_counter: Node
    vis_lo: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_sorted(
    token: SortedToken,
    kv: SortedKVInput,
    *,
    is_sorted_marker: Node,
    is_sort_result_id: Node,
    is_sort_result_value: Node,
    pos_encoding: PosEncoding,
    max_walls: int,
) -> SortedTokenOutput:
    """Wire up the 3-position SORT pipeline.

    Args:
        token: Per-position host-fed fields (just ``wall_counter``).
        kv: Cross-position values read via attention.
        is_sorted_marker: ±1 detector for ``SORTED_WALL`` markers.
        is_sort_result_id: ±1 detector for ``SORT_RESULT`` identifiers.
        is_sort_result_value: ±1 detector for SORT_RESULT VALUE
            positions (VALUE whose preceding identifier was
            SORT_RESULT).
        pos_encoding: The graph's positional encoding.
        max_walls: Number of walls in the scene (bounds the wall_index
            one-hot width).
    """
    assert max_walls <= _WALL_ONEHOT_WIDTH, (
        f"max_walls={max_walls} exceeds the SORT pipeline's one-hot "
        f"width {_WALL_ONEHOT_WIDTH}; widen if the vocabulary grows."
    )

    # ---- SORTED_WALL marker: fixed SORT_RESULT next-token. ----
    marker_next_embedding = create_literal_value(
        embed_lookup("SORT_RESULT"), name="marker_next_sort_result"
    )

    # ---- SORT_RESULT id: quadratic-equality attention. ----
    wall_index, picked_bsp_rank = _quadratic_equality_pick(
        token.wall_counter,
        kv,
        is_sort_result_id=is_sort_result_id,
        max_walls=max_walls,
    )

    # Sort_done fires when the host has requested a wall at rank >
    # picked_bsp_rank — i.e., every renderable wall has been sorted
    # already, and the attention returned the highest-rank renderable
    # wall as a best-effort (but its rank < N).  The caller reads
    # this at SORT_RESULT id positions and terminates the render loop.
    with annotate("sort/exhaustion_check"):
        rank_gap = subtract(token.wall_counter, picked_bsp_rank)
        sort_done = compare(rank_gap, 0.5)

    # wall_index factored into the VALUE embedding the transformer
    # emits at this step (picked up by the host on the next step).
    with annotate("sort/emit_wall_index"):
        sort_result_next_embedding = emit_integer_value_embedding(
            wall_index, max_int=max_walls - 1, name="SORT_RESULT"
        )

    # Wall_counter increments at SORT_RESULT id so the next SORTED
    # cycle sees N+1.
    next_wall_counter = add_const(token.wall_counter, 1.0)

    # ---- SORT_RESULT VALUE: read vis_lo + emit RENDER next. ----
    value_next_embedding = create_literal_value(
        embed_lookup("RENDER"), name="sort_value_next_render"
    )

    with annotate("sort/vis_lo_lookup"):
        vis_lo = _read_vis_lo_for_this_wall(
            kv,
            pos_encoding=pos_encoding,
            is_sort_result_value=is_sort_result_value,
            max_walls=max_walls,
        )

    return SortedTokenOutput(
        marker_next_embedding=marker_next_embedding,
        sort_result_next_embedding=sort_result_next_embedding,
        value_next_embedding=value_next_embedding,
        wall_index=wall_index,
        picked_bsp_rank=picked_bsp_rank,
        sort_done=sort_done,
        next_wall_counter=next_wall_counter,
        vis_lo=vis_lo,
    )


# ---------------------------------------------------------------------------
# Quadratic-equality attention
# ---------------------------------------------------------------------------


def _quadratic_equality_pick(
    wall_counter: Node,
    kv: SortedKVInput,
    *,
    is_sort_result_id: Node,
    max_walls: int,
):
    """Pick the wall whose ``bsp_rank`` equals ``wall_counter`` (= ``N``).

    Runs ``attend_argmax_dot`` with the expanded quadratic key/query.
    The returned value block is ``[wall_index_onehot (max_walls wide),
    bsp_rank_scalar]``; we Linear-decode the one-hot into a scalar
    ``wall_index`` and pass the bsp_rank straight through for the
    exhaustion check.
    """
    with annotate("sort/quad_attention_query"):
        # Query = [2N, 1], gated to zero at non-SORT_RESULT-id positions
        # so spurious query dots don't fire elsewhere.
        two_n = multiply_const(wall_counter, 2.0)
        one_literal = create_literal_value(
            torch.tensor([1.0]), name="sort_quad_query_one"
        )
        query_raw = Concatenate([two_n, one_literal])
        query_gated = cond_gate(is_sort_result_id, query_raw)

    with annotate("sort/quad_attention_key"):
        # Key is already sentinel-valued at non-BSP_RANK-id positions by
        # thinking_wall (select with sentinel on the off-branch).  No
        # additional gating needed.
        key = Concatenate([kv.bsp_rank_scalar, kv.bsp_rank_neg_sq])

    with annotate("sort/quad_attention_value"):
        # Value carries (wall_index_onehot, bsp_rank_scalar).  Only
        # BSP_RANK id positions should contribute; cond_gate zero at
        # other positions so the soft-average from any incidental
        # softmax mass lands at zero rather than mid-residual garbage.
        #
        # identifier_wall_index_onehot is wall_j_onehot gated by
        # is_any_identifier — non-zero at every thinking identifier
        # position.  The is_bsp_rank_id gate narrows further so only
        # BSP_RANK id positions contribute to V.
        bsp_only_wall_onehot = cond_gate(
            kv.is_bsp_rank_id, kv.identifier_wall_index_onehot
        )
        bsp_only_rank = cond_gate(kv.is_bsp_rank_id, kv.bsp_rank_scalar)
        value_block = Concatenate([bsp_only_wall_onehot, bsp_only_rank])

    with annotate("sort/quad_attention"):
        # assert_hardness_gt intentionally omitted: the query is gated to
        # zero at non-SORT_RESULT-id positions, where the softmax
        # degenerates to uniform and the output is garbage.  The
        # _assemble_output cascade only routes the wall_index output at
        # SORT_RESULT-id positions, so garbage elsewhere is harmless.
        selected = attend_argmax_dot(
            query_vector=query_gated,
            key_vector=key,
            value=value_block,
            match_gain=_QUAD_MATCH_GAIN,
        )

    with annotate("sort/unpack_attention"):
        block_width = max_walls + 1
        picked_onehot = extract_from(
            selected, block_width, 0, max_walls, "sort_picked_onehot"
        )
        picked_bsp_rank = extract_from(
            selected, block_width, max_walls, 1, "sort_picked_bsp_rank"
        )

        # one-hot → scalar via Linear([0, 1, 2, …, max_walls-1]).
        indices_weight = torch.arange(max_walls, dtype=torch.float32).unsqueeze(1)
        wall_index_raw = Linear(picked_onehot, indices_weight, name="sort_wall_index")
        # Clamp into the declared value_range so downstream factored
        # emit / host-facing scalar stays tight.
        wall_index = clamp(wall_index_raw, 0.0, float(max_walls - 1))

    return wall_index, picked_bsp_rank


# ---------------------------------------------------------------------------
# VIS_LO lookup at SORT_RESULT VALUE position
# ---------------------------------------------------------------------------


def _read_vis_lo_for_this_wall(
    kv: SortedKVInput,
    *,
    pos_encoding: PosEncoding,
    is_sort_result_value: Node,
    max_walls: int,
):
    """At SORT_RESULT VALUE, read the per-wall ``vis_lo`` from thinking
    VIS_LO VALUE positions keyed by ``(identifier=VIS_LO, wall_index)``.

    ``wall_index`` at this position is the same wall_index encoded in
    the token's own host-echoed VALUE payload.  We decode it from the
    local embedding (layer 0 KV), convert to an 8-wide one-hot, and
    query against thinking VIS_LO VALUE positions whose KV carries both
    an ``is_vis_lo_value`` flag and the thinking position's
    ``wall_index_onehot``.

    The standard :class:`ThinkingReadback.get_value_after_last` helper
    returns the *most recent* VIS_LO regardless of wall — not what we
    want when 8 walls all emitted VIS_LO earlier in the thinking phase.
    We build a local attention with a composite
    ``(is_vis_lo_value, wall_index_onehot)`` key instead.
    """
    # 1. Decode wall_index from the local VALUE payload at layer 0.
    with annotate("sort/decode_local_wall_index"):
        wall_index_local = _decode_local_value_to_float(kv.embedding, "SORT_RESULT")
        wall_index_clamped = clamp(wall_index_local, 0.0, float(max_walls - 1))
        wi_p1 = add_const(wall_index_clamped, 1.0)
        query_wall_onehot = bool_to_01(
            in_range(wall_index_clamped, wi_p1, _WALL_ONEHOT_WIDTH)
        )

    # 2. Build the composite key at thinking VIS_LO VALUE positions.
    # Key channel 0: 1 at VIS_LO VALUE positions, 0 elsewhere (gated
    # form of ``is_vis_lo_value``).
    # Key channels 1..max_walls: wall_index_onehot at thinking VALUE
    # positions (already gated to 0 elsewhere by thinking_wall).
    with annotate("sort/vis_lo_key_compose"):
        # Restrict to VIS_LO-id-preceded VALUE positions via the shared
        # indicator on the readback handle.
        is_vis_lo_value = kv.readback.is_value_of("VIS_LO")

        # Gate wall_index channels by is_vis_lo_value so non-VIS_LO
        # VALUE positions don't contaminate the dot.  (At non-thinking
        # positions, wall_index_onehot is already zero from thinking_wall.)
        key_type = bool_to_01(is_vis_lo_value)
        key_wall = cond_gate(is_vis_lo_value, kv.value_wall_index_onehot)
        composite_key = Concatenate([key_type, key_wall])

    # 3. Build the composite query at SORT_RESULT VALUE positions.
    with annotate("sort/vis_lo_query_compose"):
        one_literal = create_literal_value(
            torch.tensor([1.0]), name="sort_vis_lo_query_one"
        )
        query_raw = Concatenate([one_literal, query_wall_onehot])
        query_gated = cond_gate(is_sort_result_value, query_raw)

    # 4. Value-side: the 1-wide raw slot of thinking VIS_LO positions.
    # Gate by is_vis_lo_value to zero elsewhere.  Narrowing from the
    # old 17-wide (E8_VALUE + Gray) payload to just the raw slot is
    # the Phase B Part 3 encoding win — a single scalar flows through
    # the attention's V head and decodes via one scalar affine.
    with annotate("sort/vis_lo_value_gate"):
        from torchwright.doom.embedding import D_CATEGORY, D_RAW_SLOT

        vis_lo_raw = extract_from(
            kv.embedding, D_EMBED, D_CATEGORY, D_RAW_SLOT, "sort_vis_lo_raw"
        )
        gated_payload = cond_gate(is_vis_lo_value, vis_lo_raw)

    # 5. Attention + Linear decode.
    with annotate("sort/vis_lo_attention"):
        # assert_hardness_gt omitted for the same reason as the quad
        # attention above: the query is gated to zero outside
        # SORT_RESULT VALUE positions; softmax degenerates harmlessly
        # since the result is only consumed at those positions.
        matched_payload = attend_most_recent_matching(
            pos_encoding=pos_encoding,
            query_vector=query_gated,
            key_vector=composite_key,
            value=gated_payload,
            match_gain=_CONTENT_MATCH_GAIN,
        )

    with annotate("sort/vis_lo_decode"):
        vis_lo = _decode_value_payload_to_float(matched_payload, "VIS_LO")

    return vis_lo


# ---------------------------------------------------------------------------
# Helpers: build is_X_value indicator, local payload decode
# ---------------------------------------------------------------------------


def _decode_local_value_to_float(embedding: Node, name: str) -> Node:
    """Decode the local position's 1-wide raw slot to a dequantized float.

    Same decode Linear as
    :func:`torchwright.doom.thinking_readback._decode_payload_to_float`,
    but reads the CURRENT position's embedding rather than attending to
    a prior one.  Useful when the caller has already been placed at the
    position whose payload carries the value (e.g., SORT_RESULT VALUE
    decoding its own emitted wall_index).

    Raw slot layout: cols [D_CATEGORY : D_CATEGORY + D_RAW_SLOT] carries
    ``(2k + 1) / 131072`` for VALUE_k.  See
    :func:`torchwright.doom.thinking_readback._decode_payload_to_float`
    for the half-LSB-offset decode math.
    """
    from torchwright.doom.embedding import D_CATEGORY, D_RAW_SLOT

    raw = extract_from(embedding, D_EMBED, D_CATEGORY, D_RAW_SLOT, f"sort_local_{name}")
    return _decode_value_payload_to_float(raw, name)


def _decode_value_payload_to_float(payload: Node, name: str) -> Node:
    """Decode a 1-wide raw slot to a dequantized float.

    Mirrors
    :func:`torchwright.doom.thinking_readback._decode_payload_to_float`
    (local copy so ``sorted.py`` doesn't import the private helper).
    ``payload`` carries ``(2k + 1) / 131072`` for VALUE_k.
    """
    from torchwright.ops.quantization import DEFAULT_N_LEVELS

    lo, hi = VALUE_RANGE_BY_NAME[name]
    lsb = (hi - lo) / (DEFAULT_N_LEVELS - 1)
    weights = torch.tensor([[65536.0 * lsb]])
    bias = torch.tensor([lo - 0.5 * lsb])

    decoded = Linear(payload, weights, bias, name=f"sort_decode_{name}")
    return assert_in_range(decoded, lo, hi)
