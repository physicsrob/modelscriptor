"""Codec between thinking-token float values and 72-wide VALUE embeddings.

Two sides of the same boundary:

* **Emit** (producer): a per-wall identifier step computes a float (e.g.
  ``CROSS_A = -17.3``) and this module's :func:`emit_continuous_value_embedding`
  / :func:`emit_integer_value_embedding` / :func:`emit_boolean_value_embedding`
  produces the 72-wide ``W_EMBED`` row the host argmaxes against to pick
  the next VALUE token ID.
* **Readback** (consumer): a later thinking step needs a prior value
  from the KV cache (e.g. ``T_LO`` reads ``CROSS_A``) and
  :class:`ThinkingReadback` runs a single ``attend_most_recent_matching``
  + a Linear decode to turn the 64-wide VALUE payload back into a
  dequantized float.

The wire format between the two is the 16-bit integer VALUE ID: every
VALUE row ``k`` in ``W_EMBED`` shares the same 8-wide E8 category code
in cols [0:8] and carries ``k``'s 16-bit payload as a 4+4+4+4 factored
one-hot in cols [8:72].

Float ↔ VALUE-ID mapping per identifier name is
``VALUE_RANGE_BY_NAME[name]``:

    q = (value - lo) * (N_VALUES - 1) / (hi - lo)          # produce side
    value = lo + q * (hi - lo) / (N_VALUES - 1)             # consume side

where ``N_VALUES = 65536``.  Host-side uint16 rounding happens between
the two and contributes one LSB per quantization boundary.

Depth accounting for the emit path (continuous value):

    clamp           1 layer
    floor h3        1
    floor h2        1 (sequential with h3 via residue subtract)
    floor h1        1
    in_range × 4    1 (final layer, parallel over digits)

≈5 layers after the value is computed.  Integer / boolean emits are
~2 layers (a single ``Linear`` row lookup or ``select`` between two
embedding rows).
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_in_range
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_const,
    bool_to_01,
    clamp,
    compare,
    multiply_const,
    subtract,
    thermometer_floor_div,
)
from torchwright.ops.attention_ops import attend_most_recent_matching
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true
from torchwright.ops.map_select import in_range
from torchwright.ops.quantization import DEFAULT_N_LEVELS, quantize_to_range

from torchwright.doom.embedding import (
    D_EMBED,
    E8_VALUE,
    IDENTIFIER_NAMES,
    VALUE_RANGE_BY_NAME,
    embed_lookup,
)
from torchwright.doom.graph_utils import extract_from

__all__ = [
    "factor_q_to_embedding",
    "emit_continuous_value_embedding",
    "emit_integer_value_embedding",
    "emit_boolean_value_embedding",
    "ThinkingReadback",
    "build_thinking_readback",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Width of the VALUE payload region in the embedding (cols [8:72]).  The
# first 8 cols are the E8_VALUE category code shared across all VALUE
# rows; the remaining 64 cols factor the 16-bit payload as 4 blocks of
# 16-wide one-hots.
_VALUE_PAYLOAD_START = 8
_VALUE_PAYLOAD_WIDTH = D_EMBED - _VALUE_PAYLOAD_START
assert _VALUE_PAYLOAD_WIDTH == 64

_HEX_BLOCK = 16  # one-hot width per hex digit
_N_HEX_DIGITS = 4

# Match-gain for the readback's ``attend_most_recent_matching``.  Same
# value the prev-id attention uses at 16-wide (M4/Part 2 empirics).
_READBACK_MATCH_GAIN = 12000.0


# ---------------------------------------------------------------------------
# Emit helpers (producer side)
# ---------------------------------------------------------------------------


def factor_q_to_embedding(q: Node, suffix: str = "") -> Node:
    """Convert a quantized integer float ``q ∈ [0, 65535]`` to its
    72-wide VALUE-row embedding.

    Decomposes ``q`` into 4 hex digits and builds the 4+4+4+4 factored
    one-hot payload, prepended by the shared ``E8_VALUE`` category code
    in cols [0:8].

    This is the *slot-agnostic* half of the emit path — the per-slot
    quantize affine (``value → q``) lives outside.  The whole point of
    factoring it this way is so the cascade runs **once** per position
    even when many candidate values are computed per position; only the
    selected ``q`` flows in.

    **Phase B Part 1 — hi/lo split.**  Split ``q`` into hi/lo bytes
    ``q_hi = q // 256`` and ``q_lo = q - q_hi * 256`` first, then
    extract the two hex digits of each byte.  The two ``// 16`` digit
    extractions run in parallel off ``q_hi`` and ``q_lo`` — two serial
    ``thermometer_floor_div`` calls instead of three.

    Depth: ~3 MLP sublayers of digit cascade (clamp → q_hi → h3 ∥ h1
    → in_range × 4 in parallel).

    Args:
        q: 1-wide float node holding a near-integer value in
            ``[0, 65535]``.  Far-from-integer inputs hit
            ``thermometer_floor_div`` ramp zones at ``k * divisor - 0.5``
            (``divisor`` is 256 for the hi/lo split and 16 for the
            inner digit extraction); at those boundaries two adjacent
            slots in one block soften to ~0.5 each and the host's
            argmax against ``W_EMBED.T`` picks the nearer neighbour
            (≤1 LSB error in the recovered float).
        suffix: Used to namespace the literal nodes built inside.

    Returns:
        72-wide embedding node — the row the host argmaxes against
        ``W_EMBED.T`` to pick the next VALUE token ID.
    """
    q_clamped = clamp(q, 0.0, float(DEFAULT_N_LEVELS - 1))

    # Split ``q`` into hi/lo bytes.  ``q_hi = q // 256`` is the serial
    # dependency; ``q_lo`` is a pure affine on (q_clamped, q_hi).
    q_hi = thermometer_floor_div(q_clamped, 256, DEFAULT_N_LEVELS - 1)
    q_lo = subtract(q_clamped, multiply_const(q_hi, 256.0))

    # Per-byte digit extraction runs in parallel: h3/h2 from q_hi,
    # h1/h0 from q_lo.  Each byte needs a single ``// 16`` — the two
    # ``thermometer_floor_div`` calls share an MLP layer instead of
    # chaining.
    h3 = thermometer_floor_div(q_hi, 16, 255)
    h2 = subtract(q_hi, multiply_const(h3, 16.0))
    h1 = thermometer_floor_div(q_lo, 16, 255)
    h0 = subtract(q_lo, multiply_const(h1, 16.0))

    # 16-wide one-hot per digit: ``in_range(d, d+1, 16)`` fires slot
    # ``round(d)``; ``bool_to_01`` converts the ±1 result to 0/1.
    h3_oh = bool_to_01(in_range(h3, add_const(h3, 1.0), _HEX_BLOCK))
    h2_oh = bool_to_01(in_range(h2, add_const(h2, 1.0), _HEX_BLOCK))
    h1_oh = bool_to_01(in_range(h1, add_const(h1, 1.0), _HEX_BLOCK))
    h0_oh = bool_to_01(in_range(h0, add_const(h0, 1.0), _HEX_BLOCK))

    e8_cat = create_literal_value(E8_VALUE, name=f"e8_value_cat{suffix}")
    return Concatenate([e8_cat, h3_oh, h2_oh, h1_oh, h0_oh])


def emit_continuous_value_embedding(
    value: Node,
    name: str,
) -> Node:
    """Build a 72-wide VALUE embedding for a continuous float.

    Convenience wrapper that quantizes ``value`` into ``[0, 65535]``
    using ``VALUE_RANGE_BY_NAME[name]`` and then runs
    :func:`factor_q_to_embedding`.  Useful when a caller emits exactly
    one value type and doesn't need the per-slot select-then-factor
    structure.
    """
    lo, hi = VALUE_RANGE_BY_NAME[name]
    q = quantize_to_range(value, lo, hi)
    return factor_q_to_embedding(q, suffix=f"_{name}")


def emit_integer_value_embedding(
    integer_value: Node,
    max_int: int,
    name: str,
) -> Node:
    """Build a 72-wide VALUE embedding for an integer ``v ∈ [0, max_int]``.

    Used for identifiers whose value is a small-cardinality integer
    (``BSP_RANK`` 0..7, ``RESOLVED_ANGLE`` 0..255).  The quantize +
    factor path from :func:`emit_continuous_value_embedding` is correct
    but overkill; here we build a one-hot over the small-cardinality
    domain and let a single ``Linear`` index the ``W_EMBED`` rows
    ``[VALUE_0, VALUE_1, …, VALUE_{max_int}]`` directly.

    Depth: 2 MLP sublayers (1 for ``in_range``, 1 for the Linear row
    lookup — the ``bool_to_01`` is a free affine).

    Args:
        integer_value: 1-wide float node whose value is an integer in
            ``[0, max_int]``.
        max_int: Largest integer the value can take.  The one-hot
            width is ``max_int + 1``.
        name: Identifier name (for node debug naming).

    Returns:
        72-wide embedding node.
    """
    n = max_int + 1
    onehot = bool_to_01(in_range(integer_value, add_const(integer_value, 1.0), n))

    rows = torch.stack(
        [embed_lookup(f"VALUE_{k}") for k in range(n)], dim=0
    )  # (n, D_EMBED)
    return Linear(
        onehot,
        rows,  # (n, D_EMBED) — Linear multiplies (input @ matrix), so a
        # one-hot row selection with matrix rows of VALUE embeddings.
        torch.zeros(D_EMBED),
        name=f"emit_int_{name}",
    )


def emit_boolean_value_embedding(
    bool_value: Node,
    name: str,
) -> Node:
    """Build a 72-wide VALUE embedding for a ±1 boolean.

    +1 → ``W_EMBED[VALUE_1]``, -1 → ``W_EMBED[VALUE_0]``.

    Depth: 1 MLP sublayer (the ``cond_gate`` on the 72-wide delta).

    Args:
        bool_value: 1-wide node with value in ``{-1, +1}``.
        name: Identifier name (for node debug naming).

    Returns:
        72-wide embedding node.
    """
    e_value_0 = create_literal_value(embed_lookup("VALUE_0"), name=f"e_v0_{name}")
    e_value_1 = create_literal_value(embed_lookup("VALUE_1"), name=f"e_v1_{name}")
    # base + cond_gate(bool, delta) — same pattern thinking_wall.py
    # uses for HIT_* emission today.
    from torchwright.ops.arithmetic_ops import sum_nodes
    from torchwright.ops.logic_ops import cond_gate

    delta = subtract(e_value_1, e_value_0)
    return sum_nodes([e_value_0, cond_gate(bool_value, delta)])


# ---------------------------------------------------------------------------
# Readback (consumer side)
# ---------------------------------------------------------------------------


@dataclass
class _ReadbackContext:
    """Shared inputs the :class:`ThinkingReadback` captures once."""

    embedding: Node
    prev_id_slots: Node  # 16-wide slot one-hot from thinking_wall
    is_value_category: Node  # ±1 flag, true at VALUE positions
    pos_encoding: PosEncoding


class ThinkingReadback:
    """Handle to the thinking-token KV-cache readback machinery.

    Call :meth:`get_value_after_last` with an identifier name to
    retrieve the dequantized float emitted at the most recent VALUE
    position whose preceding identifier was that name.  Results are
    cached per name: repeated calls for the same identifier return the
    same Node.

    Emitted flag / attention / Linear machinery is rebuilt per
    identifier on first request.
    """

    def __init__(self, ctx: _ReadbackContext):
        self._ctx = ctx
        self._cache: Dict[str, Node] = {}
        self._indicator_cache: Dict[str, Node] = {}
        # The 1-wide constant query vector (``+1``) used by
        # ``attend_most_recent_matching``.  Single shared node so every
        # identifier's attention head shares a literal.
        self._query_const_1 = create_literal_value(
            torch.tensor([1.0]), name="readback_q1"
        )

    def is_value_of(self, name: str) -> Node:
        """Return the ±1 per-position indicator for "this position is
        a VALUE token whose preceding identifier was ``name``."

        Useful as a key-side channel for content-attention callers that
        want to match against ``name``-VALUE positions in the KV cache
        without paying for the readback Linear.  ``get_value_after_last``
        builds this same indicator internally; exposing it lets other
        stages (e.g., the SORTED stage's VIS_LO content attention)
        compose their own queries.

        Cached per name: repeated calls return the same Node.
        """
        if name in self._indicator_cache:
            return self._indicator_cache[name]
        if name not in VALUE_RANGE_BY_NAME:
            raise KeyError(
                f"unknown identifier {name!r}; must be one of {IDENTIFIER_NAMES}"
            )
        slot = IDENTIFIER_NAMES.index(name)
        prev_slot_i_01 = extract_from(
            self._ctx.prev_id_slots,
            len(IDENTIFIER_NAMES),
            slot,
            1,
            f"readback_prev_slot_{name}",
        )
        prev_slot_i_bool = compare(prev_slot_i_01, 0.5)
        indicator = bool_all_true([self._ctx.is_value_category, prev_slot_i_bool])
        self._indicator_cache[name] = indicator
        return indicator

    def get_value_after_last(
        self, name: str, *, assert_hardness_gt: float | None = 0.99
    ) -> Node:
        """Return the dequantized float for the most recent matching VALUE.

        ``name`` must be one of the 21 entries in ``IDENTIFIER_NAMES``.
        The returned Node is 1-wide with value range
        ``VALUE_RANGE_BY_NAME[name]``.

        **Undefined when the referenced identifier has no prior
        instance in the causal window.**  Most Phase-A call sites
        place the consuming step downstream of the producing step in
        the same frame, so the cache always contains the referenced
        identifier by the time a consumer fires.  Phase B's running-OR
        HIT_* consumers fire at every wall including wall 0; the
        caller must gate the result accordingly and may pass
        ``assert_hardness_gt=None`` to skip the runtime softmax-hardness
        check at positions where the cache might legitimately be empty.

        Caches per-name: the first call builds the flag + attention +
        Linear; subsequent calls reuse those nodes.  The
        ``assert_hardness_gt`` argument only affects the first call —
        cache hits return the original node regardless of the
        requested threshold.
        """
        if name in self._cache:
            return self._cache[name]

        # 1. Build is_X_value: ``is_value_category AND prev_slot_onehot[slot]``.
        #    The slot one-hot stored by thinking_wall is already gated
        #    to live only at identifier positions, so at a VALUE
        #    position the prev_slot_onehot reads the *previous* id's
        #    slot.  The AND with is_value_category restricts the key
        #    signal to VALUE positions only (keys at identifier
        #    positions would otherwise confuse the attention).
        is_X_value = self.is_value_of(name)

        # 2. Attention value: the 64-wide VALUE payload region of the
        #    embedding.  Category code cols [0:8] are the same at every
        #    VALUE position, so there's no point routing them through
        #    the head — they'd occupy V-columns without informing the
        #    decode.
        payload = extract_from(
            self._ctx.embedding,
            D_EMBED,
            _VALUE_PAYLOAD_START,
            _VALUE_PAYLOAD_WIDTH,
            f"readback_payload_{name}",
        )

        matched_payload = attend_most_recent_matching(
            pos_encoding=self._ctx.pos_encoding,
            query_vector=self._query_const_1,
            key_vector=is_X_value,
            value=payload,
            match_gain=_READBACK_MATCH_GAIN,
            assert_hardness_gt=assert_hardness_gt,
        )

        # 3. Decode 4+4+4+4 one-hots back to the float value via a
        #    single Linear with fused dequantize affine.
        float_value = _decode_payload_to_float(matched_payload, name)
        self._cache[name] = float_value
        return float_value


def build_thinking_readback(
    embedding: Node,
    prev_id_slots: Node,
    is_value_category: Node,
    pos_encoding: PosEncoding,
) -> ThinkingReadback:
    """Factory for :class:`ThinkingReadback`.

    Args:
        embedding: 72-wide ``W_EMBED`` leaf (per-position embedding).
        prev_id_slots: 16-wide node carrying the slot one-hot of the
            most recent identifier (as built by ``thinking_wall``'s
            prev-id attention).
        is_value_category: 1-wide ±1 flag, true at VALUE positions
            (``equals_vector(embedding[0:8], E8_VALUE)``).
        pos_encoding: The graph's positional encoding.
    """
    return ThinkingReadback(
        _ReadbackContext(
            embedding=embedding,
            prev_id_slots=prev_id_slots,
            is_value_category=is_value_category,
            pos_encoding=pos_encoding,
        )
    )


# ---------------------------------------------------------------------------
# Decode Linear (readback internals)
# ---------------------------------------------------------------------------


def _decode_payload_to_float(payload: Node, name: str) -> Node:
    """Decode a 64-wide VALUE payload (4×16 one-hots) to dequantized float.

    The 4+4+4+4 factored one-hot encodes an integer
    ``k = 4096·h3 + 256·h2 + 16·h1 + h0``.  Dequantize:
    ``value = lo + k · (hi - lo) / (N_LEVELS - 1)``.

    Composing the "decode to integer" and "dequantize to float" steps
    into a single Linear: the ``(64, 1)`` weight matrix has column 0
    equal to the dequantize scale times the integer weight of each
    one-hot position.  Bias column is ``lo``.
    """
    lo, hi = VALUE_RANGE_BY_NAME[name]
    inv_scale = (hi - lo) / (DEFAULT_N_LEVELS - 1)

    # Integer weights per block position: 4096·i for h3, 256·i for h2,
    # 16·i for h1, i for h0.  Positions run i = 0..15 within each block.
    weights = torch.zeros(_VALUE_PAYLOAD_WIDTH, 1)
    for i in range(_HEX_BLOCK):
        weights[0 * _HEX_BLOCK + i, 0] = i * 4096.0
        weights[1 * _HEX_BLOCK + i, 0] = i * 256.0
        weights[2 * _HEX_BLOCK + i, 0] = i * 16.0
        weights[3 * _HEX_BLOCK + i, 0] = float(i)
    weights = weights * inv_scale
    bias = torch.tensor([lo])

    decoded = Linear(payload, weights, bias, name=f"readback_decode_{name}")
    # Declare the float's value_range so downstream ops (T_LO / VIS
    # clip-and-project) get tight value_type bounds.  Without this the
    # compiler treats the decoded float as unbounded, which inflates
    # every downstream clamp / cond_gate M offset — test_affine_bounds
    # fires on the resulting blow-up.
    return assert_in_range(decoded, lo, hi)
