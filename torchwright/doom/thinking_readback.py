"""Codec between thinking-token float values and 25-wide VALUE embeddings.

Two sides of the same boundary:

* **Emit** (producer): a per-wall identifier step computes a float (e.g.
  ``CROSS_A = -17.3``) and this module's :func:`emit_continuous_value_embedding`
  / :func:`emit_integer_value_embedding` / :func:`emit_boolean_value_embedding`
  produces the 25-wide ``W_EMBED`` row the host argmaxes against to pick
  the next VALUE token ID.
* **Readback** (consumer): a later thinking step needs a prior value
  from the KV cache (e.g. ``T_LO`` reads ``CROSS_A``) and
  :class:`ThinkingReadback` runs a single ``attend_most_recent_matching``
  + a Linear decode to turn the 1-wide raw slot back into a
  dequantized float.

The wire format between the two is the 16-bit integer VALUE ID: every
VALUE row ``k`` in ``W_EMBED`` shares the same 8-wide E8 category code
in cols [0:8], carries the normalized value ``k / 65535`` in col 8 (the
raw slot), and a 16-wide ±1 Gray code of ``k`` in cols [9:25]. The
encoder writes the same 25-wide row directly; argmax against
``W_EMBED.T`` resolves it to the nearest VALUE_k.

Float ↔ VALUE-ID mapping per identifier name is
``VALUE_RANGE_BY_NAME[name]``:

    q = (value - lo) * (N_VALUES - 1) / (hi - lo)          # produce side
    value = lo + q * (hi - lo) / (N_VALUES - 1)             # consume side

where ``N_VALUES = 65536``.  Host-side uint16 rounding happens between
the two and contributes one LSB per quantization boundary.

Depth accounting for the emit path (continuous value):

    clamp + scale   0 layers (pure affine, fused)
    L1 triangle PL  1 MLP sublayer (9-channel output)
    L2 triangle PL  1 MLP sublayer (7-channel output, off T_128 of L1)
    compare × 16    1 MLP sublayer (parallel)

≈3 MLP sublayers after the value is computed. Integer / boolean emits
are ~2 layers (a single ``Linear`` row lookup or ``select`` between two
embedding rows).
"""

import math
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
    piecewise_linear,
    subtract,
)
from torchwright.ops.attention_ops import attend_most_recent_matching
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.logic_ops import bool_all_true
from torchwright.ops.map_select import in_range
from torchwright.ops.quantization import DEFAULT_N_LEVELS, quantize_to_range

from torchwright.doom.embedding import (
    D_CATEGORY,
    D_EMBED,
    D_GRAY_PAYLOAD,
    D_RAW_SLOT,
    E8_VALUE,
    IDENTIFIER_NAMES,
    VALUE_RANGE_BY_NAME,
    embed_lookup,
)
from torchwright.doom.graph_utils import extract_from

__all__ = [
    "encode_value_binary",
    "emit_continuous_value_embedding",
    "emit_integer_value_embedding",
    "emit_boolean_value_embedding",
    "ThinkingReadback",
    "build_thinking_readback",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Cols [0:D_CATEGORY] are the E8_VALUE category code shared across all
# VALUE rows.  The raw slot (col D_CATEGORY) carries the normalized
# value; the 16-wide Gray-code payload sits immediately after it.
_RAW_SLOT_START = D_CATEGORY
_GRAY_START = D_CATEGORY + D_RAW_SLOT
assert _RAW_SLOT_START == 8
assert _GRAY_START == 9
assert D_GRAY_PAYLOAD == 16

# Match-gain for the readback's ``attend_most_recent_matching``.  Same
# value the prev-id attention uses at 16-wide (M4/Part 2 empirics).
_READBACK_MATCH_GAIN = 12000.0

# Encoder-side breakpoint grid.  All kinks of T_m for m ∈ {1, 2, 4, 8,
# 16, 32, 64, 128} lie on the j/256 grid, so the two piecewise_linear
# calls evaluate the basis exactly at their breakpoints.
_ENCODE_BP: List[float] = [j / 256.0 for j in range(257)]

# Sharpness for the 16 bit-extraction compares.  All triangle-wave
# crossings lie on the ``1/65536`` grid while integer ``k`` lands on
# ``1/65535``; the two grids differ by at most ``1/(65535·65536)``,
# which the triangle-wave slope ``2m`` amplifies to a minimum
# ``|feature - 0.5|`` of ``~7.6e-6`` across all 16 bits.  Compare
# saturates when ``|feature - thresh| * sharpness >= 1``, so we need
# ``sharpness >= 1/7.6e-6 ≈ 131072``.  ``2^18 = 262144`` gives 2×
# margin — clean powers-of-two keep all the internal weights exact in
# float32/TF32.
_BIT_COMPARE_SHARPNESS: float = 262144.0


# ---------------------------------------------------------------------------
# Triangle-wave basis
# ---------------------------------------------------------------------------


def _triangle(m: int, x: float) -> float:
    """Triangle wave with ``m`` full peaks in [0, 1], range [0, 1].

    Period ``1/m``. Peak at ``1/(2m)`` within each period (value 1);
    zeros at the period boundaries.
    """
    y = m * x
    y_frac = y - math.floor(y)
    return 1.0 - abs(2.0 * y_frac - 1.0)


# ---------------------------------------------------------------------------
# Emit helpers (producer side)
# ---------------------------------------------------------------------------


def encode_value_binary(q: Node, suffix: str = "") -> Node:
    """Convert a quantized integer float ``q ∈ [0, 65535]`` to its
    25-wide VALUE-row embedding.

    Emits the host-argmax target for a continuous value:

        [E8_VALUE (8) | raw = q/65535 (1) | Gray code of q (16 × ±1)]

    The Gray code is produced by comparing a set of triangle-wave
    features against 0.5.  Bit 0 is ``sign(x - 0.5)`` (the MSB of the
    binary-reflected Gray code, one flip over [0, 1]); bit 15 is
    ``sign(T_16384(x) - 0.5)`` (the LSB, 32768 flips).  Adjacent k
    differ in exactly one bit so the argmax against ``W_EMBED.T``
    resolves every emit to the nearest VALUE_k.

    Pipeline (per position):

    1. ``x = (2·clamp(q, 0, 65535) + 1) / 131072``  — pure affine.  The
       shifted grid lands integer k at ``(2k+1)/131072``, which is
       exact in float32 and never coincides with a triangle-wave
       crossing (``1/(4m)``) for any ``m ∈ {1, ..., 16384}``.  Under
       the unshifted ``k/65535`` mapping, ~65 specific k values
       rounded to exact 0.5 in float32, making compare ambiguous.
    2. L1 piecewise_linear on [0, 1] with 257 breakpoints at ``j/256``
       → 9-channel output ``[x, T_1, T_2, T_4, T_8, T_16, T_32,
       T_64, T_128](x)``.  One MLP sublayer.
    3. L2 piecewise_linear on ``y = T_128(x)`` with the same grid
       → 7-channel output ``[T_1, T_2, T_4, T_8, T_16, T_32, T_64](y)
       = [T_256, T_512, T_1024, T_2048, T_4096, T_8192, T_16384](x)``.
       One MLP sublayer.  (``T_m(T_128(x)) = T_(256m)(x)``: one period
       of T_128 in x makes y sweep [0, 1] twice, doubling the feature
       count before multiplying by 128 periods.)
    4. 16 parallel ``compare(feature_i, 0.5)`` calls.  One MLP sublayer
       (packed together by the compiler — total 32 neurons).
    5. Concatenate ``[E8_VALUE, x, b_0, ..., b_15]``.  Layout-only.

    Args:
        q: 1-wide float node holding a near-integer value in
            ``[0, 65535]``. Non-integer inputs produce soft bits in the
            ramp regions of the triangle waves; the host's argmax
            against ``W_EMBED.T`` picks the nearest VALUE_k (≤1 LSB
            error in the recovered float across the full range).
        suffix: Used to namespace the literal / intermediate nodes built
            inside.

    Returns:
        25-wide embedding node — the row the host argmaxes against
        ``W_EMBED.T`` to pick the next VALUE token ID.
    """
    q_clamped = clamp(q, 0.0, float(DEFAULT_N_LEVELS - 1))
    # Shifted grid: x = (2q + 1) / 131072. Both 2/131072 = 2^-16 and
    # 1/131072 = 2^-17 are exact in float32, and the result lands on
    # the 2^-17 subgrid that never coincides with any triangle-wave
    # crossing ((2j+1)/(4m)).
    x_raw = add_const(multiply_const(q_clamped, 2.0 / 131072.0), 1.0 / 131072.0)
    # Clamp to absorb residual numerical slop so the piecewise_linear
    # calls below see inputs strictly inside [0, 1].
    x = clamp(x_raw, 0.0, 1.0)

    # L1: 9 features → bits 0..8. Feature 0 is x itself; compare(x, 0.5)
    # gives the MSB of the 16-bit Gray code.
    l1 = piecewise_linear(
        x,
        _ENCODE_BP,
        lambda v: [
            v,
            _triangle(1, v),
            _triangle(2, v),
            _triangle(4, v),
            _triangle(8, v),
            _triangle(16, v),
            _triangle(32, v),
            _triangle(64, v),
            _triangle(128, v),
        ],
        name=f"encode_l1{suffix}",
    )

    # y = T_128(x) from L1 channel 8. Extracting is a pure Linear —
    # folds into L2's input projection at compile time.
    y = extract_from(l1, 9, 8, 1, f"encode_l2_in{suffix}")

    # L2: 7 features → bits 9..15.  ``T_m(T_128(x)) = T_(256m)(x)``, so
    # to cover bits 9..15 (= T_256..T_16384 in x) we evaluate
    # T_1..T_64 in y.
    l2 = piecewise_linear(
        y,
        _ENCODE_BP,
        lambda v: [
            _triangle(1, v),
            _triangle(2, v),
            _triangle(4, v),
            _triangle(8, v),
            _triangle(16, v),
            _triangle(32, v),
            _triangle(64, v),
        ],
        name=f"encode_l2{suffix}",
    )

    # L3: 16 parallel compare-to-0.5 calls.  Every compare is a 2-neuron
    # linear_relu_linear; the compiler packs them into one MLP sublayer.
    # Sharpness must saturate on ~7.6e-6 feature-to-threshold distance —
    # see _BIT_COMPARE_SHARPNESS.
    bits: List[Node] = []
    for i in range(9):
        feat = extract_from(l1, 9, i, 1, f"encode_l1_f{i}{suffix}")
        bits.append(compare(feat, 0.5, sharpness=_BIT_COMPARE_SHARPNESS))
    for i in range(7):
        feat = extract_from(l2, 7, i, 1, f"encode_l2_f{i}{suffix}")
        bits.append(compare(feat, 0.5, sharpness=_BIT_COMPARE_SHARPNESS))

    e8_cat = create_literal_value(E8_VALUE, name=f"e8_value_cat{suffix}")
    return Concatenate([e8_cat, x, *bits])


def emit_continuous_value_embedding(
    value: Node,
    name: str,
) -> Node:
    """Build a 25-wide VALUE embedding for a continuous float.

    Convenience wrapper that quantizes ``value`` into ``[0, 65535]``
    using ``VALUE_RANGE_BY_NAME[name]`` and then runs
    :func:`encode_value_binary`.
    """
    lo, hi = VALUE_RANGE_BY_NAME[name]
    q = quantize_to_range(value, lo, hi)
    return encode_value_binary(q, suffix=f"_{name}")


def emit_integer_value_embedding(
    integer_value: Node,
    max_int: int,
    name: str,
) -> Node:
    """Build a 25-wide VALUE embedding for an integer ``v ∈ [0, max_int]``.

    Used for identifiers whose value is a small-cardinality integer
    (``BSP_RANK`` 0..7, ``RESOLVED_ANGLE`` 0..255).  Quantize +
    triangle-wave encode would also work but we build a one-hot over
    the small-cardinality domain and let a single ``Linear`` index the
    ``W_EMBED`` rows ``[VALUE_0, ..., VALUE_{max_int}]`` directly.

    Depth: 2 MLP sublayers (1 for ``in_range``, 1 for the Linear row
    lookup — the ``bool_to_01`` is a free affine).

    Args:
        integer_value: 1-wide float node whose value is an integer in
            ``[0, max_int]``.
        max_int: Largest integer the value can take.  The one-hot
            width is ``max_int + 1``.
        name: Identifier name (for node debug naming).

    Returns:
        25-wide embedding node.
    """
    from torchwright.ops.arithmetic_ops import add_const

    n = max_int + 1
    onehot = bool_to_01(in_range(integer_value, add_const(integer_value, 1.0), n))

    rows = torch.stack(
        [embed_lookup(f"VALUE_{k}") for k in range(n)], dim=0
    )  # (n, D_EMBED)
    return Linear(
        onehot,
        rows,
        torch.zeros(D_EMBED),
        name=f"emit_int_{name}",
    )


def emit_boolean_value_embedding(
    bool_value: Node,
    name: str,
) -> Node:
    """Build a 25-wide VALUE embedding for a ±1 boolean.

    +1 → ``W_EMBED[VALUE_1]``, -1 → ``W_EMBED[VALUE_0]``.

    Depth: 1 MLP sublayer (the ``cond_gate`` on the 25-wide delta).

    Args:
        bool_value: 1-wide node with value in ``{-1, +1}``.
        name: Identifier name (for node debug naming).

    Returns:
        25-wide embedding node.
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
        # The 1-wide constant query vector (``+1``) used by
        # ``attend_most_recent_matching``.  Single shared node so every
        # identifier's attention head shares a literal.
        self._query_const_1 = create_literal_value(
            torch.tensor([1.0]), name="readback_q1"
        )

    def get_value_after_last(
        self, name: str, *, assert_hardness_gt: float | None = 0.99
    ) -> Node:
        """Return the dequantized float for the most recent matching VALUE.

        ``name`` must be one of the 20 entries in ``IDENTIFIER_NAMES``.
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
        if name not in VALUE_RANGE_BY_NAME:
            raise KeyError(
                f"unknown identifier {name!r}; must be one of {IDENTIFIER_NAMES}"
            )

        slot = IDENTIFIER_NAMES.index(name)

        # 1. Build is_X_value: ``is_value_category AND prev_slot_onehot[slot]``.
        #    The slot one-hot stored by thinking_wall is already gated
        #    to live only at identifier positions, so at a VALUE
        #    position the prev_slot_onehot reads the *previous* id's
        #    slot.  The AND with is_value_category restricts the key
        #    signal to VALUE positions only (keys at identifier
        #    positions would otherwise confuse the attention).
        prev_slot_i_01 = extract_from(
            self._ctx.prev_id_slots,
            len(IDENTIFIER_NAMES),
            slot,
            1,
            f"readback_prev_slot_{name}",
        )
        prev_slot_i_bool = compare(prev_slot_i_01, 0.5)
        is_X_value = bool_all_true([self._ctx.is_value_category, prev_slot_i_bool])

        # 2. Attention value: the 1-wide raw slot at col D_CATEGORY of
        #    the embedding.  The Gray-code bits in cols [9:25] and the
        #    E8 category in cols [0:8] carry no extra information for
        #    decode — the raw slot alone is the normalized value.  The
        #    narrower V head (1 col instead of 16) cuts W_V/W_O weight
        #    count ~16× per readback head.
        raw_slot = extract_from(
            self._ctx.embedding,
            D_EMBED,
            _RAW_SLOT_START,
            D_RAW_SLOT,
            f"readback_raw_{name}",
        )

        matched_raw = attend_most_recent_matching(
            pos_encoding=self._ctx.pos_encoding,
            query_vector=self._query_const_1,
            key_vector=is_X_value,
            value=raw_slot,
            match_gain=_READBACK_MATCH_GAIN,
            assert_hardness_gt=assert_hardness_gt,
        )

        # 3. Decode the normalized raw slot back to the float value via
        #    a single scalar affine.
        float_value = _decode_payload_to_float(matched_raw, name)
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
        embedding: 25-wide ``W_EMBED`` leaf (per-position embedding).
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
    """Decode a 1-wide shifted raw slot to a float.

    The raw slot stores ``(2k + 1) / 131072`` for VALUE_k — the same
    shifted value the encoder produces.  Dequantize: first recover
    ``k / 65535 ≈ raw * (65536 / 65535) - 1 / (2 * 65535)``, then map
    to ``[lo, hi]``:

        value = lo + (raw * 65536 - 0.5) * (hi - lo) / 65535

    A single scalar affine — weights and bias collapse into the
    Linear.  The ``- 0.5 / 65535`` constant compensates for the
    half-LSB shift built into the encoder's ``(2k + 1) / 131072``
    grid; without it the decoded value would sit half an LSB above
    the true ``lo + k * LSB`` for every k.
    """
    lo, hi = VALUE_RANGE_BY_NAME[name]
    lsb = (hi - lo) / (DEFAULT_N_LEVELS - 1)
    # raw * 65536 - 0.5 ≈ k, then scale by LSB and add lo.
    weight = 65536.0 * lsb
    bias_val = lo - 0.5 * lsb
    weights = torch.tensor([[weight]])
    bias = torch.tensor([bias_val])

    decoded = Linear(payload, weights, bias, name=f"readback_decode_{name}")
    # Declare the float's value_range so downstream ops (T_LO / VIS
    # clip-and-project) get tight value_type bounds.  Without this the
    # compiler treats the decoded float as unbounded, which inflates
    # every downstream clamp / cond_gate M offset — test_affine_bounds
    # fires on the resulting blow-up.
    return assert_in_range(decoded, lo, hi)
