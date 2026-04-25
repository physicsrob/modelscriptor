"""Codec between thinking-token values and 27-wide VALUE embeddings.

Two sides of the same boundary, with separate wire formats for
continuous floats and small-cardinality integers:

* **Emit** (producer): a per-wall identifier step computes a value and
  one of :func:`emit_continuous_value_embedding`,
  :func:`emit_integer_value_embedding`,
  :func:`emit_boolean_value_embedding` produces the 27-wide row the
  host argmaxes against ``W_EMBED.T`` to pick the next VALUE token ID.
* **Readback** (consumer): a later thinking step needs a prior value
  from the KV cache and :class:`ThinkingReadback` runs a single
  ``attend_most_recent_matching`` to fetch it.  For continuous
  identifiers (CROSS_A, T_LO, …), a Linear decode dequantizes the raw
  slot back to a float.  For integer identifiers in
  :data:`INT_IDENTIFIER_NAMES` (today: ``SORT_RESULT``), the K column
  carries the integer directly — no decode Linear.

Wire formats
------------

Continuous (Phase A/B): the 16-bit integer VALUE ID.  Every VALUE row
``k`` in ``W_EMBED`` shares the same 8-wide E8 category code in cols
[0:8], carries ``(2k+1) / 131072`` in col 8 (the raw slot — the
shifted encoder grid), and a 16-wide ±1 Gray code of ``k`` in cols
[9:25].  Float ↔ VALUE-ID mapping per identifier name is
``VALUE_RANGE_BY_NAME[name]``:

    q = (value - lo) * (N_VALUES - 1) / (hi - lo)          # produce side
    value = lo + q * (hi - lo) / (N_VALUES - 1)             # consume side

where ``N_VALUES = 65536``.  Host-side uint16 rounding happens between
the two and contributes one LSB per quantization boundary.

Integer (Phase C Part 2): cols [25:27] of every VALUE row carry
``[K = k, K_NS = -k²]`` for ``k ≤ MAX_INT_K`` (zero elsewhere).
:func:`emit_integer_value_embedding` and
:func:`emit_boolean_value_embedding` *override* these two columns
with ``[K = 2·k_target, K_NS = 1]`` in the predicted embedding,
giving an argmax score contribution of
``2·k_target·k − k²`` that peaks exactly at ``k = k_target`` (margin
1 to adjacent k, plus the existing gray-code margin of 2).
:meth:`ThinkingReadback.get_int_after_last` reads the K column
directly via attention — the matched value IS the integer, no
dequantize affine, no W_consumer amplification.  See
``docs/phase_c_part2_int_slot_embedding.md`` for the full design and
the integer-emit round-trip bug it fixes.

Depth accounting for the emit path (continuous value):

    clamp + scale   0 layers (pure affine, fused)
    L1 triangle PL  1 MLP sublayer (9-channel output)
    L2 triangle PL  1 MLP sublayer (7-channel output, off T_128 of L1)
    compare × 16    1 MLP sublayer (parallel)
    K, K_NS append  0 layers (literal zeros, fused)

≈3 MLP sublayers after the value is computed.  Integer / boolean
emits are ~2 layers (one-hot ``in_range`` + Linear row lookup; the
K/K_NS override is ``multiply_const`` + literal, both fold).

Readback depth:

    Continuous (raw + decode): 1 attention sublayer; the decode Linear
        folds into the next consumer.
    Integer (K column): 1 attention sublayer; no decode.
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
    D_IS_ANY_ID,
    D_IS_VALUE_CATEGORY,
    D_K_NS_SLOT,
    D_K_SLOT,
    D_RAW_SLOT,
    D_SLOT_ONEHOT,
    E8_VALUE,
    IDENTIFIER_NAMES,
    MAX_INT_K,
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
    "INT_IDENTIFIER_NAMES",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Cols [0:D_CATEGORY] are the E8_VALUE category code shared across all
# VALUE rows.  The raw slot (col D_CATEGORY) carries the normalized
# value; the 16-wide Gray-code payload sits immediately after it.
# Phase C Part 2: cols [25:26] hold K = k for VALUE_k with k ≤
# MAX_INT_K (else 0); cols [26:27] hold K_NS = -k² (else 0).
_RAW_SLOT_START = D_CATEGORY
_GRAY_START = D_CATEGORY + D_RAW_SLOT
_K_SLOT_START = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD
_K_NS_SLOT_START = _K_SLOT_START + D_K_SLOT
assert _RAW_SLOT_START == 8
assert _GRAY_START == 9
assert _K_SLOT_START == 25
assert _K_NS_SLOT_START == 26
assert D_GRAY_PAYLOAD == 16

# Identifier names that should round-trip through the K column rather
# than the raw + decode path.  Today this is only SORT_RESULT — its
# producer (``emit_integer_value_embedding``) writes the K/K_NS
# override and its consumer (RENDER's wall_index readback, SORTED's
# local wall_index decode) reads K directly.  Other small-int names
# (BSP_RANK, IS_RENDERABLE, HIT_*) currently round-trip correctly
# through the continuous emit path (encode_value_binary +
# raw-slot + decode) — they could migrate later if a hot-path
# consumer pays the decode cost, but they're not broken today.
INT_IDENTIFIER_NAMES: set[str] = {"SORT_RESULT"}

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
    # Phase C Part 2: continuous emits don't activate the K/K_NS
    # mechanism (their target k > MAX_INT_K, so the W_EMBED row's K
    # and K_NS columns are 0).  Predicted K=0, K_NS=0 contributes 0
    # to argmax — same behavior as before D_EMBED grew by 2.
    k_zero = create_literal_value(torch.tensor([0.0]), name=f"encode_K_zero{suffix}")
    k_ns_zero = create_literal_value(
        torch.tensor([0.0]), name=f"encode_K_NS_zero{suffix}"
    )
    # Phase D Part 1: type-tag block matches the VALUE row pattern in
    # W_EMBED — slot one-hot all −1, is_any_identifier −1,
    # is_value_category +1.  Argmax against W_EMBED.T still picks the
    # closest VALUE row; the type-tag dot is constant across the VALUE
    # block (slot one-hot self-dot 21 + is_any_id self-dot 1 +
    # is_value_cat self-dot 1 = 23), so the existing E8/raw/gray/K/K_NS
    # argmax shape is preserved.
    slot_onehot_neg = create_literal_value(
        -torch.ones(D_SLOT_ONEHOT), name=f"encode_slot_onehot{suffix}"
    )
    is_any_id_neg = create_literal_value(
        -torch.ones(D_IS_ANY_ID), name=f"encode_is_any_id{suffix}"
    )
    is_value_cat_pos = create_literal_value(
        torch.ones(D_IS_VALUE_CATEGORY), name=f"encode_is_value_cat{suffix}"
    )
    return Concatenate(
        [
            e8_cat,
            x,
            *bits,
            k_zero,
            k_ns_zero,
            slot_onehot_neg,
            is_any_id_neg,
            is_value_cat_pos,
        ]
    )


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
    """Build a ``D_EMBED``-wide VALUE embedding for an integer ``v ∈ [0, max_int]``.

    Used for identifiers whose value is a small-cardinality integer
    (today: SORT_RESULT carrying wall_index 0..max_walls-1).  ``max_int``
    must be ≤ ``MAX_INT_K`` so the K/K_NS columns are populated for
    every reachable VALUE row.

    Phase C Part 2 design: the predicted embedding's E8/raw/gray columns
    come from a one-hot lookup over ``W_EMBED[VALUE_0..VALUE_{max_int}]``
    (preserves the existing argmax discrimination via gray Hamming
    distance).  The K and K_NS columns are *overridden* with the
    quadratic-equality pattern ``[2·integer_value, 1]``: for any target
    ``k_target``, the K/K_NS contribution to score(VALUE_k) is
    ``2·k_target·k − k²``, peaking at ``k = k_target`` with margin 1
    to adjacent k.  Total argmax margin (gray + K/K_NS) is 3 — same
    shape as the prior 25-wide design but with K/K_NS as the primary
    integer discriminator.

    The override fixes a latent round-trip bug in the prior design: a
    pure row-lookup put ``raw = (2·k_target+1)/131072`` in the predicted
    embedding, which the consumer's ``_decode_payload_to_float``
    (calibrated for the *continuous* emit's quantization) decoded to
    ``≈ k_target / 9362``, not ``k_target``.  See
    ``docs/phase_c_part2_int_slot_embedding.md``.

    Depth: 2 MLP sublayers (``in_range`` builds the one-hot; the row
    lookup is a single Linear; ``multiply_const`` and ``Concatenate``
    are layout/Linear and fold).

    Args:
        integer_value: 1-wide float node whose value is an integer in
            ``[0, max_int]``.
        max_int: Largest integer the value can take.  Must be ≤
            ``MAX_INT_K``; the one-hot width is ``max_int + 1``.
        name: Identifier name (for node debug naming).

    Returns:
        ``D_EMBED``-wide embedding node.
    """
    from torchwright.ops.arithmetic_ops import add_const, multiply_const

    assert max_int <= MAX_INT_K, (
        f"emit_integer_value_embedding: max_int={max_int} exceeds "
        f"MAX_INT_K={MAX_INT_K}; the K/K_NS argmax mechanism only "
        f"covers VALUE rows up to MAX_INT_K.  Either widen MAX_INT_K "
        f"in embedding.py (mind the float32 precision tradeoff) or "
        f"emit via the continuous path."
    )

    n = max_int + 1
    onehot = bool_to_01(in_range(integer_value, add_const(integer_value, 1.0), n))

    # Build E8/raw/gray columns from VALUE_0..VALUE_max_int rows.  Slice
    # off the K and K_NS columns from each row — those will be overridden
    # below with the quadratic-equality pattern.
    base_cols = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD  # 25
    rows_base = torch.stack(
        [embed_lookup(f"VALUE_{k}")[:base_cols] for k in range(n)], dim=0
    )  # (n, 25)
    base = Linear(
        onehot,
        rows_base,
        torch.zeros(base_cols),
        name=f"emit_int_base_{name}",
    )

    # K column = 2·integer_value, K_NS column = 1.  The compiler folds
    # the multiply_const + literal into the next consumer.
    two_k = multiply_const(integer_value, 2.0)
    one_lit = create_literal_value(
        torch.tensor([1.0]), name=f"emit_int_kns_{name}"
    )
    # Phase D Part 1: type-tag block matches the VALUE row pattern in
    # W_EMBED — slot one-hot all −1, is_any_identifier −1,
    # is_value_category +1.  These literals fold into the next
    # consumer.
    slot_onehot_neg = create_literal_value(
        -torch.ones(D_SLOT_ONEHOT), name=f"emit_int_slot_onehot_{name}"
    )
    is_any_id_neg = create_literal_value(
        -torch.ones(D_IS_ANY_ID), name=f"emit_int_is_any_id_{name}"
    )
    is_value_cat_pos = create_literal_value(
        torch.ones(D_IS_VALUE_CATEGORY), name=f"emit_int_is_value_cat_{name}"
    )
    return Concatenate(
        [base, two_k, one_lit, slot_onehot_neg, is_any_id_neg, is_value_cat_pos]
    )


def emit_boolean_value_embedding(
    bool_value: Node,
    name: str,
) -> Node:
    """Build a ``D_EMBED``-wide VALUE embedding for a ±1 boolean.

    +1 → predicted ``VALUE_1``, -1 → predicted ``VALUE_0``.

    Phase C Part 2: the E8/raw/gray columns blend ``W_EMBED[VALUE_0]``
    and ``W_EMBED[VALUE_1]``'s first 25 cols (cond_gate on the ±1
    bool — same pattern thinking_wall.py uses for HIT_* emission).
    The K and K_NS columns are overridden with the quadratic-equality
    pattern: ``K = bool_value + 1`` (= 2 for true, 0 for false, =
    ``2·k_target``), ``K_NS = 1``.

    Depth: 1 MLP sublayer (the ``cond_gate`` on the 25-wide delta);
    ``add_const`` and the literal fold into the next consumer.

    Args:
        bool_value: 1-wide node with value in ``{-1, +1}``.
        name: Identifier name (for node debug naming).

    Returns:
        ``D_EMBED``-wide embedding node.
    """
    base_cols = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD  # 25
    e_value_0 = create_literal_value(
        embed_lookup("VALUE_0")[:base_cols], name=f"e_v0_{name}"
    )
    e_value_1 = create_literal_value(
        embed_lookup("VALUE_1")[:base_cols], name=f"e_v1_{name}"
    )
    from torchwright.ops.arithmetic_ops import add_const, sum_nodes
    from torchwright.ops.logic_ops import cond_gate

    delta = subtract(e_value_1, e_value_0)
    base = sum_nodes([e_value_0, cond_gate(bool_value, delta)])

    # K = 2·k_target.  bool_value ∈ {-1, +1}; k_target = (bool+1)/2 ∈
    # {0, 1}; 2·k_target = bool + 1.
    two_k = add_const(bool_value, 1.0)
    one_lit = create_literal_value(
        torch.tensor([1.0]), name=f"emit_bool_kns_{name}"
    )
    # Phase D Part 1: type-tag block matches the VALUE row pattern in
    # W_EMBED — slot one-hot all −1, is_any_identifier −1,
    # is_value_category +1.
    slot_onehot_neg = create_literal_value(
        -torch.ones(D_SLOT_ONEHOT), name=f"emit_bool_slot_onehot_{name}"
    )
    is_any_id_neg = create_literal_value(
        -torch.ones(D_IS_ANY_ID), name=f"emit_bool_is_any_id_{name}"
    )
    is_value_cat_pos = create_literal_value(
        torch.ones(D_IS_VALUE_CATEGORY), name=f"emit_bool_is_value_cat_{name}"
    )
    return Concatenate(
        [base, two_k, one_lit, slot_onehot_neg, is_any_id_neg, is_value_cat_pos]
    )


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
        # Phase D Part 1: prev_id_slots' V is the ±1 slot one-hot
        # column block from W_EMBED, so the extract is already the
        # ±1 bool that bool_all_true expects — no compare(0.5) needed.
        prev_slot_i_bool = extract_from(
            self._ctx.prev_id_slots,
            len(IDENTIFIER_NAMES),
            slot,
            1,
            f"readback_prev_slot_{name}",
        )
        indicator = bool_all_true([self._ctx.is_value_category, prev_slot_i_bool])
        self._indicator_cache[name] = indicator
        return indicator

    def get_value_after_last(
        self, name: str, *, assert_hardness_gt: float | None = 0.99
    ) -> Node:
        """Return the value carried by the most recent matching VALUE.

        ``name`` must be one of the 21 entries in ``IDENTIFIER_NAMES``.
        The returned Node is 1-wide with value range
        ``VALUE_RANGE_BY_NAME[name]``.

        Phase C Part 2: this method is now a dispatcher.  Identifiers
        in :data:`INT_IDENTIFIER_NAMES` (today: ``SORT_RESULT``) route
        to :meth:`get_int_after_last`, which reads the K column
        directly — no decode Linear, output is the integer scalar
        with O(ε) noise.  All other identifiers continue to use the
        raw + decode path.  Callers don't change.

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
        if name in INT_IDENTIFIER_NAMES:
            return self.get_int_after_last(
                name, assert_hardness_gt=assert_hardness_gt
            )

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

    def get_int_after_last(
        self, name: str, *, assert_hardness_gt: float | None = 0.99
    ) -> Node:
        """Return the integer carried by the K column of the most recent
        matching VALUE — no decode Linear, no W_consumer amplification.

        ``name`` must be in :data:`INT_IDENTIFIER_NAMES`.  The returned
        Node is 1-wide; its value is the integer ``k`` of the matched
        VALUE_k token (within softmax-leakage noise ε·MAX_INT_K).
        Cached per-name (results shared with :meth:`get_value_after_last`
        when the dispatcher routes here).

        Mechanism: attends the K column of the embedding (col 25)
        with V = K_slot.  At VALUE_k positions the K column carries
        ``k`` itself for ``k ≤ MAX_INT_K``; at non-VALUE / non-matching
        positions the K column is 0 (either by W_EMBED layout or by
        the ``is_X_value`` softmax gate).  The matched K value IS the
        integer; no dequantize affine is needed.

        See ``docs/phase_c_part2_int_slot_embedding.md`` for the full
        design.
        """
        cache_key = f"__int__:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if name not in INT_IDENTIFIER_NAMES:
            raise KeyError(
                f"get_int_after_last: {name!r} is not an int identifier; "
                f"INT_IDENTIFIER_NAMES = {sorted(INT_IDENTIFIER_NAMES)}.  "
                f"Use get_value_after_last for continuous identifiers."
            )

        is_X_value = self.is_value_of(name)
        k_slot = extract_from(
            self._ctx.embedding,
            D_EMBED,
            _K_SLOT_START,
            D_K_SLOT,
            f"readback_k_{name}",
        )
        matched_k = attend_most_recent_matching(
            pos_encoding=self._ctx.pos_encoding,
            query_vector=self._query_const_1,
            key_vector=is_X_value,
            value=k_slot,
            match_gain=_READBACK_MATCH_GAIN,
            assert_hardness_gt=assert_hardness_gt,
        )
        # Declare the integer's value range so downstream ops get tight
        # bounds.  For SORT_RESULT, range is (0, max_walls-1) — but the
        # K column is populated for every k ≤ MAX_INT_K, so the
        # value_range here is the K column's full domain at this
        # identifier's emit positions.
        lo, hi = VALUE_RANGE_BY_NAME[name]
        result = assert_in_range(matched_k, lo, hi)
        self._cache[cache_key] = result
        return result


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
