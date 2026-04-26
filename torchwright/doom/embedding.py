"""DOOM's token vocabulary and embedding table.

Every autoregressive step consumes and emits a single token ID.  The
host feeds ``token_ids`` as a 1-wide input slot; the graph looks the
ID up in ``W_EMBED`` to produce a 49-wide residual leaf.  On the
output side, the 49-wide output slice is projected through
``W_EMBED.T`` and argmaxed to pick the next ID.

Vocabulary layout:

  |      0 .. 65535 | VALUE (quantized 16-bit integers)            |
  |  65536 .. 65543 | THINKING_WALL markers 0..7                   |
  |  65544 .. 65560 | Per-wall identifiers (17: BSP_RANK..HIT_Y)   |
  |  65561 .. 65563 | RESOLVED identifiers (X / Y / ANGLE)         |
  |           65564 | SORT_RESULT identifier                       |
  |  65565 .. 65567 | Decode tokens (SORTED_WALL, RENDER, DONE)    |
  |  65568 .. 65575 | Prompt-position categories                   |

Total ``V = 65576``.

Embedding layout (``d_embed = 49``):

  cols [ 0 :  8] — E8 category code (distinct per category name)
  cols [ 8 :  9] — raw slot: (2k+1)/131072 for VALUE_k, else 0
  cols [ 9 : 25] — 16-wide ±1 Gray code of k for VALUE_k, else 0
  cols [25 : 26] — K slot:    k          for VALUE_k where k ≤ MAX_INT_K, else 0
  cols [26 : 47] — slot one-hot: ±1 per IDENTIFIER_NAMES entry; +1 at
                   the row's slot column for an IDENTIFIER row, −1
                   elsewhere (including non-identifier rows). Stored
                   as ±1 (not {0, 1}) so a per-position direct extract
                   IS the ±1 bool that ``cond_gate`` and
                   ``bool_all_true`` consumers expect. The {0, 1}
                   semantics the original cond_gate-built V provided
                   are recovered downstream by adjusting Linear
                   weights/bias at the one ``Linear(prev_slot_onehot,
                   next_after_slot, ...)`` consumer.
  cols [47 : 48] — is_any_identifier: +1 for the 21 IDENTIFIER_NAMES
                   rows, −1 for everything else
  cols [48 : 49] — is_value_category: +1 for the 65,536 VALUE rows,
                   −1 elsewhere

The raw slot and Gray-code columns are zero for non-VALUE rows. All
65,536 VALUE rows share a single ``E8_VALUE`` category code; they are
distinguished by the raw slot (dense, gives a monotone cue) and the
Gray-code payload (±1 per bit, Hamming 1 for adjacent k).

The type-tag block at cols [26:49] lets the readback chain skip the
equals_vector + cond_gate + Concat construction that would otherwise
build these flags at depth 4-5.  Storing them as ±1 columns in
W_EMBED makes them direct extracts at depth 1.

The K column carries small-cardinality integer identifiers in a form
that's directly readable by attention with no decode Linear: for
VALUE_k with k ≤ ``MAX_INT_K``, K stores k literally.  A consumer
does ``attend(V = K_column)`` to recover the integer directly, with
no decode Linear and no W_consumer amplification.  Producers
(``emit_integer_value_embedding``, ``emit_boolean_value_embedding``)
leave the K column at 0 in the predicted embedding — gray's
Hamming-1 margin already drives the host argmax to the right VALUE_k
row, and any non-zero predicted K would bias argmax monotonically
toward larger k.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from torchwright.graph.embedding import Embedding
from torchwright.graph.spherical_codes import index_to_vector

D_CATEGORY: int = 8
D_RAW_SLOT: int = 1
D_GRAY_PAYLOAD: int = 16
D_K_SLOT: int = 1
# Type-tag block.  D_SLOT_ONEHOT = len(IDENTIFIER_NAMES); the assertion
# below pins it once IDENTIFIER_NAMES is defined.
D_SLOT_ONEHOT: int = 21
D_IS_ANY_ID: int = 1
D_IS_VALUE_CATEGORY: int = 1
D_EMBED: int = (
    D_CATEGORY
    + D_RAW_SLOT
    + D_GRAY_PAYLOAD
    + D_K_SLOT
    + D_SLOT_ONEHOT
    + D_IS_ANY_ID
    + D_IS_VALUE_CATEGORY
)
assert D_EMBED == 49

# Column offsets for the type-tag block.  Imported by callers that
# need to slice the W_EMBED row directly (``_detect_token_types`` in
# game_graph, ``find_prev_identifier`` in thinking_wall,
# ``is_value_of`` in thinking_readback).
_SLOT_ONEHOT_START: int = (
    D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD + D_K_SLOT
)  # 26
_IS_ANY_ID_COL: int = _SLOT_ONEHOT_START + D_SLOT_ONEHOT  # 47
_IS_VALUE_CATEGORY_COL: int = _IS_ANY_ID_COL + D_IS_ANY_ID  # 48
assert _SLOT_ONEHOT_START == 26
assert _IS_ANY_ID_COL == 47
assert _IS_VALUE_CATEGORY_COL == 48

N_VALUES: int = 65536  # 2**16 VALUE IDs

# K column is populated only for VALUE_k with k ≤ MAX_INT_K.  The cap
# matters only for what an in-graph attention can read back as a
# literal integer (a square op at the consumer would extend the range
# — there's no float32 precision wall on K alone).
MAX_INT_K: int = 255


# ---------------------------------------------------------------------------
# Category → E8 lattice index
#
# Each distinct category name gets a 8-wide E8 code in cols [0:8].  For
# categories that already have an E8 lattice index in graph_constants.py
# we keep the same index (so the 8-wide category code matches the
# historical ``E8_X`` constants).  Fresh indices for the categories
# that didn't exist under M4 (DONE, BSP_RANK, etc.) pick from the
# unused 270..283 range.  Any two indices 0..1023 give distinct E8
# codes; the exact picks are arbitrary as long as they're unique.
# ---------------------------------------------------------------------------
_CATEGORY_INDEX: Dict[str, int] = {
    # VALUE: shared category code across all 65,536 VALUE IDs.
    "VALUE": 261,
    # THINKING_WALL markers: one category code per marker.
    **{f"THINKING_WALL_{i}": 250 + i for i in range(8)},
    # Per-wall identifiers (17).  HIT_FULL/X/Y keep their M4 indices.
    "BSP_RANK": 270,
    "IS_RENDERABLE": 271,
    "CROSS_A": 272,
    "DOT_A": 273,
    "CROSS_B": 274,
    "DOT_B": 275,
    "T_STAR_L": 284,
    "T_STAR_R": 285,
    "T_LO": 276,
    "T_HI": 277,
    "COL_A": 286,
    "COL_B": 287,
    "VIS_LO": 278,
    "VIS_HI": 279,
    "HIT_FULL": 258,
    "HIT_X": 259,
    "HIT_Y": 260,
    # RESOLVED identifiers (3).
    "RESOLVED_X": 280,
    "RESOLVED_Y": 281,
    "RESOLVED_ANGLE": 282,
    # Sort-phase identifier (1).
    "SORT_RESULT": 288,
    # Decode tokens (3).
    "SORTED_WALL": 3,
    "RENDER": 4,
    "DONE": 283,
    # Prompt-position categories (8).
    "INPUT": 0,
    "BSP_NODE": 7,
    "WALL": 1,
    "EOS": 2,
    "TEX_COL": 5,
    "PLAYER_X": 240,
    "PLAYER_Y": 241,
    "PLAYER_ANGLE": 242,
}


def _category_code(name: str) -> torch.Tensor:
    """Return the 8-wide E8 category code for a category name."""
    return index_to_vector(_CATEGORY_INDEX[name])


# ---------------------------------------------------------------------------
# Vocabulary ID allocation.
# ---------------------------------------------------------------------------

_VALUE_ID_BASE = 0  # 0 .. 65535
_THINKING_WALL_BASE = N_VALUES  # 65536 .. 65543

_PER_WALL_IDENTIFIERS: List[str] = [
    "BSP_RANK",
    "IS_RENDERABLE",
    "CROSS_A",
    "DOT_A",
    "CROSS_B",
    "DOT_B",
    "T_STAR_L",
    "T_STAR_R",
    "T_LO",
    "T_HI",
    "COL_A",
    "COL_B",
    "VIS_LO",
    "VIS_HI",
    "HIT_FULL",
    "HIT_X",
    "HIT_Y",
]
_PER_WALL_BASE = _THINKING_WALL_BASE + 8  # 65544 .. 65560

_RESOLVED_IDENTIFIERS: List[str] = ["RESOLVED_X", "RESOLVED_Y", "RESOLVED_ANGLE"]
_RESOLVED_BASE = _PER_WALL_BASE + len(_PER_WALL_IDENTIFIERS)  # 65561 .. 65563

# Sort-phase identifier: ``SORT_RESULT`` tags the VALUE token that
# emits the picked wall_index after each ``SORTED_WALL`` marker.
# Structurally it sits alongside the per-wall + RESOLVED identifiers
# — same prev-id / readback / factor-to-embedding machinery — but it
# lives at SORTED-phase positions rather than inside the thinking
# cascade.
_SORT_IDENTIFIERS: List[str] = ["SORT_RESULT"]
_SORT_BASE = _RESOLVED_BASE + len(_RESOLVED_IDENTIFIERS)  # 65564

_DECODE_TOKENS: List[str] = ["SORTED_WALL", "RENDER", "DONE"]
_DECODE_BASE = _SORT_BASE + len(_SORT_IDENTIFIERS)  # 65565 .. 65567

_PROMPT_TOKENS: List[str] = [
    "INPUT",
    "BSP_NODE",
    "WALL",
    "EOS",
    "TEX_COL",
    "PLAYER_X",
    "PLAYER_Y",
    "PLAYER_ANGLE",
]
_PROMPT_BASE = _DECODE_BASE + len(_DECODE_TOKENS)  # 65568 .. 65575

V: int = _PROMPT_BASE + len(_PROMPT_TOKENS)  # 65576
assert V == 65576, f"Vocabulary size mismatch: got {V}, expected 65576"


def _build_id_map() -> Dict[str, int]:
    ids: Dict[str, int] = {}
    for i in range(N_VALUES):
        ids[f"VALUE_{i}"] = _VALUE_ID_BASE + i
    for i in range(8):
        ids[f"THINKING_WALL_{i}"] = _THINKING_WALL_BASE + i
    for i, name in enumerate(_PER_WALL_IDENTIFIERS):
        ids[name] = _PER_WALL_BASE + i
    for i, name in enumerate(_RESOLVED_IDENTIFIERS):
        ids[name] = _RESOLVED_BASE + i
    for i, name in enumerate(_SORT_IDENTIFIERS):
        ids[name] = _SORT_BASE + i
    for i, name in enumerate(_DECODE_TOKENS):
        ids[name] = _DECODE_BASE + i
    for i, name in enumerate(_PROMPT_TOKENS):
        ids[name] = _PROMPT_BASE + i
    return ids


_VOCAB_IDS: Dict[str, int] = _build_id_map()


def _build_vocab_list() -> List[str]:
    """Ordered list of vocab-entry strings (length V).

    ``DOOM_VOCAB[id]`` returns the name for vocab ID ``id``.  VALUE
    entries are named ``VALUE_<int>`` for introspection; everything
    else is named by its identifier / marker / category.
    """
    vocab: List[str] = [""] * V
    for name, vid in _VOCAB_IDS.items():
        vocab[vid] = name
    # Every slot must be filled.
    assert all(v != "" for v in vocab), "Vocab has a hole"
    return vocab


DOOM_VOCAB: List[str] = _build_vocab_list()


# ---------------------------------------------------------------------------
# Gray-code payload helper
#
# The VALUE_k row stores the 16-wide ±1 bit pattern the emit-side
# triangle-wave encoder produces at x = k/65535.  That pattern is a
# Gray-like code — every 65,536 patterns are unique and every adjacent
# ``k, k+1`` pair differs in exactly one bit — but it is *not* the
# canonical reflected-binary Gray code (``k XOR (k >> 1)``).  The
# flip points of ``compare(T_m(x), 0.5)`` sit on a ``1/(4m)`` grid in
# x, while reflected-binary Gray bits flip on a uniform integer grid
# in k; the two grids don't align at integer k values, so the two
# codes differ.
#
# Adjacent k differ in exactly one bit → adjacent rows have Hamming
# distance 1 → Gray-payload dot 14 instead of 16.  Together with the
# raw slot (at most 1 self-dot, ≤1 for cross) and the shared 8-wide
# E8_VALUE prefix (self-dot 1600), this gives a margin of ≥ 1.75
# between any VALUE_k self-dot and any other VALUE row's cross-dot
# (worst case at raw ≈ 0.5).  Host-side argmax against ``W_EMBED.T``
# resolves every VALUE emit to the nearest k.
# ---------------------------------------------------------------------------


# The triangle-wave encoder evaluates on a shifted grid ``x_k = (2k + 1)
# / 131072`` instead of ``k / 65535`` for two reasons:
#
# 1. All the arithmetic is exact in float32 — ``2k + 1`` and ``131072 =
#    2^17`` have small powers of two as factors, and ``m * x_k`` always
#    lands on a ``2^-17`` grid with no rounding.
# 2. Integer ``k`` never lands on a triangle-wave crossing.  For every
#    m ∈ {1, 2, 4, ..., 16384}, the minimum ``|T_m(x_k) - 0.5|`` across
#    k = 0..65535 is ``2^(e - 16)`` where ``m = 2^e`` — ranging from
#    ``1.5e-5`` (bit 1) up to ``0.25`` (bit 15).  No float32 rounding
#    puts the feature on the 0.5 line.
#
# Using ``k / 65535`` as input caused ~65 specific k values to land
# exactly on 0.5 in float32 (T_16384 rounded to 0.5), making compare
# ambiguous at those k.
_X_SHIFT_SCALE: float = 1.0 / 131072.0
_X_SHIFT_BIAS: float = 1.0 / 131072.0


def _shifted_x(k: int) -> float:
    """Return the float32 value of ``(2k + 1) / 131072`` used by the encoder."""
    return (2 * k + 1) * _X_SHIFT_SCALE


def gray_code_16(k: int) -> torch.Tensor:
    """Return the 16-wide ±1 bit pattern the triangle-wave encoder
    produces at the shifted input ``x = (2k + 1) / 131072``.

    Bit 0 is ``sign(x - 0.5)`` (one flip across the range — bit 0
    changes from -1 to +1 at k = 32767 → 32768).  Bit ``i`` for
    ``i ≥ 1`` is ``sign(T_{2^(i-1)}(x) - 0.5)``, where ``T_m(x) =
    1 - |2·frac(m·x) - 1|`` is the triangle wave with ``m`` peaks in
    ``[0, 1]``.  Bit 15 (T_16384) flips 32768 times across k = 0..65535.

    This is a valid Gray-like code (65,536 unique patterns, adjacent-k
    Hamming distance 1) but not the canonical reflected-binary Gray
    code — the triangle-wave crossings sit on a ``1/(4m)`` grid in x
    while reflected-binary Gray flips on a uniform integer grid in k,
    and the two grids don't align at integer k values.
    """
    assert 0 <= k < N_VALUES, f"gray_code_16 out of range: {k}"
    # Compute in float32 so the result bit-exactly matches what the
    # compiled encoder produces in float32.
    x = torch.tensor(_shifted_x(k), dtype=torch.float32).item()
    bits = torch.empty(D_GRAY_PAYLOAD, dtype=torch.float32)
    # Bit 0: raw x vs 0.5 (the MSB-like slowest bit).
    bits[0] = 1.0 if x > 0.5 else -1.0
    # Bits 1..15: compare T_{2^(i-1)}(x) to 0.5.
    for i in range(1, D_GRAY_PAYLOAD):
        m = 1 << (i - 1)
        y = m * x
        y_frac = y - math.floor(y)
        feature = 1.0 - abs(2.0 * y_frac - 1.0)
        bits[i] = 1.0 if feature > 0.5 else -1.0
    return bits


# ---------------------------------------------------------------------------
# W_EMBED construction
# ---------------------------------------------------------------------------


def _build_w_embed() -> torch.Tensor:
    """Construct the (V, D_EMBED) embedding matrix.

    Rows 0..65535 (VALUE) share the ``E8_VALUE`` category code in
    cols [0:8], carry ``(2k + 1) / 131072`` in col 8 (the raw slot —
    shifted so it exactly matches the encoder's ``x`` at integer k),
    and the ±1 Gray-like code of ``k`` in cols [9:25].

    Rows 0..MAX_INT_K of the VALUE block additionally carry K=k in
    col 25 (the int-slot scheme for small-cardinality integer
    identifiers).  Rows MAX_INT_K+1..65535 leave it zero.

    Rows 65536..V-1 (non-VALUE) carry a distinct category code per
    row in cols [0:8] and zeros in cols [8:26].

    Type-tag block at cols [26:49]:

      * cols [26:47] — slot one-hot.  +1 at column ``_SLOT_ONEHOT_START + i``
        for the IDENTIFIER row at slot ``i``; −1 in every other one of
        the 21 columns. Non-identifier rows carry all −1.  Stored as
        ±1 so a per-position direct extract IS the ±1 bool that
        ``cond_gate`` and ``bool_all_true`` consumers expect.
      * col [47] — is_any_identifier: +1 for the 21 IDENTIFIER_NAMES
        rows, −1 elsewhere.
      * col [48] — is_value_category: +1 for the 65,536 VALUE rows,
        −1 elsewhere.
    """
    w = torch.zeros((V, D_EMBED), dtype=torch.float32)

    e8_value = _category_code("VALUE")
    raw_col = D_CATEGORY
    gray_start = D_CATEGORY + D_RAW_SLOT  # 9
    k_col = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD  # 25

    # Type-tag block defaults to −1 everywhere; specific +1 assignments
    # below mark the rows that carry the corresponding flag.
    w[:, _SLOT_ONEHOT_START : _SLOT_ONEHOT_START + D_SLOT_ONEHOT] = -1.0
    w[:, _IS_ANY_ID_COL] = -1.0
    w[:, _IS_VALUE_CATEGORY_COL] = -1.0

    for vid in range(N_VALUES):
        w[vid, 0:D_CATEGORY] = e8_value
        w[vid, raw_col] = _shifted_x(vid)
        w[vid, gray_start : gray_start + D_GRAY_PAYLOAD] = gray_code_16(vid)
        if vid <= MAX_INT_K:
            w[vid, k_col] = float(vid)
        # is_value_category = +1 for every VALUE row (slot one-hot
        # stays −1 from the bulk init; is_any_identifier stays −1).
        w[vid, _IS_VALUE_CATEGORY_COL] = 1.0

    # Non-VALUE rows: only the category code is non-zero (cols [8:27]).
    _write_category_row(w, "THINKING_WALL_", start_id=_THINKING_WALL_BASE, count=8)
    for offset, name in enumerate(_PER_WALL_IDENTIFIERS):
        w[_PER_WALL_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_RESOLVED_IDENTIFIERS):
        w[_RESOLVED_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_SORT_IDENTIFIERS):
        w[_SORT_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_DECODE_TOKENS):
        w[_DECODE_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_PROMPT_TOKENS):
        w[_PROMPT_BASE + offset, 0:D_CATEGORY] = _category_code(name)

    # Type-tag block for IDENTIFIER rows.  IDENTIFIER_NAMES is built
    # below this function, so reconstruct the same ordering inline:
    # 17 per-wall + 3 RESOLVED + 1 SORT_RESULT.
    _identifier_rows = []
    for i, name in enumerate(_PER_WALL_IDENTIFIERS):
        _identifier_rows.append((_PER_WALL_BASE + i, i))
    base_offset = len(_PER_WALL_IDENTIFIERS)
    for i, name in enumerate(_RESOLVED_IDENTIFIERS):
        _identifier_rows.append((_RESOLVED_BASE + i, base_offset + i))
    base_offset += len(_RESOLVED_IDENTIFIERS)
    for i, name in enumerate(_SORT_IDENTIFIERS):
        _identifier_rows.append((_SORT_BASE + i, base_offset + i))

    for row_id, slot in _identifier_rows:
        w[row_id, _SLOT_ONEHOT_START + slot] = 1.0
        w[row_id, _IS_ANY_ID_COL] = 1.0

    return w


def _write_category_row(
    w: torch.Tensor, name_prefix: str, *, start_id: int, count: int
) -> None:
    for i in range(count):
        w[start_id + i, 0:D_CATEGORY] = _category_code(f"{name_prefix}{i}")


W_EMBED: torch.Tensor = _build_w_embed()
assert W_EMBED.shape == (V, D_EMBED)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


# Ordered list of the 21 identifier names: 17 per-wall + 3 RESOLVED
# (thinking-phase cascade) + 1 SORT_RESULT (tags the VALUE token that
# carries the picked wall_index after each SORTED_WALL marker).  The
# first 20 entries form the thinking-phase cascade — each entry at
# index ``i`` is the identifier emitted at the VALUE step whose most
# recent identifier was the entry at index ``i - 1``.  SORT_RESULT at
# index 20 sits outside that cascade: its identifier-position emit
# comes from the SORTED stage's quadratic-equality attention (bypasses
# thinking_wall's norm_by_slot), but it participates in the shared
# prev-id storage + readback machinery so ``ThinkingReadback`` and
# downstream consumers can retrieve its VALUE uniformly.
IDENTIFIER_NAMES: List[str] = (
    list(_PER_WALL_IDENTIFIERS) + list(_RESOLVED_IDENTIFIERS) + list(_SORT_IDENTIFIERS)
)
assert len(IDENTIFIER_NAMES) == 21
# The slot-onehot block in W_EMBED has one column per IDENTIFIER_NAMES
# entry; the constant must match the list length.
assert D_SLOT_ONEHOT == len(IDENTIFIER_NAMES)

# Subset used by the thinking-phase state machine (``thinking_wall``
# builds per-slot norm/factor machinery only over these).  SORT_RESULT
# is intentionally absent — its payload is computed by the SORTED
# stage, not by the thinking cascade.
THINKING_IDENTIFIER_NAMES: List[str] = list(_PER_WALL_IDENTIFIERS) + list(
    _RESOLVED_IDENTIFIERS
)
assert len(THINKING_IDENTIFIER_NAMES) == 20


# Float range of every identifier's VALUE payload.  The producing
# identifier step runs ``quantize_to_range(value, lo, hi)`` to convert
# its float into a continuous float in ``[0, N_VALUES - 1]``; the
# consuming step runs ``dequantize_from_range(q, lo, hi)`` to get back
# the float.  Single LSB == (hi - lo) / (N_VALUES - 1).
#
# For integer-valued ranges (BSP_RANK 0..7; 0/1 booleans for
# IS_RENDERABLE / HIT_*; 0..255 for RESOLVED_ANGLE), the identifier
# step emits a specific VALUE_k row directly rather than going through
# the generic quantize → Gray-code encoder; the (lo, hi) entries here
# still describe the conceptual range so the readback helper decodes
# VALUE_k back to the correct float via a single Linear on the raw
# slot.
VALUE_RANGE_BY_NAME: Dict[str, tuple[float, float]] = {
    "BSP_RANK": (0.0, 7.0),
    "IS_RENDERABLE": (0.0, 1.0),
    "CROSS_A": (-40.0, 40.0),
    "DOT_A": (-40.0, 40.0),
    "CROSS_B": (-40.0, 40.0),
    "DOT_B": (-40.0, 40.0),
    "T_STAR_L": (-2.0, 2.0),
    "T_STAR_R": (-2.0, 2.0),
    "T_LO": (0.0, 1.0),
    "T_HI": (0.0, 1.0),
    "COL_A": (-2.0, 122.0),
    "COL_B": (-2.0, 122.0),
    "VIS_LO": (-2.0, 122.0),
    "VIS_HI": (-2.0, 122.0),
    "HIT_FULL": (0.0, 1.0),
    "HIT_X": (0.0, 1.0),
    "HIT_Y": (0.0, 1.0),
    # RESOLVED_X/Y range covers the ``max_coord = 100`` ceiling that
    # scale-find / normalize_coord operate against.  Host treats
    # RESOLVED_X/Y as opaque scalars — it echoes them back as the next
    # frame's PLAYER_X / PLAYER_Y, so the graph re-normalizes
    # consistently each frame from whatever real-units position the
    # host fed in.  RESOLVED_ANGLE stays in [0, 255].
    "RESOLVED_X": (-100.0, 100.0),
    "RESOLVED_Y": (-100.0, 100.0),
    "RESOLVED_ANGLE": (0.0, 255.0),
    # SORT_RESULT's VALUE carries the picked wall_index ∈ [0, max_walls).
    # The range declared here must cover every wall_index value the
    # autoregressive loop can emit; max_walls=8 today, so 0..7.  If a
    # future max_walls exceeds 8, widen this range.
    "SORT_RESULT": (0.0, 7.0),
}
assert set(VALUE_RANGE_BY_NAME.keys()) == set(
    IDENTIFIER_NAMES
), "VALUE_RANGE_BY_NAME must cover all IDENTIFIER_NAMES"

# Slot-indexed view for callers that build per-slot machinery.
VALUE_RANGE_BY_IDX: List[tuple[float, float]] = [
    VALUE_RANGE_BY_NAME[name] for name in IDENTIFIER_NAMES
]


def vocab_id(name: str) -> int:
    """Look up the integer vocab ID for a named token (raises KeyError)."""
    return _VOCAB_IDS[name]


def value_id(n: int) -> int:
    """VALUE IDs are identical to their integer payload."""
    assert 0 <= n < N_VALUES, f"value_id out of range: {n}"
    return n


def embed_lookup(name: str) -> torch.Tensor:
    """Return the 25-wide ``W_EMBED`` row for a named token."""
    return W_EMBED[vocab_id(name)]


def category_code(name: str) -> torch.Tensor:
    """Return the 8-wide E8 category code for a category name."""
    return _category_code(name)


# E8_VALUE is the shared category code in cols [0:8] across every
# VALUE row.  Convenience export for the "is this any VALUE token?"
# category-only detector.
E8_VALUE: torch.Tensor = _category_code("VALUE")


def build_doom_embedding(input_name: str = "token_ids") -> Embedding:
    """Factory for the DOOM Embedding graph node.

    ``input_name`` is the slot :class:`Embedding.compute` reads from
    ``input_values`` — wired to the 1-wide integer ``token_ids``
    input declared on the DOOM graph.
    """
    return Embedding(
        vocab=DOOM_VOCAB,
        d_embed=D_EMBED,
        table=W_EMBED,
        input_name=input_name,
        special_tokens=[],  # no <unk> prefix; row i ≡ vocab_id(DOOM_VOCAB[i])
    )
