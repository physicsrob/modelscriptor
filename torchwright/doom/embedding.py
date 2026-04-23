"""DOOM's token vocabulary and embedding table.

Phase A Part 1: every autoregressive step consumes and emits a single
token ID. The host feeds ``token_ids`` as a 1-wide input slot; the
graph looks the ID up in ``W_EMBED`` to produce a 72-wide residual
leaf. On the output side, the 72-wide output slice is projected
through ``W_EMBED.T`` and argmaxed to pick the next ID.

Vocabulary layout matches ``docs/phase_a_plan.md`` §"Vocabulary ID
ranges" verbatim:

  |      0 .. 65535 | VALUE (quantized 16-bit integers)          |
  |  65536 .. 65543 | THINKING_WALL markers 0..7                 |
  |  65544 .. 65556 | Per-wall identifiers (BSP_RANK..HIT_Y)     |
  |  65557 .. 65559 | RESOLVED identifiers (X / Y / ANGLE)       |
  |  65560 .. 65562 | Decode tokens (SORTED_WALL, RENDER, DONE)  |
  |  65563 .. 65570 | Prompt-position categories                 |

Total ``V = 65571``.

Embedding layout (``d_embed = 72``):

  cols [ 0 :  8] — E8 category code (distinct per category name)
  cols [ 8 : 24] — one_hot(h3), bits 12..15 of VALUE payload
  cols [24 : 40] — one_hot(h2), bits  8..11
  cols [40 : 56] — one_hot(h1), bits  4..7
  cols [56 : 72] — one_hot(h0), bits  0..3

The payload columns are zero for non-VALUE rows. All 65,536 VALUE
rows share a single ``E8_VALUE`` category code; their payload
columns distinguish them via a 4+4+4+4 factored one-hot.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from torchwright.graph.embedding import Embedding
from torchwright.graph.spherical_codes import index_to_vector

D_EMBED: int = 72
D_CATEGORY: int = 8
N_HEX_DIGITS: int = 4
HEX_WIDTH: int = 16  # one-hot over 16 values per hex digit
assert D_CATEGORY + N_HEX_DIGITS * HEX_WIDTH == D_EMBED

N_VALUES: int = 1 << (N_HEX_DIGITS * 4)  # 65,536
assert N_VALUES == HEX_WIDTH**N_HEX_DIGITS


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
    # Per-wall identifiers (13).  HIT_FULL/X/Y keep their M4 indices.
    "BSP_RANK": 270,
    "IS_RENDERABLE": 271,
    "CROSS_A": 272,
    "DOT_A": 273,
    "CROSS_B": 274,
    "DOT_B": 275,
    "T_LO": 276,
    "T_HI": 277,
    "VIS_LO": 278,
    "VIS_HI": 279,
    "HIT_FULL": 258,
    "HIT_X": 259,
    "HIT_Y": 260,
    # RESOLVED identifiers (3).
    "RESOLVED_X": 280,
    "RESOLVED_Y": 281,
    "RESOLVED_ANGLE": 282,
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
# Vocabulary ID allocation (matches docs/phase_a_plan.md §"Vocabulary ID
# ranges" exactly).
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
    "T_LO",
    "T_HI",
    "VIS_LO",
    "VIS_HI",
    "HIT_FULL",
    "HIT_X",
    "HIT_Y",
]
_PER_WALL_BASE = _THINKING_WALL_BASE + 8  # 65544 .. 65556

_RESOLVED_IDENTIFIERS: List[str] = ["RESOLVED_X", "RESOLVED_Y", "RESOLVED_ANGLE"]
_RESOLVED_BASE = _PER_WALL_BASE + len(_PER_WALL_IDENTIFIERS)  # 65557 .. 65559

_DECODE_TOKENS: List[str] = ["SORTED_WALL", "RENDER", "DONE"]
_DECODE_BASE = _RESOLVED_BASE + len(_RESOLVED_IDENTIFIERS)  # 65560 .. 65562

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
_PROMPT_BASE = _DECODE_BASE + len(_DECODE_TOKENS)  # 65563 .. 65570

V: int = _PROMPT_BASE + len(_PROMPT_TOKENS)  # 65571
assert V == 65571, f"Vocabulary size mismatch: got {V}, expected 65571"


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
# W_EMBED construction
# ---------------------------------------------------------------------------


def _build_w_embed() -> torch.Tensor:
    """Construct the (V, 72) embedding matrix.

    Rows 0..65535 (VALUE) share the ``E8_VALUE`` category code and
    distinguish themselves via a 4+4+4+4 factored one-hot of the
    16-bit payload.

    Rows 65536..V-1 (non-VALUE) carry a distinct category code per
    row and zeros in the payload columns.
    """
    w = torch.zeros((V, D_EMBED), dtype=torch.float32)

    e8_value = _category_code("VALUE")

    # VALUE rows.  Each ID's 16 bits decompose into 4 hex digits
    # (h3..h0).  Each digit is one-hot-encoded into a 16-wide block.
    value_block_starts = [D_CATEGORY + i * HEX_WIDTH for i in range(N_HEX_DIGITS)]
    for vid in range(N_VALUES):
        w[vid, 0:D_CATEGORY] = e8_value
        for digit_idx in range(N_HEX_DIGITS):
            shift = (N_HEX_DIGITS - 1 - digit_idx) * 4  # h3 is most significant
            digit = (vid >> shift) & 0xF
            col = value_block_starts[digit_idx] + digit
            w[vid, col] = 1.0

    # Non-VALUE rows: only the category code is non-zero.
    _write_category_row(w, "THINKING_WALL_", start_id=_THINKING_WALL_BASE, count=8)
    for offset, name in enumerate(_PER_WALL_IDENTIFIERS):
        w[_PER_WALL_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_RESOLVED_IDENTIFIERS):
        w[_RESOLVED_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_DECODE_TOKENS):
        w[_DECODE_BASE + offset, 0:D_CATEGORY] = _category_code(name)
    for offset, name in enumerate(_PROMPT_TOKENS):
        w[_PROMPT_BASE + offset, 0:D_CATEGORY] = _category_code(name)

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


# Ordered list of the 16 identifier names that the thinking-phase state
# machine walks per wall (13 per-wall) plus per-frame (3 RESOLVED).  The
# ordering is the cascade order: each entry at index ``i`` is the
# identifier emitted at the VALUE step whose most recent identifier was
# the entry at index ``i - 1``.  The ``thinking_wall`` stage and
# ``_detect_token_types`` both iterate this list to build the 16-wide
# slot machinery.
IDENTIFIER_NAMES: List[str] = list(_PER_WALL_IDENTIFIERS) + list(_RESOLVED_IDENTIFIERS)
assert len(IDENTIFIER_NAMES) == 16


def vocab_id(name: str) -> int:
    """Look up the integer vocab ID for a named token (raises KeyError)."""
    return _VOCAB_IDS[name]


def value_id(n: int) -> int:
    """VALUE IDs are identical to their integer payload."""
    assert 0 <= n < N_VALUES, f"value_id out of range: {n}"
    return n


def embed_lookup(name: str) -> torch.Tensor:
    """Return the 72-wide ``W_EMBED`` row for a named token."""
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
