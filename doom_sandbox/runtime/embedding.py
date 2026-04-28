"""Token embedding layout and embed/deembed routines.

Layout: per-vocab, computed once. The Vec encoding token T is

    [ type_one_hot(N_TYPES) | per-(type, slot) columns ]

so each token type owns its own dedicated columns for its declared
slots. `make_token(T, slot=v)` writes 1.0 to T's one-hot column and v
into T's `slot` column, leaving every other column zero. This means
`extract_*_slot(vec, name)` can recover a slot value by summing across
every (type, slot) column with that slot name — at most one of them is
non-zero at any moment.

`deembed` argmaxes the type one-hot, then reads each declared slot's
column for the chosen type. `IntSlot` values are rounded; `FloatSlot`
values are snapped to one of `levels` discrete steps so that the
autoregressive re-embed → deembed cycle incurs the documented one-LSB
quantization error.

Activation: token primitives (`make_token`, `extract_*_slot`,
`is_type`) need to know the layout, but the user's `forward()` doesn't
pass it. The framework activates a layout via the `active_layout(...)`
context manager around each `run()`; the API functions consult
`get_active_layout()`. Tests can use the context manager directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from ..api.tokens import (
    FloatSlot,
    IntSlot,
    Slot,
    Token,
    TokenType,
)
from ..api.vec import Vec, _make_vec


class Layout:
    """Per-vocab Vec layout: type one-hot + per-(type, slot) columns."""

    def __init__(self, types: list[TokenType]):
        names = [t.name for t in types]
        if len(set(names)) != len(names):
            raise ValueError(
                f"TokenVocab has duplicate type names: {names}"
            )
        self.types: list[TokenType] = list(types)
        self.types_by_name: dict[str, TokenType] = {t.name: t for t in types}
        self.type_columns: dict[str, int] = {t.name: i for i, t in enumerate(types)}
        n_types = len(types)

        self.slot_columns: dict[tuple[str, str], int] = {}
        self.slot_kinds: dict[tuple[str, str], Slot] = {}
        self.columns_by_slot_name: dict[str, list[tuple[str, int]]] = {}

        col = n_types
        for t in types:
            for slot_name, slot in t.slots.items():
                key = (t.name, slot_name)
                self.slot_columns[key] = col
                self.slot_kinds[key] = slot
                self.columns_by_slot_name.setdefault(slot_name, []).append(
                    (t.name, col)
                )
                col += 1
        self.width: int = col

    def has_type(self, token_type: TokenType) -> bool:
        return token_type.name in self.types_by_name

    def slot_kind(self, token_type: TokenType, slot_name: str) -> Slot | None:
        return self.slot_kinds.get((token_type.name, slot_name))


_ACTIVE_LAYOUT: Layout | None = None


def get_active_layout() -> Layout:
    if _ACTIVE_LAYOUT is None:
        raise RuntimeError(
            "No active token layout. Token primitives "
            "(make_token / extract_*_slot / is_type) can only be called "
            "inside a run(). For unit tests, wrap calls in "
            "`with active_layout(layout): ...`."
        )
    return _ACTIVE_LAYOUT


@contextmanager
def active_layout(layout: Layout) -> Iterator[Layout]:
    """Activate `layout` for the duration of the `with` block."""
    global _ACTIVE_LAYOUT
    prev = _ACTIVE_LAYOUT
    _ACTIVE_LAYOUT = layout
    try:
        yield layout
    finally:
        _ACTIVE_LAYOUT = prev


def embed(token: Token, layout: Layout) -> Vec:
    """Encode a discrete `Token` into a Vec of width `layout.width`."""
    if not layout.has_type(token.type):
        raise ValueError(
            f"Token type {token.type.name!r} is not in the active vocab"
        )
    data = np.zeros(layout.width, dtype=np.float64)
    data[layout.type_columns[token.type.name]] = 1.0
    for slot_name, value in token.values.items():
        kind = layout.slot_kind(token.type, slot_name)
        if kind is None:
            raise ValueError(
                f"Type {token.type.name!r} does not declare slot {slot_name!r}"
            )
        col = layout.slot_columns[(token.type.name, slot_name)]
        data[col] = float(value)
    return _make_vec(data, depth=0)


def deembed(vec: Vec, layout: Layout) -> Token:
    """Decode a Vec to a discrete `Token` via argmax on the type one-hot,
    then per-slot column reads. `FloatSlot` values are quantized to one of
    `levels` evenly-spaced steps; `IntSlot` values are rounded and clamped
    to `[lo, hi)`."""
    if vec.shape != layout.width:
        raise ValueError(
            f"deembed expected a Vec of width {layout.width}, got {vec.shape}"
        )
    n_types = len(layout.types)
    type_logits = vec._data[:n_types]
    type_idx = int(np.argmax(type_logits))
    token_type = layout.types[type_idx]
    values: dict[str, int | float] = {}
    for slot_name, slot in token_type.slots.items():
        col = layout.slot_columns[(token_type.name, slot_name)]
        raw = float(vec._data[col])
        if isinstance(slot, IntSlot):
            v = int(round(raw))
            v = max(slot.lo, min(slot.hi - 1, v))
            values[slot_name] = v
        elif isinstance(slot, FloatSlot):
            values[slot_name] = _quantize_float(raw, slot)
        else:  # pragma: no cover — exhaustive over Slot union
            raise TypeError(f"Unknown slot kind: {type(slot).__name__}")
    return Token(type=token_type, values=values)


def _quantize_float(raw: float, slot: FloatSlot) -> float:
    """Snap `raw` to one of `slot.levels` evenly-spaced steps over [lo, hi]."""
    span = slot.hi - slot.lo
    t = (raw - slot.lo) / span
    t = max(0.0, min(1.0, t))
    idx = round(t * (slot.levels - 1))
    return slot.lo + (idx / (slot.levels - 1)) * span
