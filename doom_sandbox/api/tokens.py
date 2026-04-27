"""Token types, slots, and per-position token construction/extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vec import Vec, _make_vec


@dataclass(frozen=True)
class IntSlot:
    """A slot carrying an integer in `[lo, hi)`. Encoded exactly."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(
                f"IntSlot requires lo < hi, got {self.lo}, {self.hi}"
            )


@dataclass(frozen=True)
class FloatSlot:
    """A slot carrying a float in `[lo, hi]`, quantized to `levels` steps."""

    lo: float
    hi: float
    levels: int = 65536

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(
                f"FloatSlot requires lo < hi, got {self.lo}, {self.hi}"
            )
        if self.levels < 2:
            raise ValueError(
                f"FloatSlot requires levels >= 2, got {self.levels}"
            )


Slot = IntSlot | FloatSlot


@dataclass(frozen=True, eq=False)
class TokenType:
    """A token type with named typed parameter slots.

    Token types are declared at module level and packed into the
    vocabulary in `setup()`. Identity is by `name` — types with the same
    name compare equal and hash equally regardless of slot definitions.
    Names are expected to be unique within a vocab.
    """

    name: str
    slots: dict[str, Slot] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TokenType) and self.name == other.name


@dataclass(frozen=True)
class Token:
    """A discrete token instance — produced by deembedding an output Vec.

    Carries the token type and a `values` mapping from slot name to
    decoded Python value (int for `IntSlot`, float for `FloatSlot`).
    """

    type: TokenType
    values: dict[str, int | float] = field(default_factory=dict)


def make_token(token_type: TokenType, **slot_values: Vec) -> Vec:
    """Build the embedding Vec for a token of `token_type` with the given slots.

    Slot values are passed as 1-shape Vecs (typically the result of PWL
    chains, slot extractions, or `past.*` results). Slots not specified
    default to 0 for `IntSlot`, 0.0 for `FloatSlot`.

    Adds depth +1.
    """
    from ..runtime.embedding import get_active_layout

    layout = get_active_layout()
    if not layout.has_type(token_type):
        raise ValueError(
            f"Token type {token_type.name!r} is not in the active vocab"
        )
    # Validate slots against the layout's canonical TokenType, not the
    # caller's instance — TokenType equality is by name only, so a
    # caller-side mismatch on `.slots` would otherwise silently produce
    # a KeyError on the column lookup below.
    canonical = layout.types_by_name[token_type.name]
    data = np.zeros(layout.width, dtype=np.float64)
    data[layout.type_columns[canonical.name]] = 1.0
    max_input_depth = 0
    for slot_name, slot_vec in slot_values.items():
        if slot_name not in canonical.slots:
            raise ValueError(
                f"Token type {canonical.name!r} does not declare slot "
                f"{slot_name!r}; declared slots: {list(canonical.slots)}"
            )
        if slot_vec.shape != 1:
            raise ValueError(
                f"slot value {slot_name!r} must be a 1-shape Vec, "
                f"got shape {slot_vec.shape}"
            )
        col = layout.slot_columns[(canonical.name, slot_name)]
        data[col] = float(slot_vec._data[0])
        max_input_depth = max(max_input_depth, slot_vec.depth)
    return _make_vec(data, depth=max_input_depth + 1)


def extract_int_slot(input_vec: Vec, name: str) -> Vec:
    """Extract the named `IntSlot`'s value from `input_vec` as a 1-shape Vec.

    If the slot isn't declared on the current input's token type, returns
    a 1-shape Vec containing 0. Use `is_type` masks to dispatch by type.

    Adds depth +1.
    """
    return _extract_slot(input_vec, name, IntSlot)


def extract_float_slot(input_vec: Vec, name: str) -> Vec:
    """Extract the named `FloatSlot`'s value from `input_vec` as a 1-shape Vec.

    If the slot isn't declared on the current input's token type, returns
    a 1-shape Vec containing 0.0. Use `is_type` masks to dispatch by type.

    Adds depth +1.
    """
    return _extract_slot(input_vec, name, FloatSlot)


def _extract_slot(
    input_vec: Vec, name: str, slot_kind: type[IntSlot] | type[FloatSlot]
) -> Vec:
    # Deferred import: runtime/embedding.py imports TokenType/Slot from
    # this module, so a top-level import would cycle.
    from ..runtime.embedding import get_active_layout

    layout = get_active_layout()
    if input_vec.shape != layout.width:
        raise ValueError(
            f"extract slot expected a Vec of width {layout.width}, "
            f"got shape {input_vec.shape}"
        )
    entries = layout.columns_by_slot_name.get(name, [])
    matching_cols = [
        col
        for type_name, col in entries
        if isinstance(layout.slot_kinds[(type_name, name)], slot_kind)
    ]
    if not entries:
        raise ValueError(
            f"slot {name!r} is not declared on any type in the active vocab"
        )
    if not matching_cols:
        kinds = sorted(
            type(layout.slot_kinds[(type_name, name)]).__name__
            for type_name, _ in entries
        )
        raise ValueError(
            f"slot {name!r} is not declared as {slot_kind.__name__} on any "
            f"type; declared as {kinds}"
        )
    value = float(np.sum(input_vec._data[matching_cols]))
    data = np.array([value], dtype=np.float64)
    return _make_vec(data, depth=input_vec.depth + 1)


def is_type(input_vec: Vec, token_type: TokenType) -> Vec:
    """Return a 1-shape Vec: 1.0 if `input_vec` is `token_type`, else 0.0.

    Types are mutually exclusive — across the vocabulary, exactly one
    `is_type(input_vec, T)` returns 1.0 at any given position.

    Adds depth +1.
    """
    from ..runtime.embedding import get_active_layout

    layout = get_active_layout()
    if not layout.has_type(token_type):
        raise ValueError(
            f"Token type {token_type.name!r} is not in the active vocab"
        )
    if input_vec.shape != layout.width:
        raise ValueError(
            f"is_type expected a Vec of width {layout.width}, "
            f"got shape {input_vec.shape}"
        )
    col = layout.type_columns[token_type.name]
    data = np.array([float(input_vec._data[col])], dtype=np.float64)
    return _make_vec(data, depth=input_vec.depth + 1)
