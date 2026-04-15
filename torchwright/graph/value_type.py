"""Static value-property types attached to every Node.

Each Node has a ``value_type: NodeValueType`` that summarises what we
statically know about its output tensor — element range, integer-ness,
one-hot-ness, and so on. Propagation rules defined per-op populate this
eagerly at graph-build time, and primitives can enforce contracts on
their inputs via the helpers in ``torchwright.graph.asserts``.

``Range`` uses ±inf for unbounded sides so arithmetic is total; no
``Optional[Range]`` handling is needed at every rule site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional


_INF = math.inf


@dataclass(frozen=True)
class Range:
    """Closed interval ``[lo, hi]`` with ±inf endpoints allowed.

    A ``Range`` always satisfies ``lo <= hi``. Use ``Range.unbounded()``
    for "no information"; use the arithmetic helpers (``__add__`` etc.)
    rather than reimplementing interval math at each op.
    """

    lo: float = -_INF
    hi: float = _INF

    def __post_init__(self):
        if not (self.lo <= self.hi):
            raise ValueError(f"Range lo must be <= hi, got lo={self.lo}, hi={self.hi}")

    @staticmethod
    def unbounded() -> "Range":
        return Range(-_INF, _INF)

    @staticmethod
    def point(v: float) -> "Range":
        return Range(v, v)

    def contains(self, other: "Range") -> bool:
        return self.lo <= other.lo and other.hi <= self.hi

    def is_finite(self) -> bool:
        return math.isfinite(self.lo) and math.isfinite(self.hi)

    def __add__(self, other: "Range") -> "Range":
        return Range(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> "Range":
        return Range(-self.hi, -self.lo)

    def __sub__(self, other: "Range") -> "Range":
        return self + (-other)

    def union(self, other: "Range") -> "Range":
        return Range(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersect(self, other: "Range") -> "Range":
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            # Empty intersection — fall back to the narrower side of other
            # so downstream rules stay total. Callers that care about
            # emptiness should check explicitly.
            return Range(lo, lo)
        return Range(lo, hi)

    def relu(self) -> "Range":
        return Range(max(0.0, self.lo), max(0.0, self.hi))


@dataclass(frozen=True)
class NodeValueType:
    """Static properties of a node's output tensor.

    Element properties hold for every scalar in the output; vector
    properties describe the width-dim vector at each position.
    """

    # --- Element properties ---
    value_range: Range = field(default_factory=Range.unbounded)
    is_integer: bool = False
    is_binary: bool = False
    is_sign: bool = False

    # --- Vector properties ---
    is_one_hot: bool = False

    def __post_init__(self):
        if self.is_binary:
            if not self.is_integer:
                raise ValueError("is_binary implies is_integer")
            if not Range(0.0, 1.0).contains(self.value_range):
                raise ValueError(
                    f"is_binary requires range ⊆ [0, 1], got {self.value_range}"
                )
        if self.is_sign:
            if not self.is_integer:
                raise ValueError("is_sign implies is_integer")
            if not Range(-1.0, 1.0).contains(self.value_range):
                raise ValueError(
                    f"is_sign requires range ⊆ [-1, 1], got {self.value_range}"
                )
        if self.is_one_hot and not self.is_binary:
            raise ValueError("is_one_hot implies is_binary")

    # --- Factory helpers ------------------------------------------------

    @staticmethod
    def unknown() -> "NodeValueType":
        return NodeValueType()

    @staticmethod
    def integer(lo: Optional[float] = None, hi: Optional[float] = None) -> "NodeValueType":
        r = Range(-_INF if lo is None else float(lo), _INF if hi is None else float(hi))
        return NodeValueType(value_range=r, is_integer=True)

    @staticmethod
    def binary() -> "NodeValueType":
        return NodeValueType(
            value_range=Range(0.0, 1.0), is_integer=True, is_binary=True
        )

    @staticmethod
    def sign() -> "NodeValueType":
        return NodeValueType(
            value_range=Range(-1.0, 1.0), is_integer=True, is_sign=True
        )

    @staticmethod
    def one_hot() -> "NodeValueType":
        return NodeValueType(
            value_range=Range(0.0, 1.0),
            is_integer=True,
            is_binary=True,
            is_one_hot=True,
        )

    @staticmethod
    def bounded(lo: float, hi: float) -> "NodeValueType":
        return NodeValueType(value_range=Range(float(lo), float(hi)))

    # --- Combinators ----------------------------------------------------

    def with_range(self, r: Range) -> "NodeValueType":
        return replace(self, value_range=r)

    def drop_vector_props(self) -> "NodeValueType":
        """Return a copy with vector-level properties cleared.

        Useful when an op preserves element-wise properties but cannot
        preserve (e.g.) one-hot-ness.
        """
        return replace(self, is_one_hot=False)


def intersect_element_props(a: NodeValueType, b: NodeValueType) -> NodeValueType:
    """Meet of element-level properties: kept only if both sides have them.

    Used by ops like ``Concatenate`` where the output's per-scalar
    properties must hold across every contributing input. Vector
    properties are dropped unconditionally — they rarely survive
    concatenation.
    """
    r = Range(
        min(a.value_range.lo, b.value_range.lo),
        max(a.value_range.hi, b.value_range.hi),
    )
    return NodeValueType(
        value_range=r,
        is_integer=a.is_integer and b.is_integer,
        is_binary=a.is_binary and b.is_binary,
        is_sign=a.is_sign and b.is_sign,
        is_one_hot=False,
    )
