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

import enum
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Union

_INF = math.inf


class Guarantee(enum.Enum):
    """Strength of a structural claim about a node's output.

    ALWAYS
        Structurally exact — the mathematical construction guarantees
        this property holds for every element.  Examples: LiteralValue
        with integer data, Linear with integer weights on integer input.

    APPROXIMATE
        Holds outside narrow piecewise-linear transition zones.  The
        ReLU-based step-function approximation has an eps-wide ramp
        where intermediate values appear.  Examples: floor_int, compare,
        in_range, thermometer_floor_div, equals_vector.
    """

    ALWAYS = "always"
    APPROXIMATE = "approximate"

    def __bool__(self) -> bool:
        """Both levels are truthy, enabling ``if vt.is_integer:`` compat."""
        return True


# Type alias for the property fields: Guarantee.ALWAYS, Guarantee.APPROXIMATE,
# or False (no claim).
GuaranteeLevel = Union[Guarantee, bool]  # bool is only ever False at runtime


def _min_guarantee(a: GuaranteeLevel, b: GuaranteeLevel) -> GuaranteeLevel:
    """AND semantics: both must claim; result is the weaker level.

    Used by ``intersect_element_props`` and propagation rules where an
    output property holds only if *all* relevant inputs have it.
    """
    if a is False or b is False:
        return False
    if a is Guarantee.APPROXIMATE or b is Guarantee.APPROXIMATE:
        return Guarantee.APPROXIMATE
    return Guarantee.ALWAYS


def _max_guarantee(a: GuaranteeLevel, b: GuaranteeLevel) -> GuaranteeLevel:
    """OR semantics: either can claim; result is the weaker of the claimants.

    Used by ``tightened_with`` where if *either* side makes a claim the
    claim survives, but at the weaker of the two guarantee levels.
    """
    if a is False and b is False:
        return False
    if a is False:
        return b
    if b is False:
        return a
    if a is Guarantee.APPROXIMATE or b is Guarantee.APPROXIMATE:
        return Guarantee.APPROXIMATE
    return Guarantee.ALWAYS


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

    Each boolean property is either ``False`` (no claim),
    ``Guarantee.ALWAYS`` (structurally exact), or
    ``Guarantee.APPROXIMATE`` (holds outside piecewise-linear
    transition zones).  Both enum values are truthy so existing
    ``if vt.is_integer:`` checks work unchanged.
    """

    # --- Element properties ---
    value_range: Range = field(default_factory=Range.unbounded)
    is_integer: GuaranteeLevel = False
    is_binary: GuaranteeLevel = False
    is_sign: GuaranteeLevel = False

    # --- Vector properties ---
    is_one_hot: GuaranteeLevel = False

    def __post_init__(self):
        # Auto-upgrade bare True → Guarantee.ALWAYS for migration safety.
        for field_name in ("is_integer", "is_binary", "is_sign", "is_one_hot"):
            val = getattr(self, field_name)
            if val is True:
                object.__setattr__(self, field_name, Guarantee.ALWAYS)

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
    def integer(
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        *,
        guarantee: Guarantee = Guarantee.ALWAYS,
    ) -> "NodeValueType":
        r = Range(-_INF if lo is None else float(lo), _INF if hi is None else float(hi))
        return NodeValueType(value_range=r, is_integer=guarantee)

    @staticmethod
    def binary(*, guarantee: Guarantee = Guarantee.ALWAYS) -> "NodeValueType":
        return NodeValueType(
            value_range=Range(0.0, 1.0), is_integer=guarantee, is_binary=guarantee
        )

    @staticmethod
    def sign(*, guarantee: Guarantee = Guarantee.ALWAYS) -> "NodeValueType":
        return NodeValueType(
            value_range=Range(-1.0, 1.0), is_integer=guarantee, is_sign=guarantee
        )

    @staticmethod
    def one_hot(*, guarantee: Guarantee = Guarantee.ALWAYS) -> "NodeValueType":
        return NodeValueType(
            value_range=Range(0.0, 1.0),
            is_integer=guarantee,
            is_binary=guarantee,
            is_one_hot=guarantee,
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


def is_integer_tensor(t) -> bool:
    """True iff every element of ``t`` equals its rounded value.

    Tolerates ``None`` by returning ``False`` (callers treat
    missing biases as "unknown" and skip the integer claim).
    """
    import torch

    if t is None:
        return False
    if not isinstance(t, torch.Tensor):
        return False
    if t.numel() == 0:
        return True
    return bool(torch.all(t == t.round()).item())


def linear_output_range(input_range: Range, matrix, bias=None) -> Range:
    """Interval range of ``x @ matrix + bias`` given ``x`` elements ∈ input_range.

    Returns the union over output columns of each column's interval
    (per-scalar range). Unbounded input ⇒ unbounded output.
    """
    import torch

    if not input_range.is_finite():
        return Range.unbounded()
    m = matrix
    lo_prod = input_range.lo * m
    hi_prod = input_range.hi * m
    mins = torch.minimum(lo_prod, hi_prod).sum(dim=0)
    maxs = torch.maximum(lo_prod, hi_prod).sum(dim=0)
    if bias is not None:
        mins = mins + bias
        maxs = maxs + bias
    return Range(float(mins.min().item()), float(maxs.max().item()))


def tightened_with(a: NodeValueType, b: NodeValueType) -> NodeValueType:
    """Combine two claims into the strictest type both admit.

    Ranges are intersected; structural claims are OR-ed (either side's
    claim survives, at the weaker of the two guarantee levels).  Range
    is further tightened to the invariant constraints implied by the
    OR-ed claims (e.g. ``is_binary`` forces range ⊆ [0, 1]).

    Used by the compiler's Assert-strip pass to transfer an Assert's
    ``claimed_type`` onto the node it wrapped, so downstream analysis
    that runs after stripping still sees the strengthened type.
    """
    r = a.value_range.intersect(b.value_range)
    is_int = _max_guarantee(a.is_integer, b.is_integer)
    is_bin = _max_guarantee(a.is_binary, b.is_binary)
    is_sgn = _max_guarantee(a.is_sign, b.is_sign)
    is_onehot = _max_guarantee(a.is_one_hot, b.is_one_hot)
    if is_bin:
        r = r.intersect(Range(0.0, 1.0))
    if is_sgn:
        r = r.intersect(Range(-1.0, 1.0))
    return NodeValueType(
        value_range=r,
        is_integer=is_int,
        is_binary=is_bin,
        is_sign=is_sgn,
        is_one_hot=is_onehot,
    )


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
        is_integer=_min_guarantee(a.is_integer, b.is_integer),
        is_binary=_min_guarantee(a.is_binary, b.is_binary),
        is_sign=_min_guarantee(a.is_sign, b.is_sign),
        is_one_hot=False,
    )
