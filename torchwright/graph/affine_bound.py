"""Affine bound propagation for the computation graph.

An ``AffineBound`` represents per-component lower and upper affine
expressions over a shared basis of ``InputNode`` components::

    lower(x)[i] = A_lo[i, :] · x + b_lo[i]
    upper(x)[i] = A_hi[i, :] · x + b_hi[i]

where ``x`` is the concatenation of all ``InputNode`` values (the basis).

All coefficient tensors are stored as ``torch.float64``, CPU-only.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

from torchwright.graph.basis import Basis
from torchwright.graph.value_type import Range


@dataclass
class AffineBound:
    """Per-component affine lower/upper bounds over a shared basis.

    Attributes:
        A_lo: Lower bound coefficient matrix, shape ``(d_output, n_basis)``.
        A_hi: Upper bound coefficient matrix, shape ``(d_output, n_basis)``.
        b_lo: Lower bound offset vector, shape ``(d_output,)``.
        b_hi: Upper bound offset vector, shape ``(d_output,)``.
        basis: The shared basis over which coefficients are defined.
    """

    A_lo: torch.Tensor
    A_hi: torch.Tensor
    b_lo: torch.Tensor
    b_hi: torch.Tensor
    basis: Basis

    def __post_init__(self):
        d = self.A_lo.shape[0]
        n = self.basis.n
        assert self.A_lo.shape == (d, n), f"A_lo shape {self.A_lo.shape} != ({d}, {n})"
        assert self.A_hi.shape == (d, n), f"A_hi shape {self.A_hi.shape} != ({d}, {n})"
        assert self.b_lo.shape == (d,), f"b_lo shape {self.b_lo.shape} != ({d},)"
        assert self.b_hi.shape == (d,), f"b_hi shape {self.b_hi.shape} != ({d},)"

    @property
    def d_output(self) -> int:
        return self.A_lo.shape[0]

    @classmethod
    def identity(cls, basis: Basis, input_node) -> "AffineBound":
        """One-hot rows for *input_node*'s basis slice, zero offsets."""
        from torchwright.graph.misc import InputNode

        assert isinstance(input_node, InputNode)
        start, width = basis.index_of(input_node)
        d = input_node.d_output
        n = basis.n

        A = torch.zeros(d, n, dtype=torch.float64)
        for i in range(d):
            A[i, start + i] = 1.0

        b = torch.zeros(d, dtype=torch.float64)
        return cls(
            A_lo=A.clone(), A_hi=A.clone(), b_lo=b.clone(), b_hi=b.clone(), basis=basis
        )

    @classmethod
    def constant(cls, basis: Basis, values: torch.Tensor) -> "AffineBound":
        """Zero ``A``, ``b_lo = b_hi = values``."""
        d = values.shape[0]
        n = basis.n
        A = torch.zeros(d, n, dtype=torch.float64)
        b = values.to(torch.float64)
        return cls(
            A_lo=A.clone(), A_hi=A.clone(), b_lo=b.clone(), b_hi=b.clone(), basis=basis
        )

    @classmethod
    def degenerate(
        cls,
        basis: Basis,
        d_output: int,
        *,
        lo: float = float("-inf"),
        hi: float = float("inf"),
    ) -> "AffineBound":
        """Zero ``A``, broadcast scalar offsets."""
        n = basis.n
        A = torch.zeros(d_output, n, dtype=torch.float64)
        b_lo = torch.full((d_output,), lo, dtype=torch.float64)
        b_hi = torch.full((d_output,), hi, dtype=torch.float64)
        return cls(A_lo=A.clone(), A_hi=A.clone(), b_lo=b_lo, b_hi=b_hi, basis=basis)

    def to_interval(self) -> list[Range]:
        """Concretize to per-component scalar intervals.

        For each output component, evaluate the affine expressions over
        the input ranges (from the basis's InputNode declared ranges) to
        get the tightest scalar interval.
        """
        import math

        x_lo, x_hi = self._basis_box()
        ranges = []
        for i in range(self.d_output):
            lo = self._eval_lower(i, x_lo, x_hi)
            hi = self._eval_upper(i, x_lo, x_hi)
            if math.isnan(lo):
                lo = float("-inf")
            if math.isnan(hi):
                hi = float("inf")
            ranges.append(Range(lo, hi))
        return ranges

    def to_scalar_range(self) -> Range:
        """Union of all per-component intervals into one scalar Range."""
        intervals = self.to_interval()
        if not intervals:
            return Range.unbounded()
        lo = min(r.lo for r in intervals)
        hi = max(r.hi for r in intervals)
        return Range(lo, hi)

    def _basis_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_lo, x_hi) vectors for the input basis box."""
        import math

        n = self.basis.n
        x_lo = torch.full((n,), float("-inf"), dtype=torch.float64)
        x_hi = torch.full((n,), float("inf"), dtype=torch.float64)

        from torchwright.graph.session import current_session

        for inp in current_session().input_nodes:
            if inp.node_id not in self.basis.slices:
                continue
            start, width = self.basis.slices[inp.node_id]
            r = inp.value_type.value_range
            for j in range(width):
                if math.isfinite(r.lo):
                    x_lo[start + j] = r.lo
                if math.isfinite(r.hi):
                    x_hi[start + j] = r.hi
        return x_lo, x_hi

    def _eval_lower(self, i: int, x_lo: torch.Tensor, x_hi: torch.Tensor) -> float:
        """Evaluate lower bound for component i: min of A_lo[i] · x + b_lo[i]."""
        a = self.A_lo[i]
        pos = torch.clamp(a, min=0)
        neg = torch.clamp(a, max=0)
        # 0 * inf = nan in IEEE; force to 0 (zero coefficient means no dependence)
        term = torch.where(a == 0, torch.zeros_like(a), pos * x_lo + neg * x_hi)
        val = term.sum() + self.b_lo[i]
        return float(val.item())

    def _eval_upper(self, i: int, x_lo: torch.Tensor, x_hi: torch.Tensor) -> float:
        """Evaluate upper bound for component i: max of A_hi[i] · x + b_hi[i]."""
        a = self.A_hi[i]
        pos = torch.clamp(a, min=0)
        neg = torch.clamp(a, max=0)
        term = torch.where(a == 0, torch.zeros_like(a), pos * x_hi + neg * x_lo)
        val = term.sum() + self.b_hi[i]
        return float(val.item())

    def __repr__(self) -> str:
        intervals = self.to_interval()
        if len(intervals) <= 4:
            ivs = ", ".join(f"[{r.lo:.3g}, {r.hi:.3g}]" for r in intervals)
        else:
            ivs = ", ".join(f"[{r.lo:.3g}, {r.hi:.3g}]" for r in intervals[:3])
            ivs += f", ... ({len(intervals)} total)"
        return (
            f"AffineBound(d={self.d_output}, n_basis={self.basis.n}, intervals=[{ivs}])"
        )
