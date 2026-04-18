"""Regression gate for affine bound tightening.

Asserts that affine-derived intervals at bound-consumer sites never
*loosen* relative to a committed snapshot — they may tighten (which is
the whole point) but must never grow.  This catches accidental
regressions in the propagation rules.

The small-graph tests here run fast and verify the tightening contract
on representative subgraphs.  A full DOOM-graph snapshot would be
committed as ``docs/affine_bounds_snapshot.json`` and verified against
the DOOM render graph fixture — deferred until Modal GPU tests exercise
the full pipeline.
"""

import json
import os

import pytest
import torch

from torchwright.graph import (
    Add,
    Concatenate,
    InputNode,
    Linear,
    LiteralValue,
    ReLU,
    finalize,
)
from torchwright.graph.session import fresh_graph_session


class TestTighteningContract:
    """Affine bounds must be at least as tight as eager scalar bounds."""

    def test_linear_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            W = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
            lin = Linear(x, W)
            eager = lin._value_type_eager.value_range
            finalize(lin)
            affine = lin.value_type.value_range
            assert affine.lo >= eager.lo - 1e-10
            assert affine.hi <= eager.hi + 1e-10

    def test_add_cancel_tighter_than_eager(self):
        """x + (-x) = 0: affine sees [0,0], eager sees [-2,2]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            neg = Linear(x, torch.tensor([[-1.0]]))
            s = Add(x, neg)
            eager = s._value_type_eager.value_range
            finalize(s)
            affine = s.value_type.value_range
            assert affine.lo >= eager.lo - 1e-10
            assert affine.hi <= eager.hi + 1e-10
            assert affine.lo == pytest.approx(0.0)
            assert affine.hi == pytest.approx(0.0)

    def test_relu_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 3.0))
            r = ReLU(x)
            eager = r._value_type_eager.value_range
            finalize(r)
            affine = r.value_type.value_range
            assert affine.lo >= eager.lo - 1e-10
            assert affine.hi <= eager.hi + 1e-10

    def test_concat_tighter_than_eager(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(0.0, 5.0))
            c = Concatenate([x, y])
            eager = c._value_type_eager.value_range
            finalize(c)
            affine = c.value_type.value_range
            assert affine.lo >= eager.lo - 1e-10
            assert affine.hi <= eager.hi + 1e-10


class TestCancellationTightening:
    """The canonical use case: offset cancellation in cond_gate-style patterns."""

    def test_add_offset_then_subtract(self):
        """x + M + (y - M) should give range(x) + range(y), not range(x) + range(y) + 2*M."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(-1.0, 1.0))
            M = 100.0
            offset = LiteralValue(torch.tensor([M]))
            neg_offset = LiteralValue(torch.tensor([-M]))
            x_plus_M = Add(x, offset)
            y_minus_M = Add(y, neg_offset)
            total = Add(x_plus_M, y_minus_M)
            finalize(total)
            r = total.value_type.value_range
            assert r.lo == pytest.approx(-2.0)
            assert r.hi == pytest.approx(2.0)

    def test_scale_then_cancel(self):
        """2x + (-2x) = 0 regardless of offset magnitude."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-50.0, 50.0))
            scaled = Linear(x, torch.tensor([[2.0]]))
            neg_scaled = Linear(x, torch.tensor([[-2.0]]))
            s = Add(scaled, neg_scaled)
            finalize(s)
            r = s.value_type.value_range
            assert r.lo == pytest.approx(0.0)
            assert r.hi == pytest.approx(0.0)
