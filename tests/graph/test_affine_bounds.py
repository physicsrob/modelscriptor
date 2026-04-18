"""Unit tests for affine bound propagation: Basis, AffineBound factories,
and to_interval() concretization.
"""

import math

import pytest
import torch

from torchwright.graph import InputNode, LiteralValue, finalize
from torchwright.graph.affine_bound import AffineBound
from torchwright.graph.basis import Basis
from torchwright.graph.session import (
    ValueTypeNotFinalized,
    fresh_graph_session,
)

# --- Basis ---------------------------------------------------------------


class TestBasis:
    def test_from_single_input(self):
        with fresh_graph_session():
            x = InputNode(3, name="x")
            basis = Basis.from_input_nodes([x])
            assert basis.n == 3
            assert basis.index_of(x) == (0, 3)

    def test_from_multiple_inputs(self):
        with fresh_graph_session():
            x = InputNode(3, name="x")
            y = InputNode(2, name="y")
            basis = Basis.from_input_nodes([x, y])
            assert basis.n == 5
            assert basis.index_of(x) == (0, 3)
            assert basis.index_of(y) == (3, 2)

    def test_index_of_missing_raises(self):
        with fresh_graph_session():
            x = InputNode(3, name="x")
            y = InputNode(2, name="y")
            basis = Basis.from_input_nodes([x])
            with pytest.raises(KeyError):
                basis.index_of(y)

    def test_empty_basis(self):
        basis = Basis.from_input_nodes([])
        assert basis.n == 0


# --- AffineBound factories -----------------------------------------------


class TestAffineBoundFactories:
    def test_identity(self):
        with fresh_graph_session():
            x = InputNode(3, name="x", value_range=(-1.0, 1.0))
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.identity(basis, x)
            assert ab.d_output == 3
            assert ab.basis is basis
            assert torch.equal(ab.A_lo, torch.eye(3, dtype=torch.float64))
            assert torch.equal(ab.A_hi, torch.eye(3, dtype=torch.float64))
            assert torch.equal(ab.b_lo, torch.zeros(3, dtype=torch.float64))
            assert torch.equal(ab.b_hi, torch.zeros(3, dtype=torch.float64))

    def test_identity_two_inputs(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            y = InputNode(3, name="y", value_range=(0.0, 5.0))
            basis = Basis.from_input_nodes([x, y])

            ab_x = AffineBound.identity(basis, x)
            assert ab_x.d_output == 2
            expected_A = torch.zeros(2, 5, dtype=torch.float64)
            expected_A[0, 0] = 1.0
            expected_A[1, 1] = 1.0
            assert torch.equal(ab_x.A_lo, expected_A)

            ab_y = AffineBound.identity(basis, y)
            assert ab_y.d_output == 3
            expected_A_y = torch.zeros(3, 5, dtype=torch.float64)
            expected_A_y[0, 2] = 1.0
            expected_A_y[1, 3] = 1.0
            expected_A_y[2, 4] = 1.0
            assert torch.equal(ab_y.A_lo, expected_A_y)

    def test_constant(self):
        with fresh_graph_session():
            x = InputNode(2, name="x")
            basis = Basis.from_input_nodes([x])
            vals = torch.tensor([3.0, 7.0])
            ab = AffineBound.constant(basis, vals)
            assert ab.d_output == 2
            assert torch.equal(ab.A_lo, torch.zeros(2, 2, dtype=torch.float64))
            assert torch.allclose(ab.b_lo, vals.double())
            assert torch.allclose(ab.b_hi, vals.double())

    def test_degenerate(self):
        with fresh_graph_session():
            x = InputNode(2, name="x")
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.degenerate(basis, 4, lo=-5.0, hi=10.0)
            assert ab.d_output == 4
            assert torch.equal(ab.A_lo, torch.zeros(4, 2, dtype=torch.float64))
            assert torch.allclose(ab.b_lo, torch.full((4,), -5.0, dtype=torch.float64))
            assert torch.allclose(ab.b_hi, torch.full((4,), 10.0, dtype=torch.float64))

    def test_degenerate_defaults_to_inf(self):
        basis = Basis.from_input_nodes([])
        ab = AffineBound.degenerate(basis, 2)
        assert ab.b_lo[0].item() == float("-inf")
        assert ab.b_hi[0].item() == float("inf")


# --- to_interval() -------------------------------------------------------


class TestToInterval:
    def test_identity_interval_matches_input_range(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 5.0))
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.identity(basis, x)
            intervals = ab.to_interval()
            assert len(intervals) == 2
            assert intervals[0].lo == pytest.approx(-3.0)
            assert intervals[0].hi == pytest.approx(5.0)
            assert intervals[1].lo == pytest.approx(-3.0)
            assert intervals[1].hi == pytest.approx(5.0)

    def test_constant_interval(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.constant(basis, torch.tensor([3.0, 7.0]))
            intervals = ab.to_interval()
            assert intervals[0].lo == pytest.approx(3.0)
            assert intervals[0].hi == pytest.approx(3.0)
            assert intervals[1].lo == pytest.approx(7.0)
            assert intervals[1].hi == pytest.approx(7.0)

    def test_degenerate_interval(self):
        with fresh_graph_session():
            x = InputNode(2, name="x")
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.degenerate(basis, 1, lo=-5.0, hi=10.0)
            intervals = ab.to_interval()
            assert intervals[0].lo == pytest.approx(-5.0)
            assert intervals[0].hi == pytest.approx(10.0)

    def test_scalar_range_union(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            basis = Basis.from_input_nodes([x])
            ab = AffineBound.constant(basis, torch.tensor([3.0, 7.0]))
            r = ab.to_scalar_range()
            assert r.lo == pytest.approx(3.0)
            assert r.hi == pytest.approx(7.0)

    def test_affine_sum_tight(self):
        """x + (-x) should give interval [0, 0], not [-2, 2]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            basis = Basis.from_input_nodes([x])
            A_lo = torch.tensor([[0.0]], dtype=torch.float64)
            A_hi = torch.tensor([[0.0]], dtype=torch.float64)
            b_lo = torch.tensor([0.0], dtype=torch.float64)
            b_hi = torch.tensor([0.0], dtype=torch.float64)
            ab = AffineBound(A_lo=A_lo, A_hi=A_hi, b_lo=b_lo, b_hi=b_hi, basis=basis)
            intervals = ab.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(0.0)


# --- Session management --------------------------------------------------


class TestSession:
    def test_fresh_session_isolates_inputs(self):
        with fresh_graph_session() as s1:
            x = InputNode(2, name="x")
            assert len(s1.input_nodes) == 1
        with fresh_graph_session() as s2:
            y = InputNode(3, name="y")
            assert len(s2.input_nodes) == 1
            assert s2.input_nodes[0] is y

    def test_nested_session_raises(self):
        with fresh_graph_session():
            with pytest.raises(RuntimeError, match="Nested"):
                with fresh_graph_session():
                    pass

    def test_affine_bound_before_finalize_raises(self):
        with fresh_graph_session():
            x = InputNode(2, name="x")
            with pytest.raises(ValueTypeNotFinalized):
                _ = x.affine_bound


# --- finalize() -----------------------------------------------------------


class TestFinalize:
    def test_finalize_sets_affine_bounds(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            lit = LiteralValue(torch.tensor([3.0, 5.0]))
            from torchwright.graph import Add

            s = Add(x, lit)
            finalize(s)
            assert s._affine_bound is not None
            assert x._affine_bound is not None
            assert lit._affine_bound is not None

    def test_finalize_idempotent(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            finalize(x)
            first = x._affine_bound
            finalize(x)
            assert x._affine_bound is first

    def test_finalize_input_identity(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            finalize(x)
            ab = x.affine_bound
            intervals = ab.to_interval()
            assert intervals[0].lo == pytest.approx(-1.0)
            assert intervals[0].hi == pytest.approx(1.0)

    def test_repr_works(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            finalize(x)
            r = repr(x.affine_bound)
            assert "AffineBound" in r


# --- Exact affine rules (Phase C) ----------------------------------------


class TestLinearRule:
    def test_identity_matrix_preserves_input(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 5.0))
            lin = __import__("torchwright.graph", fromlist=["Linear"]).Linear(
                x, torch.eye(2), name="id"
            )
            finalize(lin)
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-3.0)
            assert intervals[0].hi == pytest.approx(5.0)

    def test_scaling_matrix(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 2.0))
            W = torch.tensor([[3.0]])
            from torchwright.graph import Linear

            lin = Linear(x, W, name="scale")
            finalize(lin)
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(6.0)

    def test_negative_weight(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(1.0, 3.0))
            W = torch.tensor([[-2.0]])
            from torchwright.graph import Linear

            lin = Linear(x, W, name="neg")
            finalize(lin)
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(-6.0)
            assert intervals[0].hi == pytest.approx(-2.0)

    def test_bias(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 1.0))
            W = torch.tensor([[1.0]])
            b = torch.tensor([5.0])
            from torchwright.graph import Linear

            lin = Linear(x, W, b, name="bias")
            finalize(lin)
            intervals = lin.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(5.0)
            assert intervals[0].hi == pytest.approx(6.0)


class TestAddRule:
    def test_add_tracks_correlation(self):
        """x + (-x) should give [0, 0] via affine tracking, not [-2, 2]."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            from torchwright.graph import Linear, Add

            neg_x = Linear(x, torch.tensor([[-1.0]]), name="neg")
            s = Add(x, neg_x, name="cancel")
            finalize(s)
            intervals = s.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(0.0)

    def test_add_independent_inputs(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(0.0, 1.0))
            y = InputNode(1, name="y", value_range=(2.0, 3.0))
            from torchwright.graph import Add

            s = Add(x, y, name="sum")
            finalize(s)
            intervals = s.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(2.0)
            assert intervals[0].hi == pytest.approx(4.0)


class TestConcatRule:
    def test_concat_stacks_bounds(self):
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-1.0, 1.0))
            y = InputNode(1, name="y", value_range=(0.0, 5.0))
            from torchwright.graph import Concatenate

            c = Concatenate([x, y])
            finalize(c)
            intervals = c.affine_bound.to_interval()
            assert len(intervals) == 3
            assert intervals[0].lo == pytest.approx(-1.0)
            assert intervals[0].hi == pytest.approx(1.0)
            assert intervals[2].lo == pytest.approx(0.0)
            assert intervals[2].hi == pytest.approx(5.0)


class TestLiteralRule:
    def test_literal_constant_bound(self):
        with fresh_graph_session():
            x = InputNode(1, name="x")
            lit = LiteralValue(torch.tensor([3.0, 7.0]))
            finalize(lit)
            intervals = lit.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(3.0)
            assert intervals[0].hi == pytest.approx(3.0)
            assert intervals[1].lo == pytest.approx(7.0)
            assert intervals[1].hi == pytest.approx(7.0)


class TestDualRail:
    def test_value_type_affine_tightens_range(self):
        """Affine bounds should tighten value_type range after finalize."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-1.0, 1.0))
            from torchwright.graph import Linear, Add

            neg_x = Linear(x, torch.tensor([[-1.0]]), name="neg")
            s = Add(x, neg_x, name="cancel")
            # Before finalize, eager range is [-2, 2]
            eager_range = s._value_type_eager.value_range
            assert eager_range.lo == pytest.approx(-2.0)
            assert eager_range.hi == pytest.approx(2.0)
            finalize(s)
            # After finalize, affine-derived range is [0, 0]
            assert s.value_type.value_range.lo == pytest.approx(0.0)
            assert s.value_type.value_range.hi == pytest.approx(0.0)

    def test_value_type_preserves_structural_flags(self):
        with fresh_graph_session():
            lit = LiteralValue(torch.tensor([0.0, 1.0]))
            finalize(lit)
            assert lit.value_type.is_binary is True
            assert lit.value_type.is_integer is True


class TestSoundness:
    """Randomized checks: sample from the basis box and verify actual <= bound."""

    def test_linear_soundness(self):
        import random

        random.seed(42)
        with fresh_graph_session():
            x = InputNode(2, name="x", value_range=(-3.0, 3.0))
            W = torch.randn(2, 3)
            b = torch.randn(3)
            from torchwright.graph import Linear

            lin = Linear(x, W, b, name="test")
            finalize(lin)
            intervals = lin.affine_bound.to_interval()
            for _ in range(100):
                xv = torch.FloatTensor(1, 2).uniform_(-3.0, 3.0)
                y = (xv @ W + b).squeeze(0)
                for j in range(3):
                    assert y[j].item() >= intervals[j].lo - 1e-5
                    assert y[j].item() <= intervals[j].hi + 1e-5

    def test_add_cancel_soundness(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-10.0, 10.0))
            from torchwright.graph import Linear, Add

            neg = Linear(x, torch.tensor([[-1.0]]))
            s = Add(x, neg)
            finalize(s)
            r = s.affine_bound.to_interval()[0]
            assert r.lo == pytest.approx(0.0)
            assert r.hi == pytest.approx(0.0)


# --- ReLU envelope (Phase D) ---------------------------------------------


class TestReluRule:
    def test_relu_identity_positive(self):
        """When input is fully positive, ReLU is identity."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(1.0, 3.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            finalize(r)
            intervals = r.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(1.0)
            assert intervals[0].hi == pytest.approx(3.0)

    def test_relu_zero_negative(self):
        """When input is fully negative, ReLU output is 0."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-5.0, -1.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            finalize(r)
            intervals = r.affine_bound.to_interval()
            assert intervals[0].lo == pytest.approx(0.0)
            assert intervals[0].hi == pytest.approx(0.0)

    def test_relu_straddling(self):
        """Straddling case uses linear envelope."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-2.0, 4.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            finalize(r)
            intervals = r.affine_bound.to_interval()
            # Lower bound: alpha=1.0 (h=4 >= -l=2), so lower = max(0, x) >= x
            # Lower bound at x=-2: max(0, -2) = 0; but affine lower bound = alpha * x = -2
            # So the affine lower is clamped: actually lo = alpha * lo = -2. But relu floor = 0.
            # The affine bound computes lo = alpha*lo = 1.0*(-2.0) = -2.0
            # Upper bound: slope = h/(h-l) = 4/6 = 2/3
            # Upper at x=-2: 2/3 * (-2 - (-2)) = 0; at x=4: 2/3 * (4 - (-2)) = 4.0
            assert intervals[0].lo >= -2.0 - 1e-5
            assert intervals[0].hi == pytest.approx(4.0)
            # ReLU output must be in [0, 4], affine captures [lo, 4]
            assert intervals[0].hi <= 4.0 + 1e-5

    def test_relu_soundness(self):
        """Randomized soundness: actual relu values within affine bounds."""
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-3.0, 5.0))
            from torchwright.graph import ReLU

            r = ReLU(x)
            finalize(r)
            intervals = r.affine_bound.to_interval()
            for _ in range(200):
                xv = torch.FloatTensor(1).uniform_(-3.0, 5.0)
                yv = torch.clamp(xv, min=0.0).item()
                assert yv >= intervals[0].lo - 1e-5
                assert yv <= intervals[0].hi + 1e-5


# --- Assert pass-through (Phase D) ----------------------------------------


class TestAssertRule:
    def test_assert_preserves_coefficients(self):
        with fresh_graph_session():
            x = InputNode(1, name="x", value_range=(-5.0, 5.0))
            from torchwright.graph.asserts import assert_in_range

            a = assert_in_range(x, -3.0, 3.0)
            finalize(a)
            # Assert should pass through the input's affine bound
            assert torch.equal(a.affine_bound.A_lo, x.affine_bound.A_lo)
            assert torch.equal(a.affine_bound.A_hi, x.affine_bound.A_hi)
            # But value_type should reflect tightened range
            assert a.value_type.value_range.lo == pytest.approx(-3.0)
            assert a.value_type.value_range.hi == pytest.approx(3.0)


# --- Attn degenerate (Phase E) -------------------------------------------


class TestAttnRule:
    def test_attn_propagates_value_range(self):
        with fresh_graph_session():
            from torchwright.graph import PosEncoding, Attn

            pe = PosEncoding(d_pos=8)
            value = LiteralValue(torch.tensor([2.0, 3.0]))
            attn = Attn(
                query_in=pe,
                key_in=pe,
                value_in=value,
                query_matrix=torch.eye(8, 2),
                key_matrix=torch.eye(8, 2),
                value_matrix=torch.eye(2),
                output_matrix=torch.eye(2),
            )
            finalize(attn)
            r = attn.affine_bound.to_scalar_range()
            assert r.lo <= 2.0 + 1e-5
            assert r.hi >= 3.0 - 1e-5
