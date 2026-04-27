"""Tests for PWLDef and PWLDef2D."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import constant, pwl_def
from doom_sandbox.api import _runtime
from doom_sandbox.api.pwl import PWLDef2D
from doom_sandbox.api.vec import _make_vec
from doom_sandbox.runtime.noise import NOISE_REL


def _exact_atol(values):
    """Tolerance for "should be exact except for FP noise" comparisons.

    NOISE_REL is 1σ; ×6 gives a wide enough envelope that the test
    won't flake even at ~10 samples, while still catching a noise
    model that's off by >6×. The +1 floors the tolerance for
    near-zero values.
    """
    import numpy as _np
    return NOISE_REL * (float(_np.abs(values).max()) + 1.0) * 6


# --- PWLDef construction-time validation ---

def test_pwl_def_during_forward_raises():
    _runtime._FORWARD_RUNNING = True
    try:
        with pytest.raises(RuntimeError, match="module load"):
            pwl_def(lambda x: x, breakpoints=4, input_range=(-1.0, 1.0))
    finally:
        _runtime._FORWARD_RUNNING = False


@pytest.mark.parametrize("bp", [1, 0, -3, 1025, 2048])
def test_pwl_def_breakpoints_out_of_range_raises(bp):
    with pytest.raises(ValueError, match="breakpoints"):
        pwl_def(lambda x: x, breakpoints=bp, input_range=(-1.0, 1.0))


def test_pwl_def_inverted_input_range_raises():
    with pytest.raises(ValueError, match="lo < hi"):
        pwl_def(lambda x: x, breakpoints=4, input_range=(1.0, -1.0))


def test_pwl_def_equal_input_range_raises():
    with pytest.raises(ValueError, match="lo < hi"):
        pwl_def(lambda x: x, breakpoints=4, input_range=(1.0, 1.0))


# --- PWLDef call: shape, depth, basic correctness ---

def test_pwl_call_preserves_shape():
    sq = pwl_def(lambda x: x * x, breakpoints=64, input_range=(-2.0, 2.0))
    v = constant([0.0, 1.0, -1.0, 0.5])
    out = sq(v)
    assert out.shape == v.shape


def test_pwl_call_increments_depth_by_1():
    sq = pwl_def(lambda x: x * x, breakpoints=64, input_range=(-2.0, 2.0))
    v = _make_vec(np.array([0.5]), depth=4)
    out = sq(v)
    assert out.depth == 5


# --- Affine functions are exact (modulo small relative noise) ---

def test_affine_exact_with_2_breakpoints():
    """f(x) = 2x + 3 sampled at 2 points reproduces exactly under interp."""
    f = pwl_def(lambda x: 2.0 * x + 3.0, breakpoints=2, input_range=(-5.0, 5.0))
    xs = np.array([-5.0, -2.0, 0.0, 1.7, 5.0])
    out = f(_make_vec(xs, depth=0))
    expected = 2.0 * xs + 3.0
    # Only the relative gaussian noise should be present (~1e-6 * |value|).
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_affine_exact_with_many_breakpoints():
    """More breakpoints don't change exactness for an affine function."""
    f = pwl_def(lambda x: -0.5 * x + 2.0, breakpoints=512, input_range=(-10.0, 10.0))
    xs = np.linspace(-10.0, 10.0, 21)
    out = f(_make_vec(xs, depth=0))
    expected = -0.5 * xs + 2.0
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_constant_function_is_exact():
    f = pwl_def(lambda x: 7.0, breakpoints=8, input_range=(-3.0, 3.0))
    out = f(constant([-3.0, 0.0, 1.5]))
    assert np.allclose(out._data, [7.0, 7.0, 7.0], atol=NOISE_REL * 50.0)


# --- Nonlinear functions accumulate the real interp residual ---

def test_nonlinear_pwl_residual_decreases_with_more_breakpoints():
    """Linear interp residual on x*x scales as ~h^2; doubling breakpoints
    cuts max error by ~4x."""
    coarse = pwl_def(lambda x: x * x, breakpoints=8, input_range=(-2.0, 2.0))
    fine = pwl_def(lambda x: x * x, breakpoints=64, input_range=(-2.0, 2.0))
    xs = np.linspace(-2.0, 2.0, 101)
    truth = xs * xs
    coarse_err = np.max(np.abs(coarse(_make_vec(xs, depth=0))._data - truth))
    fine_err = np.max(np.abs(fine(_make_vec(xs, depth=0))._data - truth))
    # 8x more breakpoints -> ~64x less residual; comfortable margin.
    assert fine_err < coarse_err / 10.0


# --- Clamping ---

def test_pwl_clamps_inputs_outside_range():
    """Inputs outside input_range get clamped to the boundary value."""
    f = pwl_def(lambda x: 3.0 * x, breakpoints=2, input_range=(0.0, 1.0))
    xs = np.array([-5.0, 0.0, 0.5, 1.0, 100.0])
    out = f(_make_vec(xs, depth=0))
    expected = np.array([0.0, 0.0, 1.5, 3.0, 3.0])
    assert np.allclose(out._data, expected, atol=NOISE_REL * 50.0)


# --- Determinism of noise ---

def test_pwl_deterministic_within_same_pwldef():
    """Same PWLDef + same input bytes -> identical output."""
    f = pwl_def(lambda x: x * x, breakpoints=32, input_range=(-2.0, 2.0))
    xs = np.array([0.3, -0.7, 1.4])
    a = f(_make_vec(xs, depth=0))
    b = f(_make_vec(xs, depth=0))
    assert np.array_equal(a._data, b._data)


def test_pwl_noise_seed_differs_across_pwldefs():
    """Different PWLDefs with the same fn produce independent noise."""
    f1 = pwl_def(lambda x: x * x, breakpoints=32, input_range=(-2.0, 2.0))
    f2 = pwl_def(lambda x: x * x, breakpoints=32, input_range=(-2.0, 2.0))
    xs = np.array([0.3, -0.7, 1.4, 0.9, -1.5])
    a = f1(_make_vec(xs, depth=0))
    b = f2(_make_vec(xs, depth=0))
    # Same interp result, but additive noise differs.
    assert not np.array_equal(a._data, b._data)


def test_pwl_noise_zero_at_zero_value():
    """Relative noise (sigma = 1e-6 * |value|) means zero values get zero noise."""
    f = pwl_def(lambda x: 0.0, breakpoints=4, input_range=(-1.0, 1.0))
    xs = np.array([-1.0, 0.0, 0.5, 1.0])
    out = f(_make_vec(xs, depth=0))
    assert np.array_equal(out._data, np.zeros(4))


def test_pwl_noise_scales_with_value_magnitude():
    """At magnitude 10000, noise std ~ 0.01; at magnitude 1, noise std ~ 1e-6."""
    f = pwl_def(lambda x: x, breakpoints=2, input_range=(0.0, 1e5))
    big = f(_make_vec(np.full(10000, 1e4), depth=0))
    small = f(_make_vec(np.full(10000, 1.0), depth=0))
    big_dev = np.std(big._data - 1e4)
    small_dev = np.std(small._data - 1.0)
    # Roughly 1e-6 * value for both; ratio should be ~10000x.
    assert 0.5 < big_dev / 1e-2 < 2.0
    assert 0.5 < small_dev / 1e-6 < 2.0


# --- PWLDef2D ---

def test_pwl_2d_construction_validates_breakpoints():
    with pytest.raises(ValueError, match="per-dim breakpoints"):
        PWLDef2D(lambda a, b: a * b, breakpoints=1, input_range=((0.0, 1.0), (0.0, 1.0)))


def test_pwl_2d_tuple_breakpoints_per_axis_lower_bound():
    with pytest.raises(ValueError, match="per-dim breakpoints"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=(1, 4),
            input_range=((0.0, 1.0), (0.0, 1.0)),
        )


def test_pwl_2d_tuple_breakpoints_per_axis_upper_bound():
    with pytest.raises(ValueError, match="per-dim breakpoints"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=(1025, 2),
            input_range=((0.0, 1.0), (0.0, 1.0)),
        )


def test_pwl_2d_breakpoints_wrong_tuple_length_raises():
    with pytest.raises(ValueError, match="2-tuple"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=(2,),
            input_range=((0.0, 1.0), (0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="2-tuple"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=(2, 2, 2),
            input_range=((0.0, 1.0), (0.0, 1.0)),
        )


def test_pwl_2d_inverted_input_range_axis_1_raises():
    with pytest.raises(ValueError, match=r"input_range\[1\]"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=4,
            input_range=((0.0, 1.0), (1.0, 0.0)),
        )


def test_pwl_2d_product_breakpoints_capped_at_1024():
    with pytest.raises(ValueError, match="<= 1024"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=(64, 32),
            input_range=((0.0, 1.0), (0.0, 1.0)),
        )


def test_pwl_2d_inverted_input_range_raises():
    with pytest.raises(ValueError, match="lo < hi"):
        PWLDef2D(
            lambda a, b: a * b,
            breakpoints=4,
            input_range=((1.0, 0.0), (0.0, 1.0)),
        )


def test_pwl_2d_call_shape_mismatch_raises():
    mul = PWLDef2D(lambda a, b: a * b, breakpoints=8, input_range=((-1.0, 1.0), (-1.0, 1.0)))
    a = constant([1.0, 2.0, 3.0])
    b = constant([1.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        mul(a, b)


def test_pwl_2d_depth_is_max_plus_1():
    mul = PWLDef2D(lambda a, b: a * b, breakpoints=8, input_range=((-1.0, 1.0), (-1.0, 1.0)))
    a = _make_vec(np.array([0.5]), depth=3)
    b = _make_vec(np.array([0.5]), depth=7)
    out = mul(a, b)
    assert out.depth == 8


def test_pwl_2d_multi_cell_interpolation_correct():
    """Interior of a multi-cell 2-D grid interpolates correctly. f(x,y) = x*y
    is bilinear, so any breakpoint count reproduces it exactly within the
    relative-noise floor — but a buggy index-into-cell or fx/fy formula would
    only show up away from cell corners."""
    f = PWLDef2D(
        lambda x, y: x * y,
        breakpoints=(8, 8),
        input_range=((0.0, 1.0), (0.0, 1.0)),
    )
    # Off-corner queries: every coordinate sits inside a cell, not on a sample.
    xs = np.array([0.13, 0.37, 0.61, 0.84])
    ys = np.array([0.07, 0.42, 0.55, 0.91])
    out = f(_make_vec(xs, depth=0), _make_vec(ys, depth=0))
    expected = xs * ys
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_pwl_def_breakpoints_1024_accepted():
    """Boundary positive case for the 1024 cap (1-D)."""
    f = pwl_def(lambda x: x, breakpoints=1024, input_range=(0.0, 1.0))
    out = f(_make_vec(np.array([0.5]), depth=0))
    assert out.shape == 1


def test_pwl_2d_breakpoints_product_1024_accepted():
    """Boundary positive case for the per-dim product cap (2-D)."""
    PWLDef2D(
        lambda a, b: a * b,
        breakpoints=(32, 32),
        input_range=((0.0, 1.0), (0.0, 1.0)),
    )


def test_pwl_2d_bilinear_exact_for_bilinear_function():
    """f(x, y) = a*x + b*y + c*x*y + d is the most general bilinear; it's
    reproduced exactly by bilinear interpolation regardless of grid size."""
    f = PWLDef2D(
        lambda x, y: 2.0 * x + 3.0 * y + 0.5 * x * y + 1.0,
        breakpoints=2,
        input_range=((-1.0, 1.0), (-2.0, 2.0)),
    )
    xs = np.array([-1.0, -0.3, 0.0, 0.7, 1.0])
    ys = np.array([-2.0, -1.1, 0.0, 1.5, 2.0])
    out = f(_make_vec(xs, depth=0), _make_vec(ys, depth=0))
    expected = 2.0 * xs + 3.0 * ys + 0.5 * xs * ys + 1.0
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_pwl_2d_clamps_both_axes():
    mul = PWLDef2D(lambda a, b: a * b, breakpoints=8, input_range=((0.0, 1.0), (0.0, 1.0)))
    a = _make_vec(np.array([-3.0, 2.0]), depth=0)
    b = _make_vec(np.array([5.0, -1.0]), depth=0)
    out = mul(a, b)
    # First clamps to (0, 1) -> 0; second clamps to (1, 0) -> 0
    assert np.allclose(out._data, [0.0, 0.0], atol=NOISE_REL * 50.0)


def test_pwl_2d_deterministic():
    f = PWLDef2D(
        lambda a, b: a * b, breakpoints=8, input_range=((-1.0, 1.0), (-1.0, 1.0))
    )
    a = _make_vec(np.array([0.2, 0.5, -0.7]), depth=0)
    b = _make_vec(np.array([0.4, -0.3, 0.9]), depth=0)
    out1 = f(a, b)
    out2 = f(a, b)
    assert np.array_equal(out1._data, out2._data)
