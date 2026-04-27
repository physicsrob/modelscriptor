"""Tests for the std stdlib factories and type_switch."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import (
    clamp,
    compare_const,
    constant,
    multiply,
    piecewise_linear,
    piecewise_linear_2d,
    relu,
    type_switch,
)
from doom_sandbox.api.vec import _make_vec
from doom_sandbox.runtime.noise import NOISE_REL


def _exact_atol(values):
    return NOISE_REL * (float(np.abs(np.asarray(values)).max()) + 1.0) * 6


# --- type_switch ---

def test_type_switch_picks_active_branch():
    on = constant(1.0)
    off = constant(0.0)
    out = type_switch(
        (on, constant([10.0, 20.0])),
        (off, constant([99.0, 99.0])),
        (off, constant([88.0, 88.0])),
    )
    assert out.shape == 2
    assert np.allclose(out._data, [10.0, 20.0], rtol=0, atol=_exact_atol([10.0, 20.0]))


def test_type_switch_all_zero_masks_returns_zero():
    off = constant(0.0)
    out = type_switch(
        (off, constant([1.0, 2.0])),
        (off, constant([3.0, 4.0])),
    )
    assert np.allclose(out._data, [0.0, 0.0], atol=NOISE_REL * 50)


def test_type_switch_blends_when_multiple_masks_partial():
    """type_switch is a linear sum of mask*value; partial masks blend."""
    half = _make_vec(np.array([0.5]), depth=0)
    out = type_switch(
        (half, constant([10.0])),
        (half, constant([20.0])),
    )
    # 0.5*10 + 0.5*20 = 15
    assert np.allclose(out._data, [15.0], rtol=0, atol=_exact_atol([15.0]))


def test_type_switch_depth_is_max_inputs_plus_1():
    mask = _make_vec(np.array([1.0]), depth=3)
    val = _make_vec(np.array([5.0, 7.0]), depth=8)
    other = _make_vec(np.array([0.0, 0.0]), depth=2)
    out = type_switch((mask, val), (mask, other))
    assert out.depth == 9


def test_type_switch_zero_branches_raises():
    with pytest.raises(ValueError, match="at least one branch"):
        type_switch()


def test_type_switch_mask_must_be_1_shape():
    bad_mask = constant([1.0, 0.0])  # 2-shape
    with pytest.raises(ValueError, match="mask must be 1-shape"):
        type_switch((bad_mask, constant([10.0, 20.0])))


def test_type_switch_value_shape_mismatch_raises():
    on = constant(1.0)
    off = constant(0.0)
    with pytest.raises(ValueError, match="all branches must agree"):
        type_switch(
            (on, constant([10.0, 20.0])),
            (off, constant([1.0])),
        )


# --- relu ---

def test_relu_exact_at_kink():
    """ReLU has a kink at 0; with the breakpoint placed at 0, the PWL
    reproduces ReLU exactly across the range."""
    r = relu((-2.0, 3.0))
    xs = np.array([-2.0, -1.5, -0.7, 0.0, 0.5, 1.7, 3.0])
    expected = np.maximum(0.0, xs)
    out = r(_make_vec(xs, depth=0))
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_relu_clamps_outside_range():
    r = relu((-1.0, 1.0))
    xs = np.array([-100.0, 100.0])
    out = r(_make_vec(xs, depth=0))
    # -100 clamps to -1 → ReLU = 0; 100 clamps to 1 → ReLU = 1.
    assert np.allclose(out._data, [0.0, 1.0], rtol=0, atol=_exact_atol([1.0]))


def test_relu_input_range_must_contain_zero():
    with pytest.raises(ValueError, match="must contain 0"):
        relu((1.0, 5.0))
    with pytest.raises(ValueError, match="must contain 0"):
        relu((-5.0, -1.0))


def test_relu_at_lo_zero_is_identity_on_range():
    """When lo == 0, ReLU is identity over the whole range — 2 breakpoints suffice."""
    r = relu((0.0, 5.0))
    xs = np.array([0.0, 1.5, 5.0])
    out = r(_make_vec(xs, depth=0))
    assert np.allclose(out._data, xs, rtol=0, atol=_exact_atol(xs))


def test_relu_at_hi_zero_is_zero_on_range():
    """When hi == 0, ReLU is zero over the whole range."""
    r = relu((-5.0, 0.0))
    xs = np.array([-5.0, -1.5, 0.0])
    out = r(_make_vec(xs, depth=0))
    assert np.allclose(out._data, [0.0, 0.0, 0.0], atol=NOISE_REL * 50)


# --- clamp ---

def test_clamp_identity_in_range_clipped_outside():
    c = clamp(-1.0, 1.0)
    xs = np.array([-5.0, -0.5, 0.0, 0.7, 5.0])
    out = c(_make_vec(xs, depth=0))
    expected = np.clip(xs, -1.0, 1.0)
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


# --- compare_const ---

def test_compare_const_zero_below_one_above():
    cmp_ = compare_const(c=0.5, input_range=(-1.0, 1.0))
    xs = np.array([-1.0, -0.5, 0.0, 0.4, 0.6, 0.9, 1.0])
    out = cmp_(_make_vec(xs, depth=0))
    # x=0.4 is below the deadband (deadband = 0.5 ± 0.002) → 0; x=0.6 above → 1.
    expected_low = out._data[xs < 0.498]
    expected_high = out._data[xs > 0.502]
    assert np.allclose(expected_low, 0.0, atol=NOISE_REL * 50)
    assert np.allclose(expected_high, 1.0, atol=NOISE_REL * 50)


def test_compare_const_c_outside_range_raises():
    with pytest.raises(ValueError, match="lo < c < hi"):
        compare_const(c=2.0, input_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="lo < c < hi"):
        compare_const(c=-1.0, input_range=(-1.0, 1.0))


def test_compare_const_c_too_close_to_boundary_raises():
    with pytest.raises(ValueError, match="deadband"):
        # eps = 0.001 * 2.0 = 0.002; c = -0.999 puts c-eps = -1.001 outside lo.
        compare_const(c=-0.999, input_range=(-1.0, 1.0))


def test_compare_const_inverted_range_raises():
    with pytest.raises(ValueError, match="lo < hi"):
        compare_const(c=0.0, input_range=(1.0, -1.0))


# --- piecewise_linear (alias) ---

def test_piecewise_linear_is_pwl_def_alias():
    pl = piecewise_linear(lambda x: x * x, breakpoints=64, input_range=(-2.0, 2.0))
    out = pl(_make_vec(np.array([1.5]), depth=0))
    # Same nonlinear approximation as pwl_def — within interp residual.
    assert abs(out._data[0] - 2.25) < 0.05


# --- multiply ---

def test_multiply_factory_returns_bilinear_exact_op():
    mul = multiply(input_range=((-2.0, 2.0), (-3.0, 3.0)))  # default breakpoints=2
    a = _make_vec(np.array([1.0, -1.5, 0.0]), depth=0)
    b = _make_vec(np.array([2.0, 0.5, -2.0]), depth=0)
    out = mul(a, b)
    expected = a._data * b._data
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_multiply_depth_is_max_plus_1():
    mul = multiply(input_range=((-1.0, 1.0), (-1.0, 1.0)))
    a = _make_vec(np.array([0.5]), depth=2)
    b = _make_vec(np.array([0.5]), depth=7)
    out = mul(a, b)
    assert out.depth == 8


# --- piecewise_linear_2d ---

def test_piecewise_linear_2d_alias():
    f = piecewise_linear_2d(
        lambda a, b: a + b,
        breakpoints=2,
        input_range=((-1.0, 1.0), (-1.0, 1.0)),
    )
    a = _make_vec(np.array([0.3, -0.5]), depth=0)
    b = _make_vec(np.array([0.4, 0.2]), depth=0)
    out = f(a, b)
    expected = a._data + b._data
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))
