"""Tests for the std stdlib factories and type_switch."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import (
    clamp,
    compare_const,
    constant,
    linear,
    multiply,
    one_hot,
    piecewise_linear,
    piecewise_linear_2d,
    reduce_sum,
    relu,
    sum,
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


# --- linear ---

def test_linear_identity_is_identity():
    v = _make_vec(np.array([1.0, 2.0, 3.0]), depth=0)
    out = linear(v, np.eye(3))
    assert out.shape == 3
    assert np.allclose(out._data, [1.0, 2.0, 3.0], rtol=0, atol=_exact_atol([3.0]))


def test_linear_ones_column_sum_reduces():
    v = _make_vec(np.array([1.0, 2.0, 3.0, 4.0]), depth=0)
    out = linear(v, np.ones((4, 1)))
    assert out.shape == 1
    assert np.allclose(out._data, [10.0], rtol=0, atol=_exact_atol([10.0]))


def test_linear_one_hot_column_picks_element():
    """A one-hot column extracts a single element by index."""
    v = _make_vec(np.array([7.0, 8.0, 9.0]), depth=0)
    pick_1 = np.array([[0.0], [1.0], [0.0]])
    out = linear(v, pick_1)
    assert np.allclose(out._data, [8.0], rtol=0, atol=_exact_atol([8.0]))


def test_linear_arbitrary_projection():
    v = _make_vec(np.array([1.0, 2.0]), depth=0)
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = linear(v, matrix)
    expected = np.array([1.0 * 1 + 2.0 * 4, 1.0 * 2 + 2.0 * 5, 1.0 * 3 + 2.0 * 6])
    assert out.shape == 3
    assert np.allclose(out._data, expected, rtol=0, atol=_exact_atol(expected))


def test_linear_accepts_nested_list():
    v = _make_vec(np.array([1.0, 1.0]), depth=0)
    out = linear(v, [[1.0], [1.0]])
    assert np.allclose(out._data, [2.0], rtol=0, atol=_exact_atol([2.0]))


def test_linear_depth_is_input_plus_1():
    v = _make_vec(np.array([1.0, 2.0]), depth=5)
    out = linear(v, np.ones((2, 3)))
    assert out.depth == 6


def test_linear_rejects_1d_matrix():
    v = _make_vec(np.array([1.0, 2.0]), depth=0)
    with pytest.raises(ValueError, match="must be 2-D"):
        linear(v, np.array([1.0, 1.0]))


def test_linear_rejects_shape_mismatch():
    v = _make_vec(np.array([1.0, 2.0]), depth=0)
    with pytest.raises(ValueError, match="does not match input width"):
        linear(v, np.ones((3, 1)))  # vec.shape=2, matrix expects 3


# --- sum ---

def test_sum_two_vecs_pointwise():
    a = _make_vec(np.array([1.0, 2.0, 3.0]), depth=0)
    b = _make_vec(np.array([10.0, 20.0, 30.0]), depth=0)
    out = sum(a, b)
    assert out.shape == 3
    assert np.allclose(out._data, [11.0, 22.0, 33.0], rtol=0, atol=_exact_atol([33.0]))


def test_sum_n_vecs_pointwise():
    vecs = [_make_vec(np.array([1.0, 2.0]), depth=0) for _ in range(5)]
    out = sum(*vecs)
    assert np.allclose(out._data, [5.0, 10.0], rtol=0, atol=_exact_atol([10.0]))


def test_sum_single_vec_is_passthrough_with_depth_increment():
    v = _make_vec(np.array([1.0, 2.0, 3.0]), depth=4)
    out = sum(v)
    assert np.allclose(out._data, [1.0, 2.0, 3.0], rtol=0, atol=_exact_atol([3.0]))
    assert out.depth == 5


def test_sum_depth_is_max_input_plus_1():
    a = _make_vec(np.array([1.0]), depth=2)
    b = _make_vec(np.array([1.0]), depth=7)
    c = _make_vec(np.array([1.0]), depth=4)
    out = sum(a, b, c)
    assert out.depth == 8


def test_sum_no_inputs_raises():
    with pytest.raises(ValueError, match="at least one Vec"):
        sum()


def test_sum_shape_mismatch_raises():
    a = _make_vec(np.array([1.0, 2.0]), depth=0)
    b = _make_vec(np.array([1.0, 2.0, 3.0]), depth=0)
    with pytest.raises(ValueError, match="all inputs must agree"):
        sum(a, b)


# --- reduce_sum ---

def test_reduce_sum_collapses_to_1_vec():
    v = _make_vec(np.array([1.0, 2.0, 3.0, 4.0]), depth=0)
    out = reduce_sum(v)
    assert out.shape == 1
    assert np.allclose(out._data, [10.0], rtol=0, atol=_exact_atol([10.0]))


def test_reduce_sum_of_1_vec_is_value_with_depth_increment():
    v = _make_vec(np.array([42.0]), depth=3)
    out = reduce_sum(v)
    assert out.shape == 1
    assert np.allclose(out._data, [42.0], rtol=0, atol=_exact_atol([42.0]))
    assert out.depth == 4


def test_reduce_sum_depth_is_input_plus_1():
    v = _make_vec(np.array([1.0, 2.0]), depth=11)
    out = reduce_sum(v)
    assert out.depth == 12


def test_reduce_sum_with_multiply_is_dot_product():
    """The canonical dot-product chain: multiply two same-shape Vecs
    and reduce_sum the result."""
    a = _make_vec(np.array([1.0, 2.0, 3.0]), depth=0)
    b = _make_vec(np.array([4.0, 5.0, 6.0]), depth=0)
    mul = multiply(((-10.0, 10.0), (-10.0, 10.0)))
    products = mul(a, b)
    out = reduce_sum(products)
    # Dot product: 1*4 + 2*5 + 3*6 = 32. Loose atol because multiply
    # is bilinear PWL with breakpoints=2 at this range — exact for
    # bilinear inputs but accumulates noise across two ops.
    assert np.allclose(out._data, [32.0], rtol=0, atol=_exact_atol([32.0]) * 2)


# --- one_hot ---

def test_one_hot_integer_input_is_clean():
    """At any integer idx in [0, n), exactly slot idx is 1.0 and the rest are 0.0."""
    for idx_int in range(8):
        idx = _make_vec(np.array([float(idx_int)]), depth=0)
        out = one_hot(idx, n=8)
        assert out.shape == 8
        expected = np.zeros(8)
        expected[idx_int] = 1.0
        assert np.allclose(
            out._data, expected, rtol=0, atol=_exact_atol(expected)
        )


def test_one_hot_half_integer_blends_5050():
    """At idx = k + 0.5, slots k and k+1 each carry 0.5 (trapezoid edges meet)."""
    idx = _make_vec(np.array([2.5]), depth=0)
    out = one_hot(idx, n=8)
    assert np.isclose(out._data[2], 0.5, atol=_exact_atol([0.5]) + 0.01)
    assert np.isclose(out._data[3], 0.5, atol=_exact_atol([0.5]) + 0.01)
    for k in [0, 1, 4, 5, 6, 7]:
        assert np.isclose(out._data[k], 0.0, atol=_exact_atol([0.5]) + 0.01)


def test_one_hot_quarter_offset_stays_on_plateau():
    """At idx = k + 0.25, we're inside the plateau (|d|=0.25 < 0.45) so slot k stays 1.0."""
    idx = _make_vec(np.array([3.25]), depth=0)
    out = one_hot(idx, n=8)
    assert np.isclose(out._data[3], 1.0, atol=_exact_atol([1.0]) + 0.01)
    assert np.isclose(out._data[4], 0.0, atol=_exact_atol([1.0]) + 0.01)


def test_one_hot_inside_transition_zone_ramps_linearly():
    """At idx = k + 0.5, both slots are 0.5; at idx = k + 0.45, slot k is 1.0
    and slot k+1 is 0.0 (just outside its plateau toward k); at idx = k + 0.55,
    slot k is 0.0 and slot k+1 is 1.0."""
    out_45 = one_hot(_make_vec(np.array([2.45]), depth=0), n=8)
    assert np.isclose(out_45._data[2], 1.0, atol=0.01)
    assert np.isclose(out_45._data[3], 0.0, atol=0.01)
    out_55 = one_hot(_make_vec(np.array([2.55]), depth=0), n=8)
    assert np.isclose(out_55._data[2], 0.0, atol=0.01)
    assert np.isclose(out_55._data[3], 1.0, atol=0.01)


def test_one_hot_clamps_negative_input_to_slot_0():
    idx = _make_vec(np.array([-3.0]), depth=0)
    out = one_hot(idx, n=8)
    expected = np.zeros(8)
    expected[0] = 1.0
    assert np.allclose(out._data, expected, atol=_exact_atol(expected))


def test_one_hot_clamps_overflow_input_to_last_slot():
    idx = _make_vec(np.array([99.0]), depth=0)
    out = one_hot(idx, n=8)
    expected = np.zeros(8)
    expected[7] = 1.0
    assert np.allclose(out._data, expected, atol=_exact_atol(expected))


def test_one_hot_n_1_always_returns_1_vec_with_value_1():
    idx = _make_vec(np.array([0.0]), depth=0)
    out = one_hot(idx, n=1)
    assert out.shape == 1
    assert np.isclose(out._data[0], 1.0, atol=_exact_atol([1.0]))


def test_one_hot_depth_is_input_plus_1():
    idx = _make_vec(np.array([2.0]), depth=7)
    out = one_hot(idx, n=4)
    assert out.depth == 8


def test_one_hot_rejects_non_1_shape_input():
    with pytest.raises(ValueError, match="1-shape Vec"):
        one_hot(_make_vec(np.array([0.0, 1.0]), depth=0), n=4)


def test_one_hot_rejects_non_positive_n():
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        one_hot(_make_vec(np.array([0.0]), depth=0), n=0)


def test_one_hot_pairs_with_multiply_and_reduce_sum_for_indexed_extract():
    """Canonical use: one_hot turns a slot value into a mask; multiply
    elementwise, reduce_sum collapses to the indexed value."""
    values = _make_vec(np.array([10.0, 20.0, 30.0, 40.0]), depth=0)
    idx = _make_vec(np.array([2.0]), depth=0)
    mask = one_hot(idx, n=4)
    mul = multiply(((-100.0, 100.0), (-2.0, 2.0)))
    masked = mul(values, mask)
    picked = reduce_sum(masked)
    assert np.isclose(picked._data[0], 30.0, atol=_exact_atol([30.0]) + 0.01)


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
