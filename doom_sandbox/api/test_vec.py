"""Tests for Vec and constant()."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from doom_sandbox.api import Vec, constant
from doom_sandbox.api import _runtime
from doom_sandbox.api.vec import _make_vec


def test_constant_scalar_makes_1_vec():
    v = constant(3.5)
    assert v.shape == 1
    assert v.depth == 0
    assert v._data.tolist() == [3.5]


def test_constant_sequence_carries_values():
    v = constant([0.0, 1.0, 2.0])
    assert v.shape == 3
    assert v.depth == 0
    assert v._data.tolist() == [0.0, 1.0, 2.0]


def test_constant_negative_scalar():
    v = constant(-1000.0)
    assert v.shape == 1
    assert v._data.tolist() == [-1000.0]


def test_constant_empty_sequence_raises():
    with pytest.raises(ValueError, match="at least one"):
        constant([])


def test_constant_2d_sequence_raises():
    with pytest.raises(ValueError, match="1-D"):
        constant([[1.0, 2.0], [3.0, 4.0]])


def test_constant_during_forward_raises():
    _runtime._FORWARD_RUNNING = True
    try:
        with pytest.raises(RuntimeError, match="module load"):
            constant(1.0)
    finally:
        _runtime._FORWARD_RUNNING = False


def test_vec_repr_shows_shape_and_depth():
    v = constant([1.0, 2.0, 3.0])
    assert repr(v) == "Vec(shape=3, depth=0)"


def test_vec_is_frozen():
    v = constant(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.depth = 5  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v._data = np.array([99.0])  # type: ignore[misc]


def test_constant_int_input_coerces_to_float64():
    v = constant(3)
    assert v._data.dtype == np.float64
    assert v._data.tolist() == [3.0]


def test_constant_int_sequence_coerces_to_float64():
    v = constant([1, 2, 3])
    assert v._data.dtype == np.float64
    assert v._data.tolist() == [1.0, 2.0, 3.0]


def test_make_vec_carries_positive_depth():
    v = _make_vec(np.array([1.0, 2.0]), depth=7)
    assert v.depth == 7
    assert v.shape == 2
    assert v._data.dtype == np.float64
    assert v._data.tolist() == [1.0, 2.0]


def test_make_vec_coerces_non_contiguous_input():
    """Strided slice goes in; contiguous float64 comes out."""
    src = np.arange(10, dtype=np.float64)[::2]  # non-contiguous view
    assert not src.flags["C_CONTIGUOUS"]
    v = _make_vec(src, depth=0)
    assert v._data.flags["C_CONTIGUOUS"]
    assert v._data.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_make_vec_rejects_2d():
    with pytest.raises(ValueError, match="1-D"):
        _make_vec(np.zeros((2, 2)), depth=0)


def test_make_vec_rejects_negative_depth():
    with pytest.raises(ValueError, match="depth"):
        _make_vec(np.zeros(3), depth=-1)
