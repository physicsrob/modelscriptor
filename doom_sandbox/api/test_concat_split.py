"""Tests for concat() and split()."""

from __future__ import annotations

import pytest

from . import concat, constant, split
from .vec import _make_vec
import numpy as np


def _vec(values, depth=0):
    return _make_vec(np.asarray(values, dtype=np.float64), depth=depth)


def test_concat_single_vec_round_trips():
    a = constant([1.0, 2.0])
    out = concat(a)
    assert out.shape == 2
    assert out.depth == 0
    assert out._data.tolist() == [1.0, 2.0]


def test_concat_single_vec_preserves_nonzero_depth():
    a = _vec([1.0, 2.0], depth=4)
    out = concat(a)
    assert out.depth == 4


def test_concat_multiple_vecs_packs_in_order():
    a = constant([1.0, 2.0])
    b = constant([3.0])
    c = constant([4.0, 5.0, 6.0])
    out = concat(a, b, c)
    assert out.shape == 6
    assert out._data.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_concat_depth_is_max():
    a = _vec([1.0], depth=2)
    b = _vec([2.0], depth=5)
    c = _vec([3.0], depth=1)
    out = concat(a, b, c)
    assert out.depth == 5


def test_concat_depth_with_ties():
    a = _vec([1.0], depth=3)
    b = _vec([2.0], depth=3)
    out = concat(a, b)
    assert out.depth == 3


def test_concat_zero_args_raises():
    with pytest.raises(ValueError, match="at least one"):
        concat()


def test_split_round_trips_concat():
    a = constant([1.0, 2.0])
    b = constant([3.0])
    c = constant([4.0, 5.0, 6.0])
    packed = concat(a, b, c)
    pa, pb, pc = split(packed, [2, 1, 3])
    assert pa._data.tolist() == [1.0, 2.0]
    assert pb._data.tolist() == [3.0]
    assert pc._data.tolist() == [4.0, 5.0, 6.0]


def test_split_carries_parent_depth():
    parent = _vec([1.0, 2.0, 3.0, 4.0], depth=7)
    pieces = split(parent, [2, 2])
    assert all(p.depth == 7 for p in pieces)


def test_split_size_mismatch_too_large_raises():
    v = constant([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sum to"):
        split(v, [2, 2])


def test_split_size_mismatch_too_small_raises():
    v = constant([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sum to"):
        split(v, [1, 1])


def test_split_zero_size_raises():
    v = constant([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        split(v, [3, 0])


def test_split_negative_size_raises():
    v = constant([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        split(v, [4, -1])
