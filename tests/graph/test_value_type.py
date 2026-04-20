"""Unit tests for ``Range`` arithmetic and ``NodeValueType``."""

import math

import pytest

from torchwright.graph import NodeValueType, Range
from torchwright.graph.value_type import tightened_with

# --- Range ------------------------------------------------------------


def test_range_default_is_unbounded():
    r = Range()
    assert r.lo == -math.inf
    assert r.hi == math.inf


def test_range_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Range(1.0, 0.0)


def test_range_point():
    r = Range.point(3.0)
    assert r.lo == 3.0 and r.hi == 3.0


def test_range_add():
    a = Range(0.0, 1.0)
    b = Range(2.0, 5.0)
    assert a + b == Range(2.0, 6.0)


def test_range_neg_and_sub():
    a = Range(1.0, 3.0)
    assert -a == Range(-3.0, -1.0)
    assert a - Range(0.0, 1.0) == Range(0.0, 3.0)


def test_range_union_and_intersect():
    a = Range(0.0, 2.0)
    b = Range(1.0, 3.0)
    assert a.union(b) == Range(0.0, 3.0)
    assert a.intersect(b) == Range(1.0, 2.0)


def test_range_relu_clamps_negatives_to_zero():
    assert Range(-2.0, 3.0).relu() == Range(0.0, 3.0)
    assert Range(-5.0, -1.0).relu() == Range(0.0, 0.0)
    assert Range(2.0, 5.0).relu() == Range(2.0, 5.0)


def test_range_contains():
    outer = Range(0.0, 10.0)
    assert outer.contains(Range(1.0, 5.0))
    assert not outer.contains(Range(-1.0, 5.0))


# --- NodeValueType ---------------------------------------------------


def test_unknown_has_unbounded_range():
    t = NodeValueType.unknown()
    assert t.value_range == Range.unbounded()


def test_bounded_factory():
    t = NodeValueType.bounded(0.0, 9.0)
    assert t.value_range == Range(0.0, 9.0)


# --- Combinators ------------------------------------------------------


def test_tightened_with_intersects_ranges():
    a = NodeValueType.bounded(-5.0, 5.0)
    b = NodeValueType.bounded(0.0, 3.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 3.0)


def test_tightened_with_unbounded_and_bounded():
    a = NodeValueType.unknown()
    b = NodeValueType.bounded(0.0, 9.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 9.0)


def test_tightened_with_both_same_range():
    a = NodeValueType.bounded(0.0, 9.0)
    b = NodeValueType.bounded(0.0, 9.0)
    m = tightened_with(a, b)
    assert m.value_range == Range(0.0, 9.0)
