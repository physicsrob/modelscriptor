"""Unit tests for ``Range`` arithmetic, ``NodeValueType`` factories, and
their cross-field invariants.
"""

import math

import pytest

from torchwright.graph import NodeValueType, Range
from torchwright.graph.value_type import intersect_element_props


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


# --- NodeValueType factories -----------------------------------------


def test_unknown_has_no_properties():
    t = NodeValueType.unknown()
    assert t.value_range == Range.unbounded()
    assert not t.is_integer
    assert not t.is_binary
    assert not t.is_sign
    assert not t.is_one_hot


def test_integer_factory_defaults_to_unbounded():
    t = NodeValueType.integer()
    assert t.is_integer
    assert t.value_range == Range.unbounded()


def test_integer_factory_with_bounds():
    t = NodeValueType.integer(0, 9)
    assert t.is_integer
    assert t.value_range == Range(0.0, 9.0)


def test_binary_factory():
    t = NodeValueType.binary()
    assert t.is_binary and t.is_integer
    assert t.value_range == Range(0.0, 1.0)


def test_sign_factory():
    t = NodeValueType.sign()
    assert t.is_sign and t.is_integer
    assert t.value_range == Range(-1.0, 1.0)


def test_one_hot_factory_implies_binary():
    t = NodeValueType.one_hot()
    assert t.is_one_hot and t.is_binary and t.is_integer


# --- Invariants -------------------------------------------------------


def test_is_binary_requires_is_integer():
    with pytest.raises(ValueError):
        NodeValueType(value_range=Range(0.0, 1.0), is_binary=True, is_integer=False)


def test_is_binary_requires_range_subset_of_01():
    with pytest.raises(ValueError):
        NodeValueType(value_range=Range(0.0, 2.0), is_integer=True, is_binary=True)


def test_is_sign_requires_range_subset_of_pm1():
    with pytest.raises(ValueError):
        NodeValueType(value_range=Range(-2.0, 1.0), is_integer=True, is_sign=True)


def test_is_one_hot_requires_is_binary():
    with pytest.raises(ValueError):
        NodeValueType(
            value_range=Range(0.0, 1.0),
            is_integer=True,
            is_binary=False,
            is_one_hot=True,
        )


# --- Combinators ------------------------------------------------------


def test_intersect_element_props_keeps_common_properties():
    a = NodeValueType.integer(0, 9)
    b = NodeValueType.integer(-5, 3)
    m = intersect_element_props(a, b)
    assert m.is_integer
    assert m.value_range == Range(-5.0, 9.0)
    assert not m.is_binary
    assert not m.is_one_hot


def test_intersect_drops_mismatched_properties():
    a = NodeValueType.binary()
    b = NodeValueType.integer(0, 9)
    m = intersect_element_props(a, b)
    assert m.is_integer
    assert not m.is_binary
    assert not m.is_one_hot
    assert m.value_range == Range(0.0, 9.0)


def test_drop_vector_props():
    t = NodeValueType.one_hot().drop_vector_props()
    assert t.is_binary
    assert not t.is_one_hot
