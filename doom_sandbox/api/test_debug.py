"""Tests for debug/assertion primitives."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import (
    assert_,
    assert_bool,
    assert_close,
    assert_in_range,
    assert_integer,
    constant,
    debug_watch,
    print_vec,
)
from doom_sandbox.api.vec import _make_vec


# --- print_vec / debug_watch ---

def test_print_vec_writes_label_and_values(capsys):
    v = constant([1.0, 2.0, 3.0])
    print_vec(v, label="hello")
    captured = capsys.readouterr().out
    assert "hello" in captured
    assert "1.0" in captured
    assert "2.0" in captured


def test_print_vec_no_label(capsys):
    v = constant(7.5)
    print_vec(v)
    captured = capsys.readouterr().out
    assert "7.5" in captured


def test_debug_watch_prints_only_when_predicate_fires(capsys):
    v = constant([5.0])
    debug_watch(v, predicate=lambda d: d[0] > 100, label="big")
    assert capsys.readouterr().out == ""

    debug_watch(v, predicate=lambda d: d[0] > 1, label="small")
    out = capsys.readouterr().out
    assert "small" in out
    assert "5.0" in out


# --- assert_in_range ---

def test_assert_in_range_passes_when_within():
    v = constant([0.0, 0.5, 1.0])
    assert_in_range(v, 0.0, 1.0)  # boundary inclusive


def test_assert_in_range_raises_below():
    v = constant([-0.1, 0.5])
    with pytest.raises(AssertionError, match="span"):
        assert_in_range(v, 0.0, 1.0)


def test_assert_in_range_raises_above():
    v = constant([0.5, 1.1])
    with pytest.raises(AssertionError, match="span"):
        assert_in_range(v, 0.0, 1.0)


# --- assert_close ---

def test_assert_close_passes_within_atol():
    a = constant([1.0, 2.0, 3.0])
    b = constant([1.0001, 1.999, 3.0])
    assert_close(a, b, atol=1e-2)


def test_assert_close_fails_beyond_atol():
    a = constant([1.0])
    b = constant([1.5])
    with pytest.raises(AssertionError, match="max abs diff"):
        assert_close(a, b, atol=1e-3)


def test_assert_close_shape_mismatch_raises():
    a = constant([1.0, 2.0])
    b = constant([1.0])
    with pytest.raises(AssertionError, match="shapes"):
        assert_close(a, b)


# --- assert_bool ---

def test_assert_bool_passes_for_zero_and_one():
    v = constant([0.0, 1.0, 0.0001, 0.9999])
    assert_bool(v, atol=1e-2)


def test_assert_bool_fails_for_mid_value():
    v = constant([0.0, 0.5, 1.0])
    with pytest.raises(AssertionError, match="distance to 0 or 1"):
        assert_bool(v)


# --- assert_integer ---

def test_assert_integer_passes_for_near_integers():
    v = constant([0.0, 1.0001, -3.0, 7.999])
    assert_integer(v, atol=1e-2)


def test_assert_integer_fails_for_fraction():
    v = constant([0.5, 1.0])
    with pytest.raises(AssertionError, match="distance to nearest integer"):
        assert_integer(v)


def test_assert_integer_negative_values():
    v = constant([-2.0001, -3.0])
    assert_integer(v, atol=1e-2)


# --- assert_ ---

def test_assert_passes_when_predicate_true():
    v = constant([1.0, 2.0, 3.0])
    assert_(v, lambda d: d.sum() == 6.0, message="sum mismatch")


def test_assert_raises_with_message():
    v = constant([1.0])
    with pytest.raises(AssertionError, match="custom message"):
        assert_(v, lambda d: False, message="custom message")


def test_assert_default_message_when_unspecified():
    v = constant([1.0])
    with pytest.raises(AssertionError, match="predicate returned False"):
        assert_(v, lambda d: False)


# --- assertions work on framework-internal Vecs (depth >0) ---

def test_assertions_handle_deep_vecs():
    v = _make_vec(np.array([0.0, 1.0]), depth=42)
    assert_bool(v)
    assert_integer(v)
    assert_in_range(v, 0.0, 1.0)
