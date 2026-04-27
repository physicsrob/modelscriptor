"""Tests for make_token, extract_*_slot, is_type."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import (
    FloatSlot,
    IntSlot,
    Token,
    TokenType,
    constant,
    extract_float_slot,
    extract_int_slot,
    is_type,
    make_token,
)
from doom_sandbox.api.vec import _make_vec
from doom_sandbox.runtime.embedding import Layout, active_layout, embed


RENDER = TokenType("RENDER", slots={
    "col": IntSlot(0, 320),
    "chunk": IntSlot(0, 16),
})
VALUE = TokenType("VALUE", slots={
    "v": FloatSlot(-40.0, 40.0, levels=65536),
})
NO_OP = TokenType("NO_OP", slots={})
OTHER = TokenType("OTHER", slots={
    "col": IntSlot(0, 8),  # same name as RENDER.col, different type
})


def _layout(types=None):
    return Layout(types or [RENDER, VALUE, NO_OP])


# --- Active-layout enforcement ---

def test_token_primitives_raise_without_active_layout():
    v = constant(0.0)
    with pytest.raises(RuntimeError, match="No active token layout"):
        is_type(v, RENDER)
    with pytest.raises(RuntimeError, match="No active token layout"):
        extract_int_slot(v, "col")
    with pytest.raises(RuntimeError, match="No active token layout"):
        make_token(RENDER)


# --- make_token basics ---

def test_make_token_writes_type_one_hot():
    layout = _layout()
    with active_layout(layout):
        vec = make_token(RENDER, col=constant(42), chunk=constant(3))
    assert vec.shape == layout.width
    assert vec._data[layout.type_columns["RENDER"]] == 1.0
    assert vec._data[layout.type_columns["VALUE"]] == 0.0
    assert vec._data[layout.slot_columns[("RENDER", "col")]] == 42
    assert vec._data[layout.slot_columns[("RENDER", "chunk")]] == 3


def test_make_token_omitted_slots_default_to_zero():
    layout = _layout()
    with active_layout(layout):
        vec = make_token(RENDER, col=constant(5))
    assert vec._data[layout.slot_columns[("RENDER", "chunk")]] == 0.0


def test_make_token_no_slots_for_no_op_type():
    layout = _layout()
    with active_layout(layout):
        vec = make_token(NO_OP)
    assert vec._data[layout.type_columns["NO_OP"]] == 1.0
    # All other columns zero.
    others = [
        i for i in range(layout.width)
        if i != layout.type_columns["NO_OP"]
    ]
    assert all(vec._data[i] == 0.0 for i in others)


def test_make_token_unknown_slot_raises():
    layout = _layout()
    with active_layout(layout), pytest.raises(ValueError, match="does not declare slot"):
        make_token(RENDER, bogus=constant(1))


def test_make_token_non_unit_slot_raises():
    layout = _layout()
    # Use _make_vec directly so the failure is from make_token's check,
    # not from constant() rejecting the input.
    bad = _make_vec(np.array([1.0, 2.0]), depth=0)
    with active_layout(layout), pytest.raises(ValueError, match="1-shape"):
        make_token(RENDER, col=bad)


def test_make_token_leaves_other_types_columns_zero():
    layout = _layout()
    with active_layout(layout):
        vec = make_token(RENDER, col=constant(42), chunk=constant(3))
    # Sibling-type columns must be untouched.
    assert vec._data[layout.slot_columns[("VALUE", "v")]] == 0.0
    # Sibling type one-hot bits must be 0.
    assert vec._data[layout.type_columns["VALUE"]] == 0.0
    assert vec._data[layout.type_columns["NO_OP"]] == 0.0


def test_make_token_unknown_type_raises():
    layout = _layout([RENDER, VALUE])  # exclude NO_OP
    with active_layout(layout), pytest.raises(ValueError, match="not in the active vocab"):
        make_token(NO_OP)


def test_make_token_depth_max_inputs_plus_1():
    layout = _layout()
    with active_layout(layout):
        col = _make_vec(np.array([10.0]), depth=4)
        chunk = _make_vec(np.array([2.0]), depth=2)
        vec = make_token(RENDER, col=col, chunk=chunk)
    assert vec.depth == 5


def test_make_token_depth_with_no_slots_is_1():
    layout = _layout()
    with active_layout(layout):
        vec = make_token(NO_OP)
    assert vec.depth == 1


# --- make_token round-trips through deembed ---

def test_make_token_round_trips():
    from doom_sandbox.runtime.embedding import deembed
    layout = _layout()
    with active_layout(layout):
        vec = make_token(RENDER, col=constant(123), chunk=constant(7))
    tok = deembed(vec, layout)
    assert tok.type == RENDER
    assert tok.values == {"col": 123, "chunk": 7}


def test_make_token_round_trips_float_slot():
    from doom_sandbox.runtime.embedding import deembed
    layout = _layout()
    with active_layout(layout):
        vec = make_token(VALUE, v=constant(-1.234))
    tok = deembed(vec, layout)
    span = 80.0
    one_lsb = span / (65536 - 1)
    assert tok.type == VALUE
    assert abs(tok.values["v"] - (-1.234)) <= one_lsb / 2 + 1e-9


# --- is_type ---

def test_is_type_returns_1_for_matching_type():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 1, "chunk": 0}), layout)
        m_render = is_type(input_vec, RENDER)
        m_value = is_type(input_vec, VALUE)
    assert m_render._data.tolist() == [1.0]
    assert m_value._data.tolist() == [0.0]


def test_is_type_mutually_exclusive_across_vocab():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(VALUE, {"v": 0.5}), layout)
        masks = [is_type(input_vec, t) for t in (RENDER, VALUE, NO_OP)]
    total = sum(m._data[0] for m in masks)
    assert total == 1.0


def test_is_type_depth_plus_1():
    layout = _layout()
    with active_layout(layout):
        input_vec = _make_vec(np.zeros(layout.width), depth=3)
        m = is_type(input_vec, RENDER)
    assert m.depth == 4


def test_is_type_unknown_type_raises():
    layout = _layout([RENDER, VALUE])  # NO_OP excluded
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 0, "chunk": 0}), layout)
        with pytest.raises(ValueError, match="not in the active vocab"):
            is_type(input_vec, NO_OP)


# --- extract_int_slot ---

def test_extract_int_slot_returns_value_when_type_has_slot():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 99, "chunk": 4}), layout)
        col = extract_int_slot(input_vec, "col")
        chunk = extract_int_slot(input_vec, "chunk")
    assert col._data.tolist() == [99.0]
    assert chunk._data.tolist() == [4.0]


def test_extract_int_slot_returns_zero_when_type_lacks_slot():
    layout = _layout([RENDER, VALUE])
    with active_layout(layout):
        input_vec = embed(Token(VALUE, {"v": 7.0}), layout)
        col = extract_int_slot(input_vec, "col")
    assert col._data.tolist() == [0.0]


def test_extract_int_slot_sums_across_types_with_same_name():
    """When `col` is declared on multiple types, only the current type's
    column is non-zero, so summing recovers the right value."""
    layout = Layout([RENDER, OTHER])
    with active_layout(layout):
        render_vec = embed(Token(RENDER, {"col": 200, "chunk": 5}), layout)
        other_vec = embed(Token(OTHER, {"col": 3}), layout)
        assert extract_int_slot(render_vec, "col")._data.tolist() == [200.0]
        assert extract_int_slot(other_vec, "col")._data.tolist() == [3.0]


def test_extract_int_slot_returns_zero_when_shared_name_not_on_current_type():
    """`col` is declared on RENDER and OTHER, but not on VALUE — extracting
    `col` from a VALUE token still returns 0."""
    layout = Layout([RENDER, VALUE, OTHER])
    with active_layout(layout):
        value_vec = embed(Token(VALUE, {"v": 0.5}), layout)
        assert extract_int_slot(value_vec, "col")._data.tolist() == [0.0]


def test_make_token_validates_against_canonical_type_not_caller():
    """If the caller passes a TokenType with the same name as a registered
    type but different slots, validation must use the layout's canonical
    type's slots, not the caller's. A caller-side typo on `.slots` should
    raise a clear ValueError naming the canonical declared slots."""
    layout = _layout()  # canonical RENDER has slots {col, chunk}
    rogue = TokenType("RENDER", slots={"col": IntSlot(0, 320), "different": IntSlot(0, 4)})
    with active_layout(layout):
        # "different" is on the rogue type but not the canonical one.
        with pytest.raises(ValueError, match="does not declare slot 'different'"):
            make_token(rogue, different=constant(1))


def test_make_token_round_trips_through_token_primitives():
    """End-to-end round-trip without deembed: build a token, then verify
    is_type and extract_*_slot recover its fields directly."""
    layout = _layout()
    with active_layout(layout):
        vec = make_token(RENDER, col=constant(123), chunk=constant(7))
        assert is_type(vec, RENDER)._data[0] == 1.0
        assert is_type(vec, VALUE)._data[0] == 0.0
        assert extract_int_slot(vec, "col")._data[0] == 123.0
        assert extract_int_slot(vec, "chunk")._data[0] == 7.0


def test_extract_int_slot_unknown_name_raises():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 1, "chunk": 0}), layout)
        with pytest.raises(ValueError, match="not declared on any type"):
            extract_int_slot(input_vec, "nope")


def test_extract_int_slot_wrong_kind_raises():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(VALUE, {"v": 1.0}), layout)
        with pytest.raises(ValueError, match="not declared as IntSlot"):
            extract_int_slot(input_vec, "v")


def test_extract_int_slot_depth_plus_1():
    layout = _layout()
    with active_layout(layout):
        input_vec = _make_vec(np.zeros(layout.width), depth=2)
        out = extract_int_slot(input_vec, "col")
    assert out.depth == 3


# --- extract_float_slot ---

def test_extract_float_slot_returns_value():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(VALUE, {"v": -3.5}), layout)
        v = extract_float_slot(input_vec, "v")
    assert v._data.tolist() == [-3.5]


def test_extract_float_slot_returns_zero_when_type_lacks_slot():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 1, "chunk": 0}), layout)
        v = extract_float_slot(input_vec, "v")
    assert v._data.tolist() == [0.0]


def test_extract_float_slot_wrong_kind_raises():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 1, "chunk": 0}), layout)
        with pytest.raises(ValueError, match="not declared as FloatSlot"):
            extract_float_slot(input_vec, "col")
