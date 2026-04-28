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
    extract_type_slot,
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


# --- TokenType.one_hot ---

def test_one_hot_sets_type_column():
    layout = _layout()
    with active_layout(layout):
        vec = RENDER.one_hot()
    n_types = len(layout.types)
    assert vec.shape == n_types
    assert vec.depth == 0
    assert vec._data[layout.type_columns["RENDER"]] == 1.0
    for name, col in layout.type_columns.items():
        if name != "RENDER":
            assert vec._data[col] == 0.0


def test_one_hot_for_each_type_is_unique_one_hot():
    layout = _layout()
    with active_layout(layout):
        seen = []
        for T in (RENDER, VALUE, NO_OP):
            v = T.one_hot()
            assert v._data.sum() == 1.0
            seen.append(tuple(v._data.tolist()))
    assert len(set(seen)) == len(seen)  # each type's one-hot is distinct


def test_one_hot_raises_for_type_outside_active_vocab():
    layout = _layout([RENDER, VALUE])  # NO_OP excluded
    with active_layout(layout):
        with pytest.raises(ValueError, match="not in the active vocab"):
            NO_OP.one_hot()


def test_one_hot_raises_without_active_layout():
    with pytest.raises(RuntimeError, match="No active token layout"):
        RENDER.one_hot()


def test_one_hot_dot_input_type_recovers_type_match_bit():
    """The point of one_hot(): use it as a query against `input.type`.
    Their dot product is 1.0 when input matches T, 0.0 otherwise."""
    layout = _layout()
    with active_layout(layout):
        for tok_type in (RENDER, VALUE, NO_OP):
            input_vec = embed(Token(tok_type, {}), layout)
            # input.type slice = first N_TYPES columns of input_vec
            n_types = len(layout.types)
            input_type = input_vec._data[:n_types]
            for query_type in (RENDER, VALUE, NO_OP):
                q = query_type.one_hot()
                score = float(np.dot(q._data, input_type))
                expected = 1.0 if query_type == tok_type else 0.0
                assert score == expected


# --- TokenType.check ---

def test_check_returns_1_for_matching_type():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 7, "chunk": 1}), layout)
        out = RENDER.check(input_vec)
        assert out.shape == 1
        assert out._data[0] == 1.0


def test_check_returns_0_for_non_matching_type():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(VALUE, {"v": 0.5}), layout)
        out = RENDER.check(input_vec)
        assert out._data[0] == 0.0


def test_check_depth_plus_1():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {}), layout)
        out = RENDER.check(input_vec)
        assert out.depth == input_vec.depth + 1


def test_check_unknown_type_raises():
    layout = _layout([RENDER, VALUE])  # NO_OP excluded
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {}), layout)
        with pytest.raises(ValueError, match="not in the active vocab"):
            NO_OP.check(input_vec)


def test_check_full_token_only_rejects_input_type_slice():
    """T.check() requires the full token Vec. The narrower input.type
    slice (width-N_TYPES) is rejected — passing the wrong shape would
    be silent corruption otherwise."""
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {}), layout)
        n_types = len(layout.types)
        slice_vec = _make_vec(input_vec._data[:n_types], depth=0)
        with pytest.raises(ValueError, match="expected a Vec of width"):
            RENDER.check(slice_vec)


# --- extract_type_slot ---

def test_extract_type_slot_returns_value_at_specific_column():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 42, "chunk": 3}), layout)
        col = extract_type_slot(input_vec, RENDER, "col")
        chunk = extract_type_slot(input_vec, RENDER, "chunk")
        assert col._data[0] == 42
        assert chunk._data[0] == 3


def test_extract_type_slot_auto_masks_when_active_type_differs():
    """When the active input is a different type, the (T, slot) column
    is 0.0, so extract returns 0 — the auto-masking property."""
    layout = Layout([RENDER, OTHER])
    with active_layout(layout):
        # OTHER is active; RENDER's col column is 0 (not written).
        input_vec = embed(Token(OTHER, {"col": 7}), layout)
        out = extract_type_slot(input_vec, RENDER, "col")
        assert out._data[0] == 0.0


def test_extract_type_slot_disambiguates_shared_slot_names():
    """RENDER and OTHER both have a 'col' slot at different columns.
    extract_type_slot reads exactly one of them; flat
    extract_int_slot would sum them."""
    layout = Layout([RENDER, OTHER])
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 5}), layout)
        # extract_type_slot reads RENDER's col exactly.
        from_render = extract_type_slot(input_vec, RENDER, "col")
        from_other = extract_type_slot(input_vec, OTHER, "col")
        assert from_render._data[0] == 5
        assert from_other._data[0] == 0


def test_extract_type_slot_unknown_type_raises():
    layout = _layout([RENDER, VALUE])  # NO_OP excluded
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {}), layout)
        with pytest.raises(ValueError, match="not in the active vocab"):
            extract_type_slot(input_vec, NO_OP, "col")


def test_extract_type_slot_undeclared_slot_raises():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {}), layout)
        with pytest.raises(ValueError, match="does not declare slot"):
            extract_type_slot(input_vec, RENDER, "v")  # v is on VALUE, not RENDER


def test_extract_type_slot_wrong_input_width_raises():
    layout = _layout()
    with active_layout(layout):
        narrow = _make_vec(np.zeros(1), depth=0)
        with pytest.raises(ValueError, match="expected a Vec of width"):
            extract_type_slot(narrow, RENDER, "col")


def test_extract_type_slot_depth_plus_1():
    layout = _layout()
    with active_layout(layout):
        input_vec = embed(Token(RENDER, {"col": 1}), layout)
        out = extract_type_slot(input_vec, RENDER, "col")
        assert out.depth == input_vec.depth + 1


def test_extract_type_slot_raises_without_active_layout():
    with pytest.raises(RuntimeError, match="No active token layout"):
        extract_type_slot(_make_vec(np.zeros(5), depth=0), RENDER, "col")


# --- Cardinality ---

def test_int_slot_cardinality_is_hi_minus_lo():
    assert IntSlot(0, 320).cardinality == 320
    assert IntSlot(-8, 9).cardinality == 17
    assert IntSlot(0, 1).cardinality == 1


def test_float_slot_cardinality_is_levels():
    assert FloatSlot(-1.0, 1.0, levels=65536).cardinality == 65536
    assert FloatSlot(-10.0, 10.0, levels=4).cardinality == 4


def test_token_type_cardinality_no_slots_is_1():
    assert NO_OP.cardinality == 1


def test_token_type_cardinality_single_slot_matches_slot():
    assert OTHER.cardinality == 8  # single IntSlot(0, 8)


def test_token_type_cardinality_multi_slot_is_product():
    # RENDER has IntSlot(0, 320) and IntSlot(0, 16)
    assert RENDER.cardinality == 320 * 16


def test_token_type_cardinality_packed_slots_explode():
    """The point of cardinality tracking: many packed slots multiply
    into an unrenderably large vocabulary."""
    PACKED = TokenType("PACKED", slots={
        f"c{i}": IntSlot(-8, 9) for i in range(16)
    })
    assert PACKED.cardinality == 17 ** 16  # ~4.5e19
