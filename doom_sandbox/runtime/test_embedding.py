"""Tests for the Layout class and embed/deembed."""

from __future__ import annotations

import pytest

from doom_sandbox.api import (
    FloatSlot,
    IntSlot,
    Token,
    TokenType,
)
from doom_sandbox.runtime.embedding import (
    Layout,
    _quantize_float,
    deembed,
    embed,
)


# --- Layout construction ---

def _vocab():
    render = TokenType("RENDER", slots={
        "col": IntSlot(0, 320),
        "chunk": IntSlot(0, 16),
    })
    value = TokenType("VALUE", slots={
        "v": FloatSlot(-40.0, 40.0, levels=65536),
    })
    no_op = TokenType("NO_OP", slots={})
    return [render, value, no_op]


def test_layout_width_is_n_types_plus_total_slots():
    types = _vocab()
    layout = Layout(types)
    # 3 types + 2 RENDER slots + 1 VALUE slot + 0 NO_OP slots = 6 columns
    assert layout.width == 6


def test_layout_type_columns_are_zero_through_n_minus_1():
    types = _vocab()
    layout = Layout(types)
    assert layout.type_columns == {"RENDER": 0, "VALUE": 1, "NO_OP": 2}


def test_layout_slot_columns_after_type_block():
    types = _vocab()
    layout = Layout(types)
    # Order matches declaration: RENDER.col, RENDER.chunk, VALUE.v
    assert layout.slot_columns[("RENDER", "col")] == 3
    assert layout.slot_columns[("RENDER", "chunk")] == 4
    assert layout.slot_columns[("VALUE", "v")] == 5


def test_layout_columns_by_slot_name_indexes_all_types_with_slot():
    other = TokenType("OTHER", slots={"col": IntSlot(0, 10)})
    types = _vocab() + [other]
    layout = Layout(types)
    cols = sorted(c for _, c in layout.columns_by_slot_name["col"])
    # 4 type one-hots (0..3), then RENDER.col=4, RENDER.chunk=5,
    # VALUE.v=6, OTHER.col=7. So "col" lives at columns 4 and 7.
    assert cols == [4, 7]


def test_layout_duplicate_type_name_raises():
    a = TokenType("X", slots={"k": IntSlot(0, 4)})
    b = TokenType("X", slots={"k": IntSlot(0, 4)})
    with pytest.raises(ValueError, match="duplicate"):
        Layout([a, b])


# --- embed / deembed round-trip ---

def test_embed_writes_type_one_hot_and_slot_value():
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[0], values={"col": 37, "chunk": 2})
    vec = embed(tok, layout)
    assert vec.shape == 6
    assert vec.depth == 0
    assert vec._data.tolist() == [1.0, 0.0, 0.0, 37.0, 2.0, 0.0]


def test_embed_unknown_slot_raises():
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[0], values={"bogus": 1})
    with pytest.raises(ValueError, match="does not declare slot"):
        embed(tok, layout)


def test_embed_unknown_type_raises():
    types = _vocab()
    layout = Layout(types[:2])  # exclude NO_OP
    tok = Token(type=types[2], values={})
    with pytest.raises(ValueError, match="not in the active vocab"):
        embed(tok, layout)


def test_int_slot_round_trips_exactly():
    types = _vocab()
    layout = Layout(types)
    for col in [0, 1, 100, 319]:
        tok = Token(type=types[0], values={"col": col, "chunk": 5})
        decoded = deembed(embed(tok, layout), layout)
        assert decoded.values["col"] == col
        assert decoded.values["chunk"] == 5
        assert decoded.type == types[0]


def test_float_slot_round_trips_within_one_lsb():
    types = _vocab()
    layout = Layout(types)
    slot: FloatSlot = types[1].slots["v"]
    span = slot.hi - slot.lo
    one_lsb = span / (slot.levels - 1)
    for v in [-40.0, -1.234, 0.0, 7.5, 39.999]:
        tok = Token(type=types[1], values={"v": v})
        decoded = deembed(embed(tok, layout), layout)
        assert abs(decoded.values["v"] - v) <= one_lsb / 2 + 1e-9


def test_int_slot_clamps_to_hi_minus_1_on_overshoot():
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[0], values={"col": 999, "chunk": 0})
    decoded = deembed(embed(tok, layout), layout)
    assert decoded.values["col"] == 319  # IntSlot(0, 320) -> [0, 320), max 319


def test_int_slot_clamps_to_lo_on_undershoot():
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[0], values={"col": -50, "chunk": 0})
    decoded = deembed(embed(tok, layout), layout)
    assert decoded.values["col"] == 0


def test_deembed_picks_max_type_one_hot():
    """Argmax over the type one-hot recovers the type."""
    types = _vocab()
    layout = Layout(types)
    # Hand-build a noisy type-vector — `embed` always writes a clean
    # one-hot, so to exercise argmax over a non-one-hot we have to
    # bypass the public API.
    import numpy as np
    from doom_sandbox.api.vec import _make_vec
    data = np.array([0.3, 1.0, 0.4, 0.0, 0.0, 5.0])
    vec = _make_vec(data, depth=0)
    decoded = deembed(vec, layout)
    assert decoded.type == types[1]
    # And the v slot reads its column (5.0, snapped).
    assert abs(decoded.values["v"] - 5.0) <= (80.0 / (65536 - 1))


def test_deembed_no_op_has_no_slot_values():
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[2], values={})
    decoded = deembed(embed(tok, layout), layout)
    assert decoded.type == types[2]
    assert decoded.values == {}


def test_quantize_float_exact_at_endpoints():
    slot = FloatSlot(-1.0, 1.0, levels=5)
    assert _quantize_float(-1.0, slot) == -1.0
    assert _quantize_float(1.0, slot) == 1.0


def test_quantize_float_clamps_outside_range():
    slot = FloatSlot(-1.0, 1.0, levels=5)
    assert _quantize_float(-100.0, slot) == -1.0
    assert _quantize_float(100.0, slot) == 1.0


def test_quantize_float_to_nearest_step():
    slot = FloatSlot(0.0, 1.0, levels=5)  # steps at 0, 0.25, 0.5, 0.75, 1.0
    assert _quantize_float(0.6, slot) == 0.5
    assert _quantize_float(0.7, slot) == 0.75


def test_quantize_float_asymmetric_around_zero():
    """Range [-3, 5] with 9 levels → step 1.0; raw 0 lands on a step."""
    slot = FloatSlot(-3.0, 5.0, levels=9)
    assert _quantize_float(0.0, slot) == 0.0
    assert _quantize_float(0.4, slot) == 0.0
    assert _quantize_float(0.6, slot) == 1.0
    assert _quantize_float(-2.4, slot) == -2.0


def test_embed_omitted_slot_round_trips_to_zero():
    """A Token with declared slots not in `values` writes 0 into those columns;
    deembed recovers IntSlot.lo / FloatSlot quantized lo."""
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[0], values={})  # neither col nor chunk supplied
    decoded = deembed(embed(tok, layout), layout)
    assert decoded.type == types[0]
    assert decoded.values["col"] == 0  # IntSlot(0, 320) clamped/rounded → 0
    assert decoded.values["chunk"] == 0


def test_omitted_float_slot_defaults_to_zero_quantized():
    """Omitted FloatSlot reads as 0.0 (which is in-range for VALUE) and
    quantizes to its nearest level — for a [-40, 40] range that's 0.0."""
    import numpy as np
    types = _vocab()
    layout = Layout(types)
    tok = Token(type=types[1], values={})  # VALUE has slot v: FloatSlot(-40, 40)
    decoded = deembed(embed(tok, layout), layout)
    assert decoded.type == types[1]
    assert abs(decoded.values["v"]) <= (80.0 / (65536 - 1)) / 2 + 1e-9


def test_int_slot_fractional_raw_value_rounds():
    """Hand-build a Vec with a fractional value in an IntSlot column to
    exercise the round() step of deembed (not reachable via embed,
    which writes integer values directly)."""
    import numpy as np
    from doom_sandbox.api.vec import _make_vec
    types = _vocab()
    layout = Layout(types)
    col_col = layout.slot_columns[("RENDER", "col")]
    chunk_col = layout.slot_columns[("RENDER", "chunk")]
    data = np.zeros(layout.width)
    data[layout.type_columns["RENDER"]] = 1.0
    data[col_col] = 4.7    # rounds to 5
    data[chunk_col] = 2.3  # rounds to 2
    decoded = deembed(_make_vec(data, depth=0), layout)
    assert decoded.values["col"] == 5
    assert decoded.values["chunk"] == 2


def test_shared_slot_name_round_trips_independently():
    """Two types declaring the same slot name must round-trip without
    cross-contamination — each owns its own column."""
    other = TokenType("OTHER", slots={"col": IntSlot(0, 8)})
    layout = Layout(_vocab() + [other])
    render_tok = Token(type=_vocab()[0], values={"col": 200, "chunk": 5})
    other_tok = Token(type=other, values={"col": 7})
    assert deembed(embed(render_tok, layout), layout).values["col"] == 200
    assert deembed(embed(other_tok, layout), layout).values["col"] == 7
