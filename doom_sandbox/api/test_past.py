"""Tests for Past — the cross-position attention primitives."""

from __future__ import annotations

import numpy as np
import pytest

from . import (
    FloatSlot,
    IntSlot,
    Past,
    Token,
    TokenType,
    constant,
)
from .past import MARGIN
from .vec import Vec, _make_vec
from ..runtime.embedding import Layout


# Vocab used across the suite.
RENDER = TokenType("RENDER", slots={
    "col": IntSlot(0, 320),
    "chunk": IntSlot(0, 16),
})
VALUE = TokenType("VALUE", slots={
    "v": FloatSlot(-40.0, 40.0, levels=65536),
})
NO_OP = TokenType("NO_OP", slots={})


def _layout():
    return Layout([RENDER, VALUE, NO_OP])


def _make_past(records: list[tuple[Token, dict[str, Vec]]]) -> Past:
    past = Past(_layout())
    for tok, exports in records:
        past._add(tok, exports)
    return past


def _vec(values, depth=0):
    return _make_vec(np.asarray(values, dtype=np.float64), depth=depth)


# --- Auto-export resolution ---

def test_resolve_input_type_returns_one_hot():
    past = _make_past([(Token(VALUE, {"v": 0.5}), {})])
    res = past._resolve(0, "input.type")
    assert res is not None
    assert res.shape == 3
    assert res.depth == 0
    assert res._data.tolist() == [0.0, 1.0, 0.0]


def test_resolve_input_slot_returns_value():
    past = _make_past([(Token(RENDER, {"col": 42, "chunk": 3}), {})])
    res = past._resolve(0, "input.col")
    assert res.depth == 0
    assert res._data.tolist() == [42.0]


def test_resolve_input_slot_missing_returns_none():
    past = _make_past([(Token(RENDER, {"col": 1, "chunk": 0}), {})])
    res = past._resolve(0, "input.v")  # RENDER doesn't have v
    assert res is None


def test_resolve_export_returns_vec():
    e = _vec([1.0, 2.0, 3.0], depth=4)
    past = _make_past([(Token(NO_OP, {}), {"foo": e})])
    res = past._resolve(0, "foo")
    assert res is e


def test_resolve_unknown_export_returns_none():
    past = _make_past([(Token(NO_OP, {}), {})])
    assert past._resolve(0, "unknown") is None


# --- pick_argmax basics ---

def test_pick_argmax_picks_highest_qk():
    keys = [_vec([1.0, 0.0]), _vec([0.0, 1.0]), _vec([0.5, 0.5])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]})
        for i in range(3)
    ])
    query = _vec([10.0, 0.0])  # scores: [10, 0, 5]; gap = 5 >= 1.0 -> clean pick of pos 0
    out = past.pick_argmax(query, "k", "v")
    assert out.shape == 1
    # Result is value at pos 0 (10.0) plus tiny noise.
    assert abs(out._data[0] - 10.0) < 1e-3


def test_pick_argmax_blends_on_close_tie():
    """gap < 1.0 → linear blend between top two values."""
    keys = [_vec([1.0]), _vec([1.0])]
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]})
        for i in range(2)
    ])
    # Make q·k tied: q=[1.0]
    query = _vec([1.0])  # both score = 1.0; gap = 0 → 50/50 blend
    out = past.pick_argmax(query, "k", "v")
    # Either pos 0 or pos 1 could be "top" for argsort; gap=0 so 50/50
    # blend -> 15.0 either way.
    assert abs(out._data[0] - 15.0) < 1e-3


def test_pick_argmax_blend_ratio_at_partial_gap():
    """gap = 0.5 → top_weight = 0.75, runner = 0.25."""
    keys = [_vec([2.0]), _vec([1.5])]
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]})
        for i in range(2)
    ])
    query = _vec([1.0])  # scores: [2.0, 1.5]; gap = 0.5
    out = past.pick_argmax(query, "k", "v")
    # top_w = 0.5 + 0.5/2 = 0.75 → 0.75*10 + 0.25*20 = 12.5
    assert abs(out._data[0] - 12.5) < 1e-3


def test_pick_argmax_empty_set_raises():
    past = _make_past([(Token(NO_OP, {}), {})])
    with pytest.raises(RuntimeError, match="no position has"):
        past.pick_argmax(_vec([1.0]), "k", "v")


def test_pick_argmax_key_shape_mismatch_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([1.0, 2.0]), "v": _vec([10.0])}),
    ])
    with pytest.raises(ValueError, match="does not match query shape"):
        past.pick_argmax(_vec([1.0]), "k", "v")


def test_pick_argmax_uses_input_auto_export():
    """`input.col` is queryable; query against it should pick the right position."""
    past = _make_past([
        (Token(RENDER, {"col": 5, "chunk": 0}), {"v": _vec([100.0])}),
        (Token(RENDER, {"col": 50, "chunk": 0}), {"v": _vec([200.0])}),
    ])
    # query=[1] · input.col=[50] = 50 ; vs query·col=[5] = 5 ; gap = 45 → clean pick
    out = past.pick_argmax(_vec([1.0]), "input.col", "v")
    assert abs(out._data[0] - 200.0) < 1e-3


# --- pick_argmin ---

def test_pick_argmin_picks_lowest_qk():
    keys = [_vec([5.0]), _vec([1.0]), _vec([3.0])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(3)
    ])
    out = past.pick_argmin(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 20.0) < 1e-3


def test_pick_argmin_blend_ratio_at_partial_gap():
    """For min-mode, gap is `runner - top` (positive); blend formula identical."""
    keys = [_vec([1.5]), _vec([2.0])]  # scores [1.5, 2.0]; min=1.5, runner=2.0, gap=0.5
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(2)
    ])
    out = past.pick_argmin(_vec([1.0]), "k", "v")
    # top_w = 0.75 on the min (10.0), runner_w = 0.25 on the next (20.0) → 12.5
    assert abs(out._data[0] - 12.5) < 1e-3


def test_pick_argmin_blends_on_close_tie():
    """Tied scores → 50/50 blend regardless of mode."""
    keys = [_vec([1.0]), _vec([1.0])]
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(2)
    ])
    out = past.pick_argmin(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 15.0) < 1e-3


# --- pick_above_argmin ---

def test_pick_above_argmin_filters_then_picks_lowest():
    keys = [_vec([0.0]), _vec([2.0]), _vec([5.0]), _vec([10.0])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0]), _vec([40.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(4)
    ])
    # query=[1], threshold=3.0 → scores [0, 2, 5, 10]; above 3 → [5, 10] → min = 5 at pos 2.
    out = past.pick_above_argmin(_vec([1.0]), "k", "v", constant(3.0))
    assert abs(out._data[0] - 30.0) < 1e-3


def test_pick_above_argmin_no_match_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([0.5]), "v": _vec([10.0])}),
    ])
    with pytest.raises(RuntimeError, match="no position has score"):
        past.pick_above_argmin(_vec([1.0]), "k", "v", constant(10.0))


def test_pick_above_argmin_threshold_must_be_1_shape():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([0.5]), "v": _vec([10.0])}),
    ])
    with pytest.raises(ValueError, match="threshold must be a 1-shape"):
        past.pick_above_argmin(_vec([1.0]), "k", "v", _vec([1.0, 2.0]))


def test_pick_above_argmin_blends_on_close_tie_among_survivors():
    """After threshold filter, the blend rule still applies to surviving candidates."""
    keys = [_vec([0.0]), _vec([4.0]), _vec([4.5]), _vec([10.0])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0]), _vec([40.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(4)
    ])
    # threshold=3.0 filters to [4, 4.5, 10]. argmin in min mode: top=4 (val 20), runner=4.5 (val 30).
    # gap = 0.5 → top_w = 0.75 → 0.75*20 + 0.25*30 = 22.5
    out = past.pick_above_argmin(_vec([1.0]), "k", "v", constant(3.0))
    assert abs(out._data[0] - 22.5) < 1e-3


def test_pick_above_argmin_empty_export_set_raises_with_export_message():
    """When no position has the required names, the error must say so —
    NOT 'no position has score > X' (which would mislead the caller into
    thinking the threshold is the issue)."""
    past = _make_past([(Token(NO_OP, {}), {})])  # no exports at all
    with pytest.raises(RuntimeError, match="has both"):
        past.pick_above_argmin(_vec([1.0]), "k", "v", constant(0.0))


def test_pick_above_argmin_by_empty_export_set_raises_with_export_message():
    past = _make_past([(Token(NO_OP, {}), {})])
    with pytest.raises(RuntimeError, match="has both"):
        past.pick_above_argmin_by("score", "v", constant(0.0))


def test_blend_zone_value_shape_mismatch_raises():
    """If the top-two candidates have different value shapes, blending is
    ill-defined — must raise rather than silently broadcast."""
    keys = [_vec([1.0]), _vec([1.0])]  # tied scores → blend zone
    values = [_vec([10.0, 20.0]), _vec([30.0])]  # shapes 2 and 1
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(2)
    ])
    with pytest.raises(ValueError, match="inconsistent value shapes"):
        past.pick_argmax(_vec([1.0]), "k", "v")


def test_pick_above_argmin_threshold_depth_contributes():
    """`threshold.depth` is a real input; it must contribute to the result depth."""
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([5.0], depth=1), "v": _vec([10.0], depth=1)}),
    ])
    threshold = _vec([1.0], depth=20)
    out = past.pick_above_argmin(_vec([1.0], depth=1), "k", "v", threshold)
    assert out.depth == 21


# --- lookup ---

def test_lookup_clean_match_returns_value():
    """Equality-style: pos 1 has key matching query, others don't."""
    keys = [_vec([0.0, 0.0]), _vec([1.0, 0.0]), _vec([0.0, 1.0])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(3)
    ])
    # query=[10, 0]; scores: [0, 10, 0]; gap=10 → unambiguous → pos 1
    out = past.lookup(_vec([10.0, 0.0]), "k", "v")
    assert abs(out._data[0] - 20.0) < 1e-3


def test_lookup_ambiguous_raises_with_diagnostic():
    """The error must name both close-scoring positions and their scores."""
    keys = [_vec([1.0]), _vec([0.7])]  # gap = 0.3 < 1.0
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(2)
    ])
    with pytest.raises(RuntimeError) as exc:
        past.lookup(_vec([1.0]), "k", "v")
    msg = str(exc.value)
    assert "ambiguous" in msg
    # Both position indices must appear.
    assert "position 0" in msg
    assert "position 1" in msg
    # Both scores (1.0000 and 0.7000) must appear; the format string uses .4f.
    assert "1.0000" in msg
    assert "0.7000" in msg


def test_lookup_empty_set_raises():
    past = _make_past([(Token(NO_OP, {}), {})])
    with pytest.raises(RuntimeError, match="no position has"):
        past.lookup(_vec([1.0]), "k", "v")


def test_lookup_single_candidate_succeeds():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([1.0]), "v": _vec([99.0])}),
    ])
    out = past.lookup(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 99.0) < 1e-3


def test_lookup_succeeds_at_exact_margin_boundary():
    """gap == 1.0 must succeed (the rule is `gap < 1.0 → ambiguous`, strict <).
    Catches a regression that uses `<=` instead of `<` on the margin check."""
    keys = [_vec([2.0]), _vec([1.0])]  # scores [2.0, 1.0], gap exactly 1.0
    values = [_vec([10.0]), _vec([20.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(2)
    ])
    out = past.lookup(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 10.0) < 1e-3


# --- pick_most_recent ---

def test_pick_most_recent_picks_largest_position_among_matches():
    """Three matching positions; return the value at the latest."""
    keys = [_vec([1.0]), _vec([1.0]), _vec([1.0])]
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(3)
    ])
    out = past.pick_most_recent(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 30.0) < 1e-3


def test_pick_most_recent_excludes_below_margin():
    """A position with a much lower score is excluded from the recency vote."""
    keys = [_vec([1.0]), _vec([1.0]), _vec([0.0])]  # pos 2 mismatches
    values = [_vec([10.0]), _vec([20.0]), _vec([30.0])]
    past = _make_past([
        (Token(NO_OP, {}), {"k": keys[i], "v": values[i]}) for i in range(3)
    ])
    # scores [1, 1, 0]; best=1; matching = pos 0 and 1; most recent = pos 1
    out = past.pick_most_recent(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 20.0) < 1e-3


def test_pick_most_recent_empty_set_raises():
    past = _make_past([(Token(NO_OP, {}), {})])
    with pytest.raises(RuntimeError, match="no position has"):
        past.pick_most_recent(_vec([1.0]), "k", "v")


def test_pick_most_recent_strict_margin_boundary():
    """`best - score < MARGIN` is strict — gap = 1.0 excludes, gap = 0.999 includes."""
    # Position 0 is the recent low-score; position 1 is the high-score.
    # If we make pos 0's score = best - 1.0 exactly, it's excluded → return pos 1.
    keys_excluded = [_vec([0.0]), _vec([1.0])]
    values = [_vec([10.0]), _vec([20.0])]
    past_excl = _make_past([
        (Token(NO_OP, {}), {"k": keys_excluded[i], "v": values[i]}) for i in range(2)
    ])
    out = past_excl.pick_most_recent(_vec([1.0]), "k", "v")
    assert abs(out._data[0] - 20.0) < 1e-3  # only pos 1 survives the strict margin

    # scores [1.0, 0.001]; best=1.0; pos 1's gap = 0.999 — included.
    # Both positions match, so the most recent (largest index, pos 1) wins.
    keys_included = [_vec([1.0]), _vec([0.001])]
    past_inc = _make_past([
        (Token(NO_OP, {}), {"k": keys_included[i], "v": values[i]}) for i in range(2)
    ])
    out_inc = past_inc.pick_most_recent(_vec([1.0]), "k", "v")
    assert abs(out_inc._data[0] - 20.0) < 1e-3


def test_pick_argmax_uses_input_type_auto_export():
    """`input.type` is queryable as a key (one-hot Vec of width N_TYPES)."""
    layout = _layout()
    past = _make_past([
        (Token(RENDER, {"col": 1, "chunk": 0}), {"v": _vec([100.0])}),
        (Token(VALUE, {"v": 0.0}), {"v": _vec([200.0])}),
    ])
    # Query is a one-hot for VALUE (column 1 in the layout's type one-hot).
    # input.type at pos 0 = one-hot RENDER (col 0); at pos 1 = one-hot VALUE (col 1).
    # query·input.type at pos 0 = 0; at pos 1 = 1; gap = 1.0 → clean pick of pos 1.
    query = _vec([0.0, 1.0, 0.0])  # width matches len(layout.types)
    out = past.pick_argmax(query, "input.type", "v")
    assert abs(out._data[0] - 200.0) < 1e-3


# --- pick_argmax_by / pick_argmin_by ---

def test_pick_argmax_by_uses_precomputed_score():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([1.0]), "v": _vec([10.0])}),
        (Token(NO_OP, {}), {"score": _vec([5.0]), "v": _vec([20.0])}),
        (Token(NO_OP, {}), {"score": _vec([3.0]), "v": _vec([30.0])}),
    ])
    out = past.pick_argmax_by("score", "v")
    assert abs(out._data[0] - 20.0) < 1e-3


def test_pick_argmin_by_uses_precomputed_score():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([5.0]), "v": _vec([10.0])}),
        (Token(NO_OP, {}), {"score": _vec([1.0]), "v": _vec([20.0])}),
        (Token(NO_OP, {}), {"score": _vec([3.0]), "v": _vec([30.0])}),
    ])
    out = past.pick_argmin_by("score", "v")
    assert abs(out._data[0] - 20.0) < 1e-3


def test_pick_above_argmin_by_filters_then_picks_lowest():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([0.0]), "v": _vec([10.0])}),
        (Token(NO_OP, {}), {"score": _vec([2.0]), "v": _vec([20.0])}),
        (Token(NO_OP, {}), {"score": _vec([5.0]), "v": _vec([30.0])}),
        (Token(NO_OP, {}), {"score": _vec([10.0]), "v": _vec([40.0])}),
    ])
    out = past.pick_above_argmin_by("score", "v", constant(3.0))
    assert abs(out._data[0] - 30.0) < 1e-3


def test_pick_argmax_by_non_unit_score_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([1.0, 2.0]), "v": _vec([10.0])}),
    ])
    with pytest.raises(ValueError, match="must be a 1-shape Vec for"):
        past.pick_argmax_by("score", "v")


def test_pick_argmin_by_non_unit_score_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([1.0, 2.0]), "v": _vec([10.0])}),
    ])
    with pytest.raises(ValueError, match="must be a 1-shape Vec for"):
        past.pick_argmin_by("score", "v")


def test_pick_above_argmin_by_non_unit_score_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([1.0, 2.0]), "v": _vec([10.0])}),
    ])
    with pytest.raises(ValueError, match="must be a 1-shape Vec for"):
        past.pick_above_argmin_by("score", "v", constant(0.0))


# --- mean ---

def test_mean_single_contributor_is_identity():
    past = _make_past([
        (Token(NO_OP, {}), {"x": _vec([5.0, 7.0])}),
    ])
    out = past.mean("x")
    assert np.allclose(out._data, [5.0, 7.0], atol=1e-3)


def test_mean_multi_contributor_averages():
    past = _make_past([
        (Token(NO_OP, {}), {"x": _vec([10.0, 20.0])}),
        (Token(NO_OP, {}), {"x": _vec([20.0, 40.0])}),
        (Token(NO_OP, {}), {"x": _vec([30.0, 60.0])}),
    ])
    out = past.mean("x")
    assert np.allclose(out._data, [20.0, 40.0], atol=1e-3)


def test_mean_skips_positions_not_exporting():
    past = _make_past([
        (Token(NO_OP, {}), {"x": _vec([2.0])}),
        (Token(NO_OP, {}), {}),  # no x export
        (Token(NO_OP, {}), {"x": _vec([4.0])}),
    ])
    out = past.mean("x")
    assert abs(out._data[0] - 3.0) < 1e-3


def test_mean_empty_set_raises():
    past = _make_past([(Token(NO_OP, {}), {})])
    with pytest.raises(RuntimeError, match="no position has"):
        past.mean("x")


def test_mean_inconsistent_shapes_raises():
    past = _make_past([
        (Token(NO_OP, {}), {"x": _vec([1.0, 2.0])}),
        (Token(NO_OP, {}), {"x": _vec([3.0])}),
    ])
    with pytest.raises(ValueError, match="inconsistent shapes"):
        past.mean("x")


# --- Depth propagation ---

def test_pick_argmax_depth_is_max_inputs_plus_1():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([1.0], depth=3), "v": _vec([10.0], depth=4)}),
        (Token(NO_OP, {}), {"k": _vec([2.0], depth=7), "v": _vec([20.0], depth=2)}),
    ])
    # query depth 5; deepest k = 7; deepest v = 4 → result depth = 8
    out = past.pick_argmax(_vec([1.0], depth=5), "k", "v")
    assert out.depth == 8


def test_mean_depth_max_contributor_plus_1():
    past = _make_past([
        (Token(NO_OP, {}), {"x": _vec([1.0], depth=2)}),
        (Token(NO_OP, {}), {"x": _vec([1.0], depth=9)}),
    ])
    out = past.mean("x")
    assert out.depth == 10


def test_pick_argmax_by_depth_excludes_query():
    past = _make_past([
        (Token(NO_OP, {}), {"score": _vec([1.0], depth=3), "v": _vec([10.0], depth=2)}),
    ])
    out = past.pick_argmax_by("score", "v")
    # No query, so depth = max(3, 2) + 1 = 4
    assert out.depth == 4


# --- Determinism ---

def test_pick_argmax_deterministic():
    past = _make_past([
        (Token(NO_OP, {}), {"k": _vec([1.0]), "v": _vec([100.0])}),
        (Token(NO_OP, {}), {"k": _vec([2.0]), "v": _vec([200.0])}),
    ])
    a = past.pick_argmax(_vec([1.0]), "k", "v")
    b = past.pick_argmax(_vec([1.0]), "k", "v")
    assert np.array_equal(a._data, b._data)


# --- MARGIN constant exposed ---

def test_margin_is_one():
    assert MARGIN == 1.0
