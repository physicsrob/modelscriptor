"""Unit tests for the SORTED stage (torchwright.doom.stages.sorted).

Post-Phase-E: SORTED uses ``attend_argmin_above_integer``.  The
attention selects the smallest ``bsp_rank`` strictly greater than
``prev_bsp_rank`` among renderable walls.  Non-renderable walls have
all-zero ``indicators_above`` and can never win the "above" logit,
so they drop out of the selection without a separate validity gate.
"""

from typing import List

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.graph.asserts import assert_integer
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.sorted import SortedInputs, build_sorted


_MAX_WALLS = 4


def _indicators(bsp_rank: int, is_renderable: bool) -> List[float]:
    """Build the width-max_walls thermometer for a wall.

    Slot c == 1.0 iff bsp_rank >= c AND is_renderable.
    """
    if not is_renderable:
        return [0.0] * _MAX_WALLS
    return [1.0 if c <= bsp_rank else 0.0 for c in range(_MAX_WALLS)]


@pytest.fixture(scope="module")
def sorted_module():
    """Compile build_sorted's outputs for unit testing.

    ``sort_value`` (packed payload), ``indicators_above``, and
    ``prev_bsp_rank`` are fed as multi-wide ``create_input`` nodes so the
    test controls every position independently.
    """
    from torchwright.doom.wall_payload import payload_width

    pos = create_pos_encoding()
    payload_w = payload_width(_MAX_WALLS)

    sort_score = assert_integer(
        create_input("sort_score", 1, value_range=(0.0, 100.0))
    )
    sort_value = create_input("sort_value", payload_w, value_range=(-_MAX_COORD, _MAX_COORD))
    indicators_above = create_input(
        "indicators_above", _MAX_WALLS, value_range=(0.0, 1.0),
    )
    prev_bsp_rank = create_input("prev_bsp_rank", 1, value_range=(-1.0, 100.0))
    is_sorted = create_input("is_sorted", 1, value_range=(-1.0, 1.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))

    out = build_sorted(
        SortedInputs(
            sort_score=sort_score,
            sort_value=sort_value,
            indicators_above=indicators_above,
            prev_bsp_rank=prev_bsp_rank,
            is_sorted=is_sorted,
            is_wall=is_wall,
            pos_encoding=pos,
        ),
        max_walls=_MAX_WALLS,
    )
    output = Concatenate([
        out.sel_wall_data,       # 5-wide
        out.sel_onehot,          # max_walls-wide
        out.sel_bsp_rank,        # 1-wide
        out.sort_done,           # 1-wide
    ])
    return compile_headless(
        output, pos, d=2048, d_head=32, max_layers=60, verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start:start + width] = torch.tensor(
                row[name], dtype=torch.float32,
            ).reshape(width)
    return t


def _wall_payload(ax, ay, bx, by, tex_id, sort_den, C, D, E, H_inv,
                  bsp_rank, vis_lo=0.0, vis_hi=0.0,
                  onehot: List[float] = ()) -> List[float]:
    """Canonical wall payload layout — must match wall_payload.pack_wall_payload."""
    return ([ax, ay, bx, by, tex_id, sort_den, C, D, E, H_inv, bsp_rank,
             vis_lo, vis_hi] + list(onehot))


def _onehot(i: int, n: int) -> List[float]:
    """0.5-biased one-hot (matches wall_payload.add_scaled_nodes output)."""
    return [1.0 if k == i else 0.5 for k in range(n)]


# ---------------------------------------------------------------------------
# Argmin-above correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prev_bsp_rank,expected_rank,label", [
    # EOS seed: threshold = 0, so we pick the smallest bsp_rank >= 0,
    # which is 0 (wall at rank 0).
    (-1.0, 0, "eos_seed_picks_rank_0"),
    # prev=0: threshold slot 1, picks rank 1.
    (0.0, 1, "pick_rank_1_after_0"),
    # prev=1: threshold slot 2, picks rank 2.
    (1.0, 2, "pick_rank_2_after_1"),
    # prev=2: threshold slot 3, picks rank 3.
    (2.0, 3, "pick_rank_3_after_2"),
])
def test_argmin_above_picks_strict_next_rank(
    sorted_module, prev_bsp_rank, expected_rank, label,
):
    """attend_argmin_above_integer picks the smallest bsp_rank strictly
    greater than prev_bsp_rank among renderable walls."""
    wall_rows = []
    for i in range(_MAX_WALLS):
        # Wall i has bsp_rank=i, is_renderable=True.  ax=wall index so
        # we can verify the pick.
        wall_rows.append({
            "sort_score": float(i),
            "sort_value": _wall_payload(
                ax=float(i), ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                bsp_rank=float(i),
                onehot=_onehot(i, _MAX_WALLS),
            ),
            "indicators_above": _indicators(i, is_renderable=True),
            "prev_bsp_rank": prev_bsp_rank,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })
    sorted_row = {
        "sort_score": 99.0,
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "indicators_above": [0.0] * _MAX_WALLS,
        "prev_bsp_rank": prev_bsp_rank,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }
    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)
    # sel_wall_data[0] = ax = expected wall index.
    sel_ax = out[sorted_pos, 0].item()
    assert abs(sel_ax - expected_rank) < 0.3, (
        f"{label}: expected wall rank {expected_rank}, got sel_ax={sel_ax:+.2f}"
    )


def test_non_renderable_walls_are_skipped(sorted_module):
    """Walls with is_renderable=False have all-zero indicators_above and
    can't win the above-threshold logit — the attention must skip them
    and pick the smallest-rank renderable wall.

    Scenario: wall 0 has rank 0 but is non-renderable.  Wall 1 has rank 1
    and is renderable.  EOS seed (prev=-1, threshold=0) must pick wall 1.
    """
    walls = [
        (0, False, 0.5),   # wall 0: non-renderable, rank 0, ax=0.5
        (1, True, 1.5),    # wall 1: renderable, rank 1, ax=1.5
        (2, True, 2.5),    # wall 2: renderable, rank 2, ax=2.5
        (3, True, 3.5),    # wall 3: renderable, rank 3, ax=3.5
    ]
    wall_rows = []
    for i, (rank, renderable, ax) in enumerate(walls):
        # Non-renderable walls get a sentinel score that keeps
        # assert_distinct_across happy while not luring the attention.
        # Score ties would trip the pairwise-distinctness assert.
        wall_rows.append({
            "sort_score": float(rank) if renderable else float(rank) + 90.0,
            "sort_value": _wall_payload(
                ax=ax, ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                bsp_rank=float(rank),
                onehot=_onehot(i, _MAX_WALLS),
            ),
            "indicators_above": _indicators(rank, renderable),
            "prev_bsp_rank": -1.0,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })
    sorted_row = {
        "sort_score": 99.0,
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "indicators_above": [0.0] * _MAX_WALLS,
        "prev_bsp_rank": -1.0,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }
    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)
    # Expected: wall 1 picked (the smallest renderable).  ax=1.5.
    sel_ax = out[sorted_pos, 0].item()
    assert abs(sel_ax - 1.5) < 0.1, (
        f"expected renderable wall 1 (ax=1.5), got sel_ax={sel_ax:+.3f}"
    )


# ---------------------------------------------------------------------------
# sort_done semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prev_bsp_rank,expected_sel,expected_sort_done,label", [
    # EOS seed: prev=-1, sel=0.  diff=-1 → sort_done=-1 (active).
    (-1.0, 0, -1.0, "eos_seed_active"),
    # prev=2, sel=3 (progress).  diff=-1 → sort_done=-1 (active).
    (2.0, 3, -1.0, "progress_active"),
    # Last progress step: prev=2, sel=3.  Covers the diff=-1 boundary.
    (2.0, 3, -1.0, "adjacent_below_threshold"),
])
def test_sort_done_progress_signal(
    sorted_module, prev_bsp_rank, expected_sel, expected_sort_done, label,
):
    """sort_done = sign(prev_bsp_rank - sel_bsp_rank + 0.5).

    During progress the attention picks sel > prev by at least one, so
    diff = prev - sel <= -1 < -0.5 → sort_done = -1 (active).
    """
    wall_rows = []
    for i in range(_MAX_WALLS):
        wall_rows.append({
            "sort_score": float(i),
            "sort_value": _wall_payload(
                ax=0.0, ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                bsp_rank=float(i),
                onehot=_onehot(i, _MAX_WALLS),
            ),
            "indicators_above": _indicators(i, is_renderable=True),
            "prev_bsp_rank": prev_bsp_rank,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })
    sorted_row = {
        "sort_score": 99.0,
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "indicators_above": [0.0] * _MAX_WALLS,
        "prev_bsp_rank": prev_bsp_rank,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }
    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)
    # Output layout: [sel_wall_data (5), sel_onehot (max_walls), sel_bsp_rank (1), sort_done (1)]
    got_sel_rank = out[sorted_pos, 5 + _MAX_WALLS].item()
    got_sort_done = out[sorted_pos, 5 + _MAX_WALLS + 1].item()
    assert abs(got_sel_rank - expected_sel) < 0.3, (
        f"{label}: sel_bsp_rank={got_sel_rank:+.3f}, expected {expected_sel}"
    )
    assert got_sort_done * expected_sort_done > 0.5, (
        f"{label}: sort_done={got_sort_done:+.3f}, expected sign {expected_sort_done}"
    )


# ---------------------------------------------------------------------------
# Annotation checks — the inline Asserts in _argmin_above_and_derive
# ---------------------------------------------------------------------------


def test_tied_wall_scores_raise_at_reference_eval():
    """Two WALL positions with equal sort_score should fire assert_distinct_across."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_distinct_across

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_distinct_across(sort_score, is_wall, margin=0.5)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.1], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [1.0], [0.0]]),
    }
    with pytest.raises(AssertionError, match=r"valid-subset rows"):
        reference_eval(checked, input_values, n_pos=3)


def test_score_gap_fires_at_reference_eval():
    """Two WALL scores within the softmax-resolvability margin should fire."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.04], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [1.0], [0.0]]),
    }
    with pytest.raises(AssertionError, match=r"min-gap"):
        reference_eval(checked, input_values, n_pos=3)


def test_score_gap_passes_with_comfortable_margin():
    """Scores spaced well above the gap margin must pass reference eval."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.5], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [1.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)


def test_score_gap_vacuous_zero_valid():
    """Zero valid rows → vacuous pass."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.0], [1.0]]),
        "is_wall":    torch.tensor([[0.0], [0.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)


def test_score_gap_vacuous_one_valid():
    """Single valid row → vacuous pass (no pair to check)."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.0], [1.0]]),
        "is_wall":    torch.tensor([[1.0], [0.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)


def test_score_gap_ignores_non_wall_ties():
    """Ties between a valid and an invalid row are ignored."""
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1, value_range=(0.0, 100.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.0], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [0.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)
