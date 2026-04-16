"""Unit tests for the SORTED stage (torchwright.doom.stages.sorted).

Focus: verify that ``attend_argmin_unmasked`` picks the lowest-score
wall whose ``prev_mask`` bit is zero, and that its payload is correctly
unpacked into ``sel_wall_data`` + ``sel_render``.
"""

from typing import List

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.graph.asserts import assert_01, assert_integer, assert_onehot
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.sorted import SortedInputs, build_sorted


_MAX_WALLS = 4


@pytest.fixture(scope="module")
def sorted_module():
    """Compile build_sorted's outputs.

    sort_value (the packed payload) and prev_mask/position_onehot are
    exposed as multi-wide ``create_input`` nodes so the test controls
    every position independently.
    """
    from torchwright.doom.wall_payload import payload_width

    pos = create_pos_encoding()
    payload_w = payload_width(_MAX_WALLS)

    sort_score = assert_integer(create_input("sort_score", 1))
    is_renderable = create_input("is_renderable", 1)
    position_onehot = assert_onehot(create_input("position_onehot", _MAX_WALLS))
    sort_value = create_input("sort_value", payload_w)
    prev_mask = assert_01(create_input("prev_mask", _MAX_WALLS))
    prev_bsp_rank = create_input("prev_bsp_rank", 1)
    is_sorted = create_input("is_sorted", 1)
    is_wall = create_input("is_wall", 1)

    out = build_sorted(
        SortedInputs(
            sort_score=sort_score,
            is_renderable=is_renderable,
            position_onehot=position_onehot,
            sort_value=sort_value,
            prev_mask=prev_mask,
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
        out.updated_mask,        # max_walls-wide
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
                  center_dist, vis_lo=0.0, vis_hi=0.0,
                  onehot: List[float] = ()) -> List[float]:
    """Canonical wall payload layout — must match wall_payload.pack_wall_payload."""
    return ([ax, ay, bx, by, tex_id, sort_den, C, D, E, H_inv, center_dist,
             vis_lo, vis_hi] + list(onehot))


def _onehot(i: int, n: int) -> List[float]:
    """0.5-biased one-hot (matches wall_payload.add_scaled_nodes output)."""
    return [1.0 if k == i else 0.5 for k in range(n)]


# ---------------------------------------------------------------------------
# Argmin correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scores,prev_mask,expected_idx", [
    # 3 walls with distinct scores.  Lowest = wall 1 (score=2).
    ([5.0, 2.0, 8.0, 99.0], [0.0] * _MAX_WALLS, 1),
    # Mask out wall 1: next-lowest is wall 0 (score=5).
    ([5.0, 2.0, 8.0, 99.0], [0.0, 1.0, 0.0, 0.0], 0),
    # Mask out walls 0 and 1: picks wall 2 (score=8).
    ([5.0, 2.0, 8.0, 99.0], [1.0, 1.0, 0.0, 0.0], 2),
])
def test_argmin_picks_lowest_unmasked(sorted_module, scores, prev_mask, expected_idx):
    """attend_argmin_unmasked must select the wall with the lowest score whose
    mask bit is 0."""
    # Build wall rows: each WALL position has its own score, onehot, sort_value.
    # Payload's ax field is set to the wall's index so we can verify the pick.
    wall_rows = []
    for i in range(_MAX_WALLS):
        wall_rows.append({
            "sort_score": scores[i],
            "is_renderable": 1.0,
            "position_onehot": _onehot(i, _MAX_WALLS),
            "sort_value": _wall_payload(
                ax=float(i), ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                center_dist=0.0, onehot=_onehot(i, _MAX_WALLS),
            ),
            "prev_mask": prev_mask,
            "prev_bsp_rank": -1.0,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })
    # One SORTED position reads the argmin.
    sorted_row = {
        "sort_score": 99.0,
        "is_renderable": -1.0,
        "position_onehot": [0.5] * _MAX_WALLS,
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "prev_mask": prev_mask,
        "prev_bsp_rank": -1.0,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }
    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)
    # sel_wall_data[0] = selected wall's ax = expected wall index.
    sel_ax = out[sorted_pos, 0].item()
    assert abs(sel_ax - expected_idx) < 0.3, (
        f"Expected wall {expected_idx}, got sel_ax={sel_ax:+.2f} "
        f"(scores={scores}, prev_mask={prev_mask})"
    )


def test_angle_192_validity_excludes_invalid_clean_pick(sorted_module):
    """At angle=192 only the south wall is renderable; the argmin must
    pick it cleanly regardless of how dense the invalid field is.

    Previously this scenario — three unrenderable walls all tied on a
    ``bsp_sentinel`` (only separated by a ``wall_index * 0.1`` tiebreak)
    — forced the softmax to fight for the last 1.4 % of mass in a 0.1-
    wide logit budget.  With ``is_renderable`` surfaced as a per-key
    validity signal, unrenderable walls are now first-class invalid
    keys and the score can be a clean integer rank.  The pick is a hard
    argmax of one valid position.
    """
    walls = [
        (5.0, -5.0, 5.0, 5.0),     # wall 0: east (parallel to viewing ray)
        (5.0, 5.0, -5.0, 5.0),     # wall 1: north (behind player)
        (-5.0, 5.0, -5.0, -5.0),   # wall 2: west (parallel to viewing ray)
        (-5.0, -5.0, 5.0, -5.0),   # wall 3: south (renderable)
    ]
    # Clean integer rank: south is BSP-rank 0.  Other scores irrelevant
    # (the keys are invalid), but must still be pairwise distinct for
    # assert_distinct_across.
    scores = [3.0, 4.0, 5.0, 0.0]
    is_renderable = [-1.0, -1.0, -1.0, 1.0]  # only south is valid
    prev_mask = [0.0, 0.0, 0.0, 0.0]

    def pure_onehot(i: int, n: int) -> list[float]:
        return [1.0 if k == i else 0.0 for k in range(n)]

    wall_rows = []
    for i, (ax, ay, bx, by) in enumerate(walls):
        wall_rows.append({
            "sort_score": scores[i],
            "is_renderable": is_renderable[i],
            "position_onehot": pure_onehot(i, _MAX_WALLS),
            "sort_value": _wall_payload(
                ax=ax, ay=ay, bx=bx, by=by, tex_id=float(i),
                sort_den=1.0, C=0.0, D=0.0, E=0.0, H_inv=1.0,
                center_dist=0.0, onehot=pure_onehot(i, _MAX_WALLS),
            ),
            "prev_mask": prev_mask,
            "prev_bsp_rank": -1.0,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })

    sorted_row = {
        "sort_score": 0.0,
        "is_renderable": -1.0,
        "position_onehot": pure_onehot(0, _MAX_WALLS),
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "prev_mask": prev_mask,
        "prev_bsp_rank": -1.0,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }

    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)

    # Expected: south wall (wall 3) picked cleanly — no blending possible.
    sel_ax, sel_ay, sel_bx, sel_by, sel_tex = out[sorted_pos, :5].tolist()
    tol = 0.05
    assert abs(sel_ax - (-5.0)) < tol, f"sel_ax={sel_ax:+.4f} (expected -5.0)"
    assert abs(sel_ay - (-5.0)) < tol, f"sel_ay={sel_ay:+.4f} (expected -5.0)"
    assert abs(sel_bx - 5.0) < tol, f"sel_bx={sel_bx:+.4f} (expected 5.0)"
    assert abs(sel_by - (-5.0)) < tol, f"sel_by={sel_by:+.4f} (expected -5.0)"
    assert abs(sel_tex - 3.0) < tol, f"sel_tex={sel_tex:+.4f} (expected 3.0)"


def test_updated_mask_adds_selected_wall(sorted_module):
    """After selecting wall 2, updated_mask should have bit 2 set."""
    scores = [5.0, 2.0, 8.0, 99.0]
    prev_mask = [0.0, 1.0, 0.0, 0.0]  # wall 1 already picked — expect wall 0

    wall_rows = []
    for i in range(_MAX_WALLS):
        wall_rows.append({
            "sort_score": scores[i],
            "is_renderable": 1.0,
            "position_onehot": _onehot(i, _MAX_WALLS),
            "sort_value": _wall_payload(
                ax=float(i), ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                center_dist=0.0, onehot=_onehot(i, _MAX_WALLS),
            ),
            "prev_mask": prev_mask,
            "prev_bsp_rank": -1.0,
            "is_sorted": 0.0,
            "is_wall": 1.0,
        })
    sorted_row = {
        "sort_score": 99.0,
        "is_renderable": -1.0,
        "position_onehot": [0.5] * _MAX_WALLS,
        "sort_value": [0.0] * (13 + _MAX_WALLS),
        "prev_mask": prev_mask,
        "prev_bsp_rank": -1.0,
        "is_sorted": 1.0,
        "is_wall": 0.0,
    }
    inputs = _pack(sorted_module, wall_rows + [sorted_row])
    with torch.no_grad():
        out = sorted_module(inputs)
    sorted_pos = len(wall_rows)

    # output layout: [sel_wall_data (5), sel_onehot (max_walls), updated_mask (max_walls)]
    updated_mask = out[sorted_pos, 5 + _MAX_WALLS:5 + 2 * _MAX_WALLS].tolist()
    # updated_mask = prev_mask (0/1) + sel_onehot (0.5 bias or 1.0 peak).
    # Values: newly-picked ≈ 1.0, previously-picked ≈ 1.5, neither ≈ 0.5.
    # A threshold at 0.75 cleanly separates "masked" from "unmasked".
    assert updated_mask[0] > 0.75, f"bit 0 freshly picked, got {updated_mask[0]:+.2f}"
    assert updated_mask[1] > 0.75, f"bit 1 previously picked, got {updated_mask[1]:+.2f}"
    assert updated_mask[2] < 0.75, f"bit 2 should stay unmasked, got {updated_mask[2]:+.2f}"
    assert updated_mask[3] < 0.75, f"bit 3 should stay unmasked, got {updated_mask[3]:+.2f}"


# ---------------------------------------------------------------------------
# Annotation checks — the inline Asserts in _argmin_and_derive
# ---------------------------------------------------------------------------


def test_tied_wall_scores_raise_at_reference_eval():
    """Two WALL positions with equal sort_score should fire assert_distinct_across.

    Builds a tiny subgraph that only reaches the ``assert_distinct_across``
    wrapper — compiled end-to-end compilation is expensive and the assert
    fires at reference-eval time anyway.  We run ``reference_eval``
    directly on the asserted score node and expect an AssertionError
    whose message names the offending rows.
    """
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_distinct_across

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
    checked = assert_distinct_across(sort_score, is_wall, margin=0.5)

    # Two WALL rows with tied scores (1.0 and 1.1, margin 0.5 → tie).
    input_values = {
        "sort_score": torch.tensor([[1.0], [1.1], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [1.0], [0.0]]),
    }
    with pytest.raises(AssertionError, match=r"valid-subset rows"):
        reference_eval(checked, input_values, n_pos=3)


def test_score_gap_fires_at_reference_eval():
    """Two WALL scores within the softmax-resolvability margin should fire.

    Mirrors ``test_tied_wall_scores_raise_at_reference_eval`` but targets
    the tighter ``assert_score_gap_at_least`` invariant: scores 1.0 and
    1.04 are distinct (above ``assert_distinct_across``'s 0.5 margin when
    used standalone) but fall below the 0.05 softmax-gap margin.
    """
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
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

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
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

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
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

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.0], [1.0]]),
        "is_wall":    torch.tensor([[1.0], [0.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)


def test_score_gap_ignores_non_wall_ties():
    """Ties between a valid and an invalid row are ignored — only the
    valid subset is checked, so a singleton valid set passes vacuously.
    """
    from torchwright.debug.probe import reference_eval
    from torchwright.graph.asserts import assert_score_gap_at_least

    sort_score = create_input("sort_score", 1)
    is_wall = create_input("is_wall", 1)
    checked = assert_score_gap_at_least(sort_score, is_wall, margin=0.05)

    input_values = {
        "sort_score": torch.tensor([[1.0], [1.0], [99.0]]),
        "is_wall":    torch.tensor([[1.0], [0.0], [0.0]]),
    }
    reference_eval(checked, input_values, n_pos=3)
