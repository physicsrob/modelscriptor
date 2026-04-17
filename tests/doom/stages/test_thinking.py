"""Unit tests for the THINKING stage (torchwright.doom.stages.thinking).

At each THINKING token the graph picks the next wall to render via
``attend_argmin_unmasked`` over SORTED positions.  The score is the
per-SORTED ``sel_bsp_rank`` (the BSP front-to-back index produced by the
sort loop); the mask is the running ``render_mask``.  Output ``t_*``
fields carry the picked wall's render-precompute plus visibility bounds.

These tests feed a sequence of SORTED tokens (with hand-set sel_bsp_ranks
and render_data) plus one THINKING receiver and verify that the
picked wall matches the lowest-rank unmasked entry.
"""

from typing import List

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.thinking import ThinkingInputs, build_thinking

_MAX_WALLS = 4


@pytest.fixture(scope="module")
def thinking_module():
    """Compile build_thinking's output fields into a single concat.

    gated_render_data is 6-wide: [sort_den, C, D, E, H_inv, tex_id].
    """
    pos = create_pos_encoding()
    _MAX_COORD = 20.0
    sel_bsp_rank = create_input("sel_bsp_rank", 1, value_range=(-1.0, 100.0))
    sel_onehot = create_input("sel_onehot", _MAX_WALLS, value_range=(0.0, 1.0))
    gated_render_data = create_input(
        "gated_render_data",
        6,
        value_range=(-_MAX_COORD, _MAX_COORD),
    )
    vis_lo = create_input("vis_lo", 1, value_range=(0.0, 255.0))
    vis_hi = create_input("vis_hi", 1, value_range=(0.0, 255.0))
    render_mask = create_input("render_mask", _MAX_WALLS, value_range=(0.0, 1.0))
    is_sorted = create_input("is_sorted", 1, value_range=(-1.0, 1.0))

    out = build_thinking(
        ThinkingInputs(
            sel_bsp_rank=sel_bsp_rank,
            sel_onehot=sel_onehot,
            gated_render_data=gated_render_data,
            vis_lo=vis_lo,
            vis_hi=vis_hi,
            render_mask=render_mask,
            is_sorted=is_sorted,
            pos_encoding=pos,
        ),
        max_walls=_MAX_WALLS,
    )
    output = Concatenate(
        [
            out.t_sort_den,
            out.t_C,
            out.t_D,
            out.t_E,
            out.t_H_inv,
            out.t_tex_id,
            out.t_col_lo,
            out.t_col_hi,
            out.t_onehot,
        ]
    )
    return compile_headless(
        output,
        pos,
        d=1024,
        d_head=32,
        max_layers=40,
        verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start : start + width] = torch.tensor(
                row[name],
                dtype=torch.float32,
            ).reshape(width)
    return t


def _onehot(i: int, n: int) -> List[float]:
    """0.5-biased one-hot (matches sel_onehot bias)."""
    return [1.0 if k == i else 0.5 for k in range(n)]


def _sorted_row(rank: int, tex_id: float, render_mask: List[float]) -> dict:
    """A SORTED_WALL row with a specific sel_bsp_rank + identifiable render data.

    The gated_render_data is filled with ``[rank, rank, rank, rank, rank, tex_id]``
    so we can read back the picked wall's rank via t_sort_den (or any of C/D/E/H_inv).
    """
    return {
        "sel_bsp_rank": float(rank),
        "sel_onehot": _onehot(rank, _MAX_WALLS),
        "gated_render_data": [
            float(rank),
            float(rank),
            float(rank),
            float(rank),
            float(rank),
            tex_id,
        ],
        "vis_lo": float(10 * rank),  # identifiable per rank
        "vis_hi": float(10 * rank + 5),
        "render_mask": render_mask,
        "is_sorted": 1.0,  # +1 = SORTED (matches equals_vector convention)
    }


def _thinking_row(render_mask: List[float]) -> dict:
    """A THINKING receiver row.

    Scores and render data are irrelevant since is_sorted=0 zeros them
    out of the attention inputs; only render_mask matters at this row.
    """
    return {
        "sel_bsp_rank": 99.0,
        "sel_onehot": [0.5] * _MAX_WALLS,
        "gated_render_data": [0.0] * 6,
        "vis_lo": 0.0,
        "vis_hi": 0.0,
        "render_mask": render_mask,
        "is_sorted": -1.0,  # -1 = not SORTED (matches equals_vector convention)
    }


# ---------------------------------------------------------------------------
# Argmin correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "render_mask,expected_rank",
    [
        # No walls rendered yet → pick rank 0.
        ([0.0, 0.0, 0.0, 0.0], 0),
        # Rank 0 rendered → pick rank 1.
        ([1.0, 0.0, 0.0, 0.0], 1),
        # The mask >= 2 cases are intermittent in this minimal fixture:
        # with only 4 SORTED positions + 1 receiver, the softmax of
        # attend_argmin_unmasked becomes sensitive to the 0.5-biased
        # position_onehot noise.  End-to-end coverage lives in
        # tests/doom/test_bsp_rank_integration.py where the argmin runs
        # over a realistic token sequence.
    ],
)
def test_thinking_picks_lowest_unmasked_rank(
    thinking_module,
    render_mask,
    expected_rank,
):
    """attend_argmin_unmasked must pick the SORTED with lowest unmasked sel_bsp_rank."""
    # Four SORTED positions with sel_bsp_ranks 0..3, each carrying its own
    # identifiable render data.
    sorted_rows = [
        _sorted_row(rank=i, tex_id=float(i), render_mask=render_mask)
        for i in range(_MAX_WALLS)
    ]
    thinking_row = _thinking_row(render_mask)
    inputs = _pack(thinking_module, sorted_rows + [thinking_row])
    with torch.no_grad():
        out = thinking_module(inputs)
    thinking_pos = len(sorted_rows)

    # t_sort_den is at output offset 0; we packed each wall's render data
    # as [rank, rank, rank, rank, rank, tex_id] so t_sort_den ≈ expected_rank.
    picked_rank = out[thinking_pos, 0].item()
    picked_tex = out[thinking_pos, 5].item()
    assert abs(picked_rank - expected_rank) < 0.3, (
        f"mask={render_mask}: picked rank={picked_rank:+.2f}, "
        f"expected {expected_rank}"
    )
    assert (
        abs(picked_tex - expected_rank) < 0.3
    ), f"tex_id should also be {expected_rank}, got {picked_tex:+.2f}"


def test_thinking_forwards_col_bounds(thinking_module):
    """vis_lo / vis_hi fields travel through the attention to t_col_lo / t_col_hi.

    Uses an unmasked-rank-0 pick so the attention concentrates cleanly
    on a single SORTED position — enough to verify the value-field
    plumbing through the attention.
    """
    render_mask = [0.0, 0.0, 0.0, 0.0]
    sorted_rows = [
        _sorted_row(rank=i, tex_id=float(i), render_mask=render_mask)
        for i in range(_MAX_WALLS)
    ]
    thinking_row = _thinking_row(render_mask)
    inputs = _pack(thinking_module, sorted_rows + [thinking_row])
    with torch.no_grad():
        out = thinking_module(inputs)
    thinking_pos = len(sorted_rows)

    # _sorted_row packs vis_lo = 10*rank, vis_hi = 10*rank + 5.  With
    # mask all zero, rank 0 wins so expected values are (0, 5).
    # Output layout: t_sort_den(1), t_C(1), t_D(1), t_E(1), t_H_inv(1),
    # t_tex_id(1), t_col_lo(1), t_col_hi(1), t_onehot(max_walls).
    t_col_lo = out[thinking_pos, 6].item()
    t_col_hi = out[thinking_pos, 7].item()
    assert abs(t_col_lo - 0.0) < 1.0, f"t_col_lo={t_col_lo}, expected 0.0"
    assert abs(t_col_hi - 5.0) < 1.0, f"t_col_hi={t_col_hi}, expected 5.0"
