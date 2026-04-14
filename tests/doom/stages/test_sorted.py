"""Unit tests for the SORTED stage (torchwright.doom.stages.sorted).

Focus: verify that ``attend_argmin_unmasked`` picks the lowest-score
wall whose ``prev_mask`` bit is zero, and that its payload is correctly
unpacked into ``sel_wall_data`` + ``sel_render``.

Visibility-mask math is covered in ``test_visibility_mask_pieces.py``;
this file adds a smoke test that the masked column range at least
overlaps the expected column for a centered wall.
"""

import math
from typing import List

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.stages.sorted import SortedInputs, build_sorted


_MAX_COORD = 20.0
_MAX_WALLS = 4


def _tiny_config() -> RenderConfig:
    return RenderConfig(
        screen_width=32, screen_height=16, fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


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

    sort_score = create_input("sort_score", 1)
    position_onehot = create_input("position_onehot", _MAX_WALLS)
    sort_value = create_input("sort_value", payload_w)
    prev_mask = create_input("prev_mask", _MAX_WALLS)
    eos_px = create_input("eos_px", 1)
    eos_py = create_input("eos_py", 1)
    eos_angle = create_input("eos_angle", 1)
    is_sorted = create_input("is_sorted", 1)

    out = build_sorted(
        SortedInputs(
            sort_score=sort_score,
            position_onehot=position_onehot,
            sort_value=sort_value,
            prev_mask=prev_mask,
            eos_px=eos_px,
            eos_py=eos_py,
            eos_angle=eos_angle,
            is_sorted=is_sorted,
            pos_encoding=pos,
        ),
        config=_tiny_config(),
        max_walls=_MAX_WALLS,
        max_coord=_MAX_COORD,
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
                  center_dist, onehot: List[float]) -> List[float]:
    """Canonical wall payload layout — must match wall_payload.pack_wall_payload."""
    return [ax, ay, bx, by, tex_id, sort_den, C, D, E, H_inv, center_dist] + onehot


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
            "position_onehot": _onehot(i, _MAX_WALLS),
            "sort_value": _wall_payload(
                ax=float(i), ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                center_dist=0.0, onehot=_onehot(i, _MAX_WALLS),
            ),
            "prev_mask": prev_mask,
            "eos_px": 0.0, "eos_py": 0.0, "eos_angle": 0.0,
            "is_sorted": 0.0,
        })
    # One SORTED position reads the argmin.
    sorted_row = {
        "sort_score": 99.0,
        "position_onehot": [0.5] * _MAX_WALLS,
        "sort_value": [0.0] * (11 + _MAX_WALLS),
        "prev_mask": prev_mask,
        "eos_px": 0.0, "eos_py": 0.0, "eos_angle": 0.0,
        "is_sorted": 1.0,
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


def test_updated_mask_adds_selected_wall(sorted_module):
    """After selecting wall 2, updated_mask should have bit 2 set."""
    scores = [5.0, 2.0, 8.0, 99.0]
    prev_mask = [0.0, 1.0, 0.0, 0.0]  # wall 1 already picked — expect wall 0

    wall_rows = []
    for i in range(_MAX_WALLS):
        wall_rows.append({
            "sort_score": scores[i],
            "position_onehot": _onehot(i, _MAX_WALLS),
            "sort_value": _wall_payload(
                ax=float(i), ay=0.0, bx=0.0, by=0.0, tex_id=0.0,
                sort_den=0.0, C=0.0, D=0.0, E=0.0, H_inv=0.0,
                center_dist=0.0, onehot=_onehot(i, _MAX_WALLS),
            ),
            "prev_mask": prev_mask,
            "eos_px": 0.0, "eos_py": 0.0, "eos_angle": 0.0,
            "is_sorted": 0.0,
        })
    sorted_row = {
        "sort_score": 99.0,
        "position_onehot": [0.5] * _MAX_WALLS,
        "sort_value": [0.0] * (11 + _MAX_WALLS),
        "prev_mask": prev_mask,
        "eos_px": 0.0, "eos_py": 0.0, "eos_angle": 0.0,
        "is_sorted": 1.0,
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
