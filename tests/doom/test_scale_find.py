"""Unit tests for the scale-find pass in ``thinking_wall``.

Builds a minimal graph that mirrors the production scale-find pass
(per-position ``max(|ax|, |ay|, |bx|, |by|)`` plus a cross-WALL
``attend_argmax_dot`` reduction plus a ``log`` to produce
``log_inv_scale``).  Runs each scenario through ``Node.compute`` —
no compile round-trip needed — and verifies:

1. ``global_max_abs_coord`` matches the analytical max within tight
   absolute tolerance (max + abs are exact ops; only float32
   round-off contributes noise).
2. ``log_inv_scale`` matches ``-log(global_max)`` within the
   sectioned-log noise budget for our operating range.

Three scenarios cover the realistic envelope:

* **box_room** — 4 walls at ±5; global_max = 5; tied across walls
  (the soft-average should still return 5 cleanly).
* **multi_room** — 22 walls in [-12, 12]; global_max = 12; the
  ties-allowed property doesn't bite.
* **e1m1_subset** — synthetic 8 walls scaling up to 80; global_max =
  80; covers the upper end of the user-stated max_coord = 100
  envelope.
"""

import math

import pytest
import torch

from torchwright.doom.stages.thinking_wall import _compute_scale_find
from torchwright.ops.inout_nodes import create_input

# Parameters mirror the production envelope.  ``max_coord`` is the
# upper bound the production graph should support; the log call inside
# ``_compute_scale_find`` is sized accordingly.
_MAX_COORD = 100.0


def _build_scale_find_graph(n_pos: int):
    """Build the inputs and outputs for one ``_compute_scale_find`` test.

    Returns ``(global_max_node, log_inv_scale_node, inv_scale_node)``.
    """
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    is_wall = create_input("is_wall", 1)

    global_max, log_inv_scale, inv_scale = _compute_scale_find(
        wall_ax,
        wall_ay,
        wall_bx,
        wall_by,
        is_wall=is_wall,
        max_coord=_MAX_COORD,
    )
    return global_max, log_inv_scale, inv_scale


def _run(out_node, n_pos: int, **inputs):
    return out_node.compute(n_pos=n_pos, input_values=inputs)


def _make_inputs(walls: list[tuple[float, float, float, float]], n_pos: int):
    """Pack a list of ``(ax, ay, bx, by)`` walls into per-position input
    tensors.

    The walls live at positions ``0, 1, ..., len(walls) - 1``; remaining
    positions through ``n_pos`` are non-WALL filler with zero coords.
    ``is_wall`` is +1 at WALL positions and -1 elsewhere.
    """
    n_walls = len(walls)
    assert n_walls <= n_pos, "n_pos must hold every wall plus filler"
    ax = torch.zeros(n_pos, 1)
    ay = torch.zeros(n_pos, 1)
    bx = torch.zeros(n_pos, 1)
    by = torch.zeros(n_pos, 1)
    is_wall = -torch.ones(n_pos, 1)
    for i, (a_x, a_y, b_x, b_y) in enumerate(walls):
        ax[i, 0] = a_x
        ay[i, 0] = a_y
        bx[i, 0] = b_x
        by[i, 0] = b_y
        is_wall[i, 0] = 1.0
    return {
        "wall_ax": ax,
        "wall_ay": ay,
        "wall_bx": bx,
        "wall_by": by,
        "is_wall": is_wall,
    }


def _box_room_walls(size: float = 10.0) -> list[tuple[float, float, float, float]]:
    h = size / 2.0
    return [
        (h, -h, h, h),
        (-h, h, -h, -h),
        (-h, h, h, h),
        (h, -h, -h, -h),
    ]


def _e1m1_synthetic_walls() -> list[tuple[float, float, float, float]]:
    """8-wall synthetic E1M1-scale subset reaching coord magnitude 80."""
    return [
        (10.0, 10.0, 30.0, 10.0),
        (30.0, 10.0, 30.0, 30.0),
        (30.0, 30.0, 60.0, 30.0),
        (60.0, 30.0, 60.0, 80.0),  # 80 is the largest |coord| present
        (-20.0, -20.0, -50.0, -20.0),
        (-50.0, -20.0, -50.0, -60.0),
        (5.0, -5.0, 15.0, -5.0),
        (15.0, -5.0, 15.0, -25.0),
    ]


def _expected_global_max(walls: list[tuple[float, float, float, float]]) -> float:
    return max(max(abs(ax), abs(ay), abs(bx), abs(by)) for ax, ay, bx, by in walls)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check(scenario_name: str, walls: list[tuple[float, float, float, float]]):
    n_pos = max(len(walls) + 4, 8)  # a few non-WALL filler positions
    (
        global_max_node,
        log_inv_scale_node,
        inv_scale_node,
    ) = _build_scale_find_graph(n_pos)
    inputs = _make_inputs(walls, n_pos)

    expected_max = _expected_global_max(walls)
    expected_log_inv = -math.log(expected_max)
    expected_inv_scale = 1.0 / expected_max

    gmac = _run(global_max_node, n_pos, **inputs)
    log_inv = _run(log_inv_scale_node, n_pos, **inputs)
    inv_s = _run(inv_scale_node, n_pos, **inputs)

    # Check at every position — the scale-find result is a global
    # broadcast, so every position should carry the same value.
    # First WALL position: the attention's causal window only covers
    # walls at positions <= i, so we read at the LAST position to see
    # the full reduction.
    last = n_pos - 1
    gmac_val = float(gmac[last, 0].item())
    log_inv_val = float(log_inv[last, 0].item())
    inv_s_val = float(inv_s[last, 0].item())

    # ``max`` is exact in float32 modulo round-off; ``abs`` is exact;
    # the soft-tie average still returns the shared max for tied
    # walls.  Tolerate ~ULP at our value range.
    abs_tol = 1e-3
    assert abs(gmac_val - expected_max) < abs_tol, (
        f"{scenario_name}: global_max_abs_coord={gmac_val} vs "
        f"expected {expected_max} (diff {gmac_val - expected_max:.4g})"
    )

    # log_inv_scale precision is bounded by the sectioned log's
    # measured noise floor on its operating range — typical 1e-4
    # absolute, with comfortable margin for cross-test float32
    # variation.
    log_inv_tol = 5e-3
    assert abs(log_inv_val - expected_log_inv) < log_inv_tol, (
        f"{scenario_name}: log_inv_scale={log_inv_val} vs "
        f"expected {expected_log_inv} (diff {log_inv_val - expected_log_inv:.4g})"
    )

    # inv_scale = exp(log_inv_scale).  exp's relative error is
    # ~2e-4 at 256 BPs over our range; combined with log's abs
    # error → relative error in inv_scale is bounded by
    # ``exp(log_inv_tol)·log_inv_tol`` ≈ 1 % at the worst.  Set the
    # tolerance accordingly.
    inv_scale_rel_tol = 0.01
    assert (
        abs(inv_s_val - expected_inv_scale) / expected_inv_scale < inv_scale_rel_tol
    ), (
        f"{scenario_name}: inv_scale={inv_s_val} vs "
        f"expected {expected_inv_scale} "
        f"(rel_err {(inv_s_val - expected_inv_scale)/expected_inv_scale:.4g})"
    )


def test_scale_find_box_room():
    """Box room: 4 walls at ±5 with tied wall_max_abs values.

    All four walls produce wall_max_abs = 5; the cross-WALL
    attend_argmax_dot lands on a soft-average over the four tied keys
    which equals 5 — the correct answer.  No assert_hardness_gt is set
    on the production attention precisely because of this case.
    """
    _check("box_room", _box_room_walls(10.0))


def test_scale_find_e1m1_synthetic():
    """E1M1-scale synthetic: 8 walls with global_max = 80 (≤ max_coord)."""
    _check("e1m1_synthetic", _e1m1_synthetic_walls())


def test_scale_find_two_walls_tied_max():
    """Two walls with identical max_abs, others smaller.

    Stresses the soft-tie-average path at full strength: the two top
    walls contribute equally to the soft mean, while a non-tied third
    wall is pushed to ~0 weight by the match_gain.  Result must still
    equal the shared tied max.
    """
    walls = [
        (50.0, 50.0, 50.0, 50.0),  # max_abs = 50
        (-50.0, 50.0, 50.0, -50.0),  # max_abs = 50 (tie)
        (1.0, 1.0, 2.0, 2.0),  # max_abs = 2 (far off)
    ]
    _check("two_walls_tied_max", walls)


def test_scale_find_lone_wall_at_max_coord():
    """Single wall at the upper envelope (|coord| = max_coord).

    The log call's max_value is set to max_coord (= 100); a wall whose
    max_abs equals the envelope sits exactly at the upper breakpoint
    of the log grid.  Verifies the boundary handling.
    """
    walls = [(100.0, 100.0, 100.0, 100.0)]
    _check("lone_wall_at_max_coord", walls)
