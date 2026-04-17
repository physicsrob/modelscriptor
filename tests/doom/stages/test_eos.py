"""Unit tests for the EOS stage (torchwright.doom.stages.eos).

The EOS stage has two jobs:

1. **Collision resolution**: aggregate per-WALL hit flags across the
   WALL positions and decide which axes of the velocity are safe to
   apply.  Tested against ``reference_renderer.collision._ray_hits_any_segment``-style
   logic.
2. **State broadcast**: copy the resolved ``(x, y, angle)`` at the single
   EOS position out to every position via ``attend_mean_where``.

Tests feed multi-position sequences where some positions are WALLs (with
preset hit flags) and one position is the EOS token.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.eos import EosInputs, build_eos
from torchwright.doom.stages.wall import CollisionFlags

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eos_module():
    """Compile build_eos's resolved_x, resolved_y, broadcast (px, py, angle).

    The collision flags are synthesized as plain ``create_input`` nodes so
    tests can set them per-position without going through the WALL stage.
    """
    pos = create_pos_encoding()

    _MAX_COORD = 20.0
    hit_full = create_input("hit_full", 1, value_range=(-1.0, 1.0))
    hit_x = create_input("hit_x", 1, value_range=(-1.0, 1.0))
    hit_y = create_input("hit_y", 1, value_range=(-1.0, 1.0))
    is_wall = create_input("is_wall", 1, value_range=(-1.0, 1.0))
    is_eos = create_input("is_eos", 1, value_range=(-1.0, 1.0))
    player_x = create_input("player_x", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    player_y = create_input("player_y", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    vel_dx = create_input("vel_dx", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    vel_dy = create_input("vel_dy", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    new_angle = create_input("new_angle", 1, value_range=(0.0, 255.0))

    out = build_eos(
        EosInputs(
            is_wall=is_wall,
            is_eos=is_eos,
            collision=CollisionFlags(hit_full=hit_full, hit_x=hit_x, hit_y=hit_y),
            player_x=player_x,
            player_y=player_y,
            vel_dx=vel_dx,
            vel_dy=vel_dy,
            new_angle=new_angle,
            pos_encoding=pos,
        )
    )
    output = Concatenate(
        [
            out.resolved_x,
            out.resolved_y,
            out.px,
            out.py,
            out.angle,
        ]
    )
    return compile_headless(
        output,
        pos,
        d=1024,
        d_head=16,
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


def _wall_row(
    hit_full: bool, hit_x: bool, hit_y: bool, px, py, vdx, vdy, angle
) -> dict:
    """One WALL token row: ±1 collision flags, is_wall=1."""
    return {
        "hit_full": 1.0 if hit_full else -1.0,
        "hit_x": 1.0 if hit_x else -1.0,
        "hit_y": 1.0 if hit_y else -1.0,
        "is_wall": 1.0,
        "is_eos": 0.0,
        "player_x": px,
        "player_y": py,
        "vel_dx": vdx,
        "vel_dy": vdy,
        "new_angle": angle,
    }


def _eos_row(px, py, vdx, vdy, angle) -> dict:
    return {
        "hit_full": -1.0,
        "hit_x": -1.0,
        "hit_y": -1.0,  # gated anyway
        "is_wall": 0.0,
        "is_eos": 1.0,
        "player_x": px,
        "player_y": py,
        "vel_dx": vdx,
        "vel_dy": vdy,
        "new_angle": angle,
    }


# ---------------------------------------------------------------------------
# Wall-sliding scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        # No walls hit: full velocity applied.
        dict(
            name="no_hit_full_move",
            walls=[(False, False, False), (False, False, False)],
            px=1.0,
            py=2.0,
            vdx=0.2,
            vdy=0.1,
            expected_x=1.2,
            expected_y=2.1,
        ),
        # Full hit + both axes blocked: stay put (worst case).
        dict(
            name="full_and_both_axes_blocked",
            walls=[(True, True, True), (False, False, False)],
            px=1.0,
            py=2.0,
            vdx=0.2,
            vdy=0.1,
            expected_x=1.0,
            expected_y=2.0,
        ),
        # Full hit but x is clear: slide along x.
        dict(
            name="slide_x_only",
            walls=[(True, False, True), (False, False, False)],
            px=1.0,
            py=2.0,
            vdx=0.2,
            vdy=0.1,
            expected_x=1.2,
            expected_y=2.0,
        ),
        # Full hit but y is clear: slide along y.
        dict(
            name="slide_y_only",
            walls=[(True, True, False), (False, False, False)],
            px=1.0,
            py=2.0,
            vdx=0.2,
            vdy=0.1,
            expected_x=1.0,
            expected_y=2.1,
        ),
        # Multiple walls: one hits full, another hits x — combined = slide y only.
        dict(
            name="multi_wall_combined_blocks",
            walls=[(True, False, False), (False, True, False)],
            px=1.0,
            py=2.0,
            vdx=0.2,
            vdy=0.1,
            expected_x=1.0,
            expected_y=2.1,
        ),
    ],
)
def test_collision_resolution_matches_logic(eos_module, scenario):
    """Axis-sliding decisions must match the per-axis logic."""
    wall_rows = [
        _wall_row(
            hf,
            hx,
            hy,
            px=scenario["px"],
            py=scenario["py"],
            vdx=scenario["vdx"],
            vdy=scenario["vdy"],
            angle=0.0,
        )
        for (hf, hx, hy) in scenario["walls"]
    ]
    eos_row = _eos_row(
        px=scenario["px"],
        py=scenario["py"],
        vdx=scenario["vdx"],
        vdy=scenario["vdy"],
        angle=0.0,
    )
    inputs = _pack(eos_module, wall_rows + [eos_row])
    with torch.no_grad():
        out = eos_module(inputs)
    eos_pos = len(wall_rows)  # EOS row is last
    resolved_x = out[eos_pos, 0].item()
    resolved_y = out[eos_pos, 1].item()
    assert abs(resolved_x - scenario["expected_x"]) < 0.05, (
        f"{scenario['name']}: resolved_x={resolved_x:+.3f}, "
        f"expected {scenario['expected_x']:+.3f}"
    )
    assert abs(resolved_y - scenario["expected_y"]) < 0.05, (
        f"{scenario['name']}: resolved_y={resolved_y:+.3f}, "
        f"expected {scenario['expected_y']:+.3f}"
    )


def test_state_broadcast_reaches_non_eos_positions(eos_module):
    """Resolved (x, y, angle) at EOS should be visible at every position."""
    walls = [
        _wall_row(False, False, False, px=3.0, py=4.0, vdx=0.1, vdy=0.0, angle=42.0)
    ]
    eos = _eos_row(px=3.0, py=4.0, vdx=0.1, vdy=0.0, angle=42.0)
    # Receiver position (e.g. another SORTED position); the broadcast reaches it.
    receiver = _wall_row(
        False, False, False, px=0.0, py=0.0, vdx=0.0, vdy=0.0, angle=0.0
    )
    receiver["is_wall"] = 0.0  # neither WALL nor EOS

    inputs = _pack(eos_module, walls + [eos, receiver])
    with torch.no_grad():
        out = eos_module(inputs)
    # Broadcast columns: px, py, angle are outputs 2, 3, 4.
    px_at_receiver = out[2, 2].item()
    py_at_receiver = out[2, 3].item()
    angle_at_receiver = out[2, 4].item()
    # With no hits, resolved = player + velocity = (3.1, 4.0).
    assert abs(px_at_receiver - 3.1) < 0.05, f"px={px_at_receiver:+.3f}"
    assert abs(py_at_receiver - 4.0) < 0.05, f"py={py_at_receiver:+.3f}"
    assert abs(angle_at_receiver - 42.0) < 1.0, f"angle={angle_at_receiver:+.3f}"
