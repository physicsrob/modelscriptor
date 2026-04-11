"""Tests for Phase 3: compiled game graph (game logic + rendering in transformer).

Compares the compiled transformer's state output and rendered pixels
against the Python reference implementation.
"""

import numpy as np
import pytest

from torchwright.doom.compile import compile_game, step_frame_compiled
from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer import render_frame
from torchwright.reference_renderer.collision import resolve_collision_ray
from torchwright.reference_renderer.scenes import box_room
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def small_config(trig_table):
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=trig_table,
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


@pytest.fixture
def box_segments():
    return box_room()


@pytest.fixture
def box_game_module(box_segments, small_config):
    """Compile the game graph once for box_room, reuse across tests."""
    return compile_game(
        box_segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        d=1024, d_head=16, verbose=False,
    )


def _assert_frame_match(compiled, reference, max_boundary_pixels=None, atol=0.15, msg=""):
    """Full-frame comparison allowing boundary pixel mismatches."""
    H, W = compiled.shape[:2]
    if max_boundary_pixels is None:
        max_boundary_pixels = 2 * W
    mismatched = np.abs(compiled - reference) > atol
    n_bad = mismatched.any(axis=2).sum()
    if n_bad > max_boundary_pixels:
        rows, cols = np.where(mismatched.any(axis=2))
        r, c = rows[0], cols[0]
        assert False, (
            f"{msg} {n_bad}/{H * W} pixels differ (max allowed: {max_boundary_pixels}). "
            f"First at ({r},{c}): compiled={compiled[r, c]} ref={reference[r, c]}"
        )


# ── No-input: state unchanged, pixels match static render ───────────


def test_compiled_no_input(box_game_module, box_segments, small_config):
    """No inputs: state should be unchanged and pixels should match static render."""
    state = GameState(x=0.0, y=0.0, angle=0)
    frame, new_state = step_frame_compiled(
        box_game_module, state, PlayerInput(), small_config,
    )

    # State unchanged
    assert new_state.x == pytest.approx(0.0, abs=0.1)
    assert new_state.y == pytest.approx(0.0, abs=0.1)
    assert new_state.angle == 0

    # Pixels match static reference
    ref = render_frame(0.0, 0.0, 0, box_segments, small_config)
    _assert_frame_match(frame, ref, msg="no-input")


# ── Forward movement ─────────────────────────────────────────────────


def test_compiled_forward(box_game_module, box_segments, small_config, trig_table):
    """Forward input advances position; pixels match reference at new position."""
    state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput(forward=True)

    frame, new_state = step_frame_compiled(
        box_game_module, state, inputs, small_config,
    )

    # Compare state against Python reference
    ref_state = update_state(state, inputs, box_segments, trig_table)
    assert new_state.x == pytest.approx(ref_state.x, abs=0.15)
    assert new_state.y == pytest.approx(ref_state.y, abs=0.15)
    assert new_state.angle == ref_state.angle

    # Pixels match reference at the new position
    ref = render_frame(new_state.x, new_state.y, new_state.angle, box_segments, small_config)
    _assert_frame_match(frame, ref, msg="forward")


# ── Turn ─────────────────────────────────────────────────────────────


def test_compiled_turn(box_game_module, box_segments, small_config, trig_table):
    """Turn input changes angle; frame matches reference at new angle."""
    state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput(turn_right=True)

    frame, new_state = step_frame_compiled(
        box_game_module, state, inputs, small_config,
    )

    ref_state = update_state(state, inputs, box_segments, trig_table)
    assert new_state.angle == ref_state.angle

    ref = render_frame(new_state.x, new_state.y, new_state.angle, box_segments, small_config)
    _assert_frame_match(frame, ref, msg="turn")


# ── Collision: walk into wall ────────────────────────────────────────


def test_compiled_collision(box_game_module, box_segments, small_config, trig_table):
    """Walking into a wall should be blocked (position doesn't advance past wall)."""
    state = GameState(x=4.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput(forward=True)

    frame, new_state = step_frame_compiled(
        box_game_module, state, inputs, small_config,
    )

    ref_state = update_state(state, inputs, box_segments, trig_table)

    # Both should be blocked near x=4.0 (wall at x=5, move_speed=0.3)
    assert new_state.x == pytest.approx(ref_state.x, abs=0.15)
    assert new_state.x < 5.0, f"Passed through wall: x={new_state.x}"


# ── Wall sliding ─────────────────────────────────────────────────────


def test_compiled_wall_sliding(box_game_module, box_segments, small_config, trig_table):
    """Moving diagonally into a wall should slide along it."""
    # Near east wall, facing northeast (angle ~32)
    state = GameState(x=4.5, y=0.0, angle=32, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput(forward=True)

    frame, new_state = step_frame_compiled(
        box_game_module, state, inputs, small_config,
    )

    ref_state = update_state(state, inputs, box_segments, trig_table)

    # X should be blocked (near wall), Y should advance
    assert new_state.x == pytest.approx(ref_state.x, abs=0.2)
    assert new_state.y == pytest.approx(ref_state.y, abs=0.2)


# ── Multi-frame state trajectory ─────────────────────────────────────


def test_compiled_multi_frame(box_game_module, box_segments, small_config, trig_table):
    """Run multiple frames; verify state trajectory matches Python reference."""
    state_compiled = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    state_ref = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)

    input_sequence = [
        PlayerInput(forward=True),
        PlayerInput(forward=True),
        PlayerInput(turn_right=True),
        PlayerInput(forward=True),
        PlayerInput(forward=True),
    ]

    for inputs in input_sequence:
        _, state_compiled = step_frame_compiled(
            box_game_module, state_compiled, inputs, small_config,
        )
        state_ref = update_state(state_ref, inputs, box_segments, trig_table)

    assert state_compiled.x == pytest.approx(state_ref.x, abs=0.3)
    assert state_compiled.y == pytest.approx(state_ref.y, abs=0.3)
    assert state_compiled.angle == state_ref.angle


# ── Patch equivalence: sharded output matches unsharded ───────────────


def test_compiled_patch_equivalence(box_segments, small_config):
    """Compile once unsharded and once with rows_per_patch=6 (divides H=12).

    The dumb stitcher reassembles the patches using the col_idx /
    patch_row_start scalars emitted by the graph, and the final frame
    should match the unsharded render pixel-for-pixel (within a small
    numeric tolerance).
    """
    state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput()

    unsharded = compile_game(
        box_segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        d=1024, d_head=16, verbose=False,
    )
    sharded = compile_game(
        box_segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        d=1024, d_head=16, verbose=False,
        rows_per_patch=6,
    )

    assert unsharded.metadata.get("rows_per_patch") == small_config.screen_height
    assert sharded.metadata.get("rows_per_patch") == 6

    frame_u, _ = step_frame_compiled(unsharded, state, inputs, small_config)
    frame_s, _ = step_frame_compiled(sharded, state, inputs, small_config)

    assert frame_u.shape == frame_s.shape
    np.testing.assert_allclose(frame_s, frame_u, atol=0.01)


def test_metadata_plumbing_in_memory(box_segments, small_config):
    """compile_game stashes rows_per_patch in module metadata."""
    m = compile_game(
        box_segments, small_config,
        max_coord=10.0, d=1024, d_head=16, verbose=False,
        rows_per_patch=4,
    )
    assert m.metadata == {"rows_per_patch": 4}
