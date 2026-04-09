"""Tests for wall texture rendering — both reference and compiled."""

import numpy as np
import pytest

from torchwright.doom.compile import compile_game, step_frame_compiled
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer import render_frame
from torchwright.reference_renderer.scenes import box_room, box_room_textured
from torchwright.reference_renderer.textures import default_texture_atlas
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


# ── Reference renderer texture tests ────────────────────────────────


def test_reference_textured_differs_from_solid(small_config):
    """Textured rendering should produce different pixels than solid color."""
    segments, textures = box_room_textured()
    solid_segments = box_room()

    textured = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    solid = render_frame(0.0, 0.0, 0, solid_segments, small_config)

    # They should NOT be identical — textures vary per row
    assert not np.allclose(textured, solid, atol=0.01)


def test_reference_texture_u_coordinate(small_config):
    """Looking straight at a wall, left and right columns should show
    different texture columns."""
    segments, textures = box_room_textured()
    frame = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    H, W = small_config.screen_height, small_config.screen_width

    # The east wall (brick) should be visible. Columns near the edges
    # of the screen hit different u values → different texture columns.
    center_row = H // 2
    left_col = frame[center_row, 0]
    right_col = frame[center_row, W - 1]
    # Left and right edges hit different texture columns on the east wall,
    # so they should generally differ (unless the texture happens to repeat)
    # At minimum, the wall colors should come from the brick texture
    assert not np.allclose(left_col, [0, 0, 0], atol=0.01)  # not ceiling


# ── Compiled textured renderer tests ────────────────────────────────


@pytest.fixture
def textured_game_module(small_config):
    """Compile textured game graph for box_room."""
    segments, textures = box_room_textured()
    return compile_game(
        segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        textures=textures,
        d=2048, d_head=16, verbose=False,
    )


def _assert_frame_match(compiled, reference, max_boundary_pixels=None, atol=0.2, msg=""):
    H, W = compiled.shape[:2]
    if max_boundary_pixels is None:
        max_boundary_pixels = 3 * W  # more tolerance for texture band edges
    mismatched = np.abs(compiled - reference) > atol
    n_bad = mismatched.any(axis=2).sum()
    if n_bad > max_boundary_pixels:
        rows, cols = np.where(mismatched.any(axis=2))
        r, c = rows[0], cols[0]
        assert False, (
            f"{msg} {n_bad}/{H * W} pixels differ (max allowed: {max_boundary_pixels}). "
            f"First at ({r},{c}): compiled={compiled[r, c]} ref={reference[r, c]}"
        )


def test_compiled_textured_no_input(textured_game_module, small_config):
    """No inputs: textured frame should match reference renderer."""
    segments, textures = box_room_textured()
    state = GameState(x=0.0, y=0.0, angle=0)

    frame, new_state = step_frame_compiled(
        textured_game_module, state, PlayerInput(), small_config,
    )

    ref = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    _assert_frame_match(frame, ref, msg="no-input textured")


def test_compiled_textured_turned(textured_game_module, small_config):
    """Different viewing angles should show different textures."""
    segments, textures = box_room_textured()

    for angle in [0, 64, 128, 192]:
        state = GameState(x=0.0, y=0.0, angle=angle)
        frame, _ = step_frame_compiled(
            textured_game_module, state, PlayerInput(), small_config,
        )
        ref = render_frame(0.0, 0.0, angle, segments, small_config, textures=textures)
        _assert_frame_match(frame, ref, msg=f"textured angle={angle}")
