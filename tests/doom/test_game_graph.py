"""Regression tests for the walls-as-tokens renderer.

Verifies the full pipeline: compile_game → step_frame → frame
matches the reference renderer for the box room scene.
"""

import numpy as np
import pytest
import torch

from torchwright.doom.compile import compile_game, segments_to_walls, step_frame
from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.render import render_column
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


def _box_room_config():
    trig = generate_trig_table()
    config = RenderConfig(
        screen_width=16, screen_height=20, fov_columns=16,
        trig_table=trig,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    return config


def _box_room_walls(half=5.0):
    return [
        {"ax": half, "ay": -half, "bx": half, "by": half, "tex_id": 0.0},
        {"ax": -half, "ay": -half, "bx": -half, "by": half, "tex_id": 1.0},
        {"ax": -half, "ay": half, "bx": half, "by": half, "tex_id": 2.0},
        {"ax": -half, "ay": -half, "bx": half, "by": -half, "tex_id": 3.0},
    ]


def _box_room_segments(half=5.0):
    walls = _box_room_walls(half)
    return [
        Segment(ax=w["ax"], ay=w["ay"], bx=w["bx"], by=w["by"],
                color=(0.8, 0.2, 0.1), texture_id=int(w["tex_id"]))
        for w in walls
    ]


def _ref_frame(px, py, angle, segs, config, textures):
    """Render the full frame via the reference renderer."""
    H, W = config.screen_height, config.screen_width
    frame = np.zeros((H, W, 3), dtype=np.float64)
    for col in range(W):
        frame[:, col, :] = render_column(
            col, px, py, int(angle), segs, config, textures=textures,
        )
    return frame


class TestGameGraph:
    """Full pipeline tests sharing a single compile_game() call."""

    @pytest.fixture(scope="class")
    def box_room(self):
        config = _box_room_config()
        textures = default_texture_atlas()
        walls = _box_room_walls()
        segs = _box_room_segments()
        trig = generate_trig_table()
        return config, textures, walls, segs, trig

    @pytest.fixture(scope="class")
    def module(self, box_room):
        config, textures, walls, segs, trig = box_room
        return compile_game(
            config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
        )

    def test_renders_box_room(self, module, box_room):
        """Render one frame of the box room, compare against reference."""
        config, textures, walls, segs, trig = box_room

        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput()

        frame, new_state = step_frame(module, state, inputs, walls, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, 0, segs, config, textures)

        assert frame.max() > 0.1, "frame appears blank"

        max_err = np.abs(frame - ref).max()
        mean_err = np.abs(frame - ref).mean()
        print(f"\nbox room: max_err={max_err:.3f}, mean_err={mean_err:.3f}")

        assert max_err < 0.65, (
            f"max pixel error {max_err:.3f} exceeds 0.5 (mean {mean_err:.3f})"
        )

    def test_game_logic_angle_update(self, module, box_room):
        """Verify that the START token's game logic updates the player
        angle correctly when turn inputs are provided.
        """
        config, textures, walls, segs, trig = box_room

        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)

        # Turn right: angle should increase by turn_speed
        inputs = PlayerInput(turn_right=True)
        frame1, state1 = step_frame(module, state, inputs, walls, config, textures=textures)
        assert abs(state1.angle - 4) < 1, (
            f"angle after turn_right: {state1.angle}, expected ~4"
        )

        # Turn left from angle 0: should wrap to 252
        inputs = PlayerInput(turn_left=True)
        frame2, state2 = step_frame(module, state, inputs, walls, config, textures=textures)
        assert abs(state2.angle - 252) < 1, (
            f"angle after turn_left: {state2.angle}, expected ~252"
        )

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_renders_from_angle(self, module, box_room, angle):
        """Render the box room from a specific angle and verify it
        matches the reference frame.
        """
        config, textures, walls, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, walls, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        max_err = np.abs(frame - ref).max()
        assert max_err < 0.65, (
            f"angle={angle}: max pixel error {max_err:.3f} exceeds 0.5"
        )

    @pytest.mark.parametrize("angle", [20, 45, 100, 160, 210])
    def test_renders_oblique_angle(self, module, box_room, angle):
        """Render from oblique angles where walls are not perpendicular
        to the view direction.  Wall heights should vary per column
        (closer side taller, farther side shorter).
        """
        config, textures, walls, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, walls, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        max_err = np.abs(frame - ref).max()
        mean_err = np.abs(frame - ref).mean()
        print(f"\n  oblique angle={angle}: max_err={max_err:.3f}, mean_err={mean_err:.3f}")
        assert max_err < 0.65, (
            f"angle={angle}: max pixel error {max_err:.3f} exceeds 0.5"
        )

    @pytest.mark.parametrize("px,py,angle", [
        (3.0, 2.0, 20),    # near corner, looking diagonally
        (-2.0, 3.0, 240),  # off-center, looking at wall at steep angle
        (1.0, -3.0, 50),   # off-center, oblique to two walls
    ])
    def test_renders_off_center_oblique(self, module, box_room, px, py, angle):
        """Render from off-center positions at oblique angles.

        This exercises the full precomputed pipeline with walls at
        varying distances and angles across the screen — the near
        side of a wall should be taller than the far side.
        """
        config, textures, walls, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=px, y=py, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, walls, config, textures=textures)
        ref = _ref_frame(px, py, angle, segs, config, textures)

        max_err = np.abs(frame - ref).max()
        mean_err = np.abs(frame - ref).mean()
        print(f"\n  off-center ({px},{py}) angle={angle}: "
              f"max_err={max_err:.3f}, mean_err={mean_err:.3f}")
        assert max_err < 0.65, (
            f"({px},{py}) angle={angle}: max pixel error {max_err:.3f} exceeds 0.5"
        )

    # ── Collision detection tests ──────────────────────────────────

    def test_collision_blocks_wall(self, module, box_room):
        """Walking into a wall should be blocked (position doesn't advance
        past wall).
        """
        config, textures, walls, segs, trig = box_room

        # Start near the east wall (x=5), facing east (angle=0)
        state = GameState(x=4.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput(forward=True)

        _, new_state = step_frame(module, state, inputs, walls, config, textures=textures)
        ref_state = update_state(state, inputs, segs, trig)

        assert new_state.x == pytest.approx(ref_state.x, abs=0.15)
        assert new_state.x < 5.0, f"Passed through wall: x={new_state.x}"

    def test_collision_wall_sliding(self, module, box_room):
        """Moving diagonally into a wall should slide along it."""
        config, textures, walls, segs, trig = box_room

        # Near east wall, facing northeast (angle ~32)
        state = GameState(x=4.5, y=0.0, angle=32, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput(forward=True)

        _, new_state = step_frame(module, state, inputs, walls, config, textures=textures)
        ref_state = update_state(state, inputs, segs, trig)

        # X should be blocked (near wall), Y should advance
        assert new_state.x == pytest.approx(ref_state.x, abs=0.2)
        assert new_state.y == pytest.approx(ref_state.y, abs=0.2)


def test_segments_to_walls():
    """segments_to_walls converts Segment objects to wall dicts."""
    segs = _box_room_segments()
    walls = segments_to_walls(segs)
    assert len(walls) == len(segs)
    for w, s in zip(walls, segs):
        assert w["ax"] == s.ax
        assert w["ay"] == s.ay
        assert w["bx"] == s.bx
        assert w["by"] == s.by
        assert w["tex_id"] == float(s.texture_id)
