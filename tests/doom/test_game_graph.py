"""Regression tests for the walls-as-tokens renderer.

Verifies the full pipeline: compile_game → step_frame → frame
matches the reference renderer for the box room scene.
"""

import numpy as np
import pytest
import torch

from tests._utils.image_compare import compare_images
from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
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


def _box_room_segments(half=5.0):
    return [
        Segment(ax=half, ay=-half, bx=half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=0),
        Segment(ax=-half, ay=-half, bx=-half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=1),
        Segment(ax=-half, ay=half, bx=half, by=half,
                color=(0.8, 0.2, 0.1), texture_id=2),
        Segment(ax=-half, ay=-half, bx=half, by=-half,
                color=(0.8, 0.2, 0.1), texture_id=3),
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
        segs = _box_room_segments()
        trig = generate_trig_table()
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs, trig

    @pytest.fixture(scope="class")
    def module(self, box_room):
        config, textures, subset, segs, trig = box_room
        return compile_game(
            config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
        )

    def test_renders_box_room(self, module, box_room):
        """Render one frame of the box room, compare against reference."""
        config, textures, subset, segs, trig = box_room

        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput()

        frame, new_state = step_frame(module, state, inputs, subset, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, 0, segs, config, textures)

        assert frame.max() > 0.1, "frame appears blank"

        compare_images(frame, ref).assert_matches()

    def test_game_logic_angle_update(self, module, box_room):
        """Verify that the START token's game logic updates the player
        angle correctly when turn inputs are provided.
        """
        config, textures, subset, segs, trig = box_room

        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)

        # Turn right: angle should increase by turn_speed
        inputs = PlayerInput(turn_right=True)
        frame1, state1 = step_frame(module, state, inputs, subset, config, textures=textures)
        assert abs(state1.angle - 4) < 1, (
            f"angle after turn_right: {state1.angle}, expected ~4"
        )

        # Turn left from angle 0: should wrap to 252
        inputs = PlayerInput(turn_left=True)
        frame2, state2 = step_frame(module, state, inputs, subset, config, textures=textures)
        assert abs(state2.angle - 252) < 1, (
            f"angle after turn_left: {state2.angle}, expected ~252"
        )

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_renders_from_angle(self, module, box_room, angle):
        """Render the box room from a specific angle and verify it
        matches the reference frame.
        """
        config, textures, subset, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, subset, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        compare_images(frame, ref).assert_matches()

    @pytest.mark.parametrize("angle", [20, 45, 100, 160, 210])
    def test_renders_oblique_angle(self, module, box_room, angle):
        """Render from oblique angles where walls are not perpendicular
        to the view direction.  Wall heights should vary per column
        (closer side taller, farther side shorter).
        """
        config, textures, subset, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, subset, config, textures=textures)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.95, max_err=0.35,
        )

    @pytest.mark.parametrize("px,py,angle", [
        pytest.param(
            3.0, 2.0, 20,
            marks=pytest.mark.xfail(
                reason=(
                    "Phase E regression, root cause partially characterized. "
                    "At sort[0] the SORTED attend_argmin_above_integer softmax "
                    "concentrates on SORTED[0] itself (weight=1.0, logit=+800) "
                    "rather than any WALL position (logits +555..+637, expected "
                    "+1000).  Raw sel_bsp_rank reads -1171.875 instead of a "
                    "clean integer in [0, max_walls-1]; sort_done correctly "
                    "fires and the 99-sentinel replaces the bogus value, but "
                    "downstream THINKING/RENDER produce incorrect pixels.  The "
                    "magnitude of the error (score contamination on the order "
                    "of 100) is outside any documented per-op noise budget and "
                    "is most naturally explained by residual-column aliasing in "
                    "the compiled SORTED attention layer — the specific "
                    "aliasing pair has NOT yet been identified.  See "
                    "docs/postmortems/phase_e_xfail.md for the full evidence, "
                    "the calculation that rules out per-op noise, and what "
                    "would fix it.  Fixing requires compiler-internals work "
                    "beyond the plan-6 investigation scope; tracked there."
                ),
                strict=True,
            ),
        ),
        (-2.0, 3.0, 240),  # off-center, looking at wall at steep angle
        (1.0, -3.0, 50),   # off-center, oblique to two walls
    ])
    def test_renders_off_center_oblique(self, module, box_room, px, py, angle):
        """Render from off-center positions at oblique angles.

        This exercises the full precomputed pipeline with walls at
        varying distances and angles across the screen — the near
        side of a wall should be taller than the far side.
        """
        config, textures, subset, segs, trig = box_room

        inputs = PlayerInput()
        state = GameState(x=px, y=py, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(module, state, inputs, subset, config, textures=textures)
        ref = _ref_frame(px, py, angle, segs, config, textures)

        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.95, max_err=0.35,
        )

    # ── Collision detection tests ──────────────────────────────────

    def test_collision_blocks_wall(self, module, box_room):
        """Walking into a wall should be blocked (position doesn't advance
        past wall).
        """
        config, textures, subset, segs, trig = box_room

        # Start near the east wall (x=5), facing east (angle=0)
        state = GameState(x=4.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput(forward=True)

        _, new_state = step_frame(module, state, inputs, subset, config, textures=textures)
        ref_state = update_state(state, inputs, segs, trig)

        assert new_state.x == pytest.approx(ref_state.x, abs=0.15)
        assert new_state.x < 5.0, f"Passed through wall: x={new_state.x}"

    def test_collision_wall_sliding(self, module, box_room):
        """Moving diagonally into a wall should slide along it."""
        config, textures, subset, segs, trig = box_room

        # Near east wall, facing northeast (angle ~32)
        state = GameState(x=4.5, y=0.0, angle=32, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput(forward=True)

        _, new_state = step_frame(module, state, inputs, subset, config, textures=textures)
        ref_state = update_state(state, inputs, segs, trig)

        # X should be blocked (near wall), Y should advance
        assert new_state.x == pytest.approx(ref_state.x, abs=0.2)
        assert new_state.y == pytest.approx(ref_state.y, abs=0.2)
