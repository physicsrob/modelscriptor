"""Tests for TEX_COL token type — texture data via attention.

Verifies that RENDER tokens retrieve texture column pixel data from
TEX_COL tokens via attention, matching the reference renderer.
"""

import numpy as np
import pytest

from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.reference_renderer.render import render_column
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

from tests._utils.image_compare import compare_images


def _box_room_config():
    trig = generate_trig_table()
    return RenderConfig(
        screen_width=16, screen_height=20, fov_columns=16,
        trig_table=trig,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


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
    H, W = config.screen_height, config.screen_width
    frame = np.zeros((H, W, 3), dtype=np.float64)
    for col in range(W):
        frame[:, col, :] = render_column(
            col, px, py, int(angle), segs, config, textures=textures,
        )
    return frame


class TestTexCol:
    """TEX_COL attention tests sharing a single compile_game() call."""

    @pytest.fixture(scope="class")
    def box_room(self):
        config = _box_room_config()
        textures = default_texture_atlas()
        segs = _box_room_segments()
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs

    @pytest.fixture(scope="class")
    def module(self, box_room):
        config, textures, subset, segs = box_room
        return compile_game(
            config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
        )

    def test_renders_box_room(self, module, box_room):
        """Render one frame and compare against reference renderer."""
        config, textures, subset, segs = box_room

        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput()

        frame, _ = step_frame(module, state, inputs, subset, config,
                              textures=textures)
        ref = _ref_frame(0.0, 0.0, 0, segs, config, textures)

        assert frame.max() > 0.1, "frame appears blank"

        compare_images(frame, ref).assert_matches()

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_renders_from_angle(self, module, box_room, angle):
        """Render from multiple angles, verify matches reference."""
        config, textures, subset, segs = box_room

        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        inputs = PlayerInput()

        frame, _ = step_frame(module, state, inputs, subset, config,
                              textures=textures)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        compare_images(frame, ref).assert_matches()
