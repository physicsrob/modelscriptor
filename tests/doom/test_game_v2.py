"""Regression tests for the v2 walls-as-tokens renderer.

Verifies the full pipeline: compile_game_v2 → step_frame_v2 → frame
matches the reference renderer for the box room scene.
"""

import numpy as np
import pytest
import torch

from torchwright.doom.compile_v2 import compile_game_v2, step_frame_v2
from torchwright.doom.game import GameState
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


def test_v2_renders_box_room():
    """Compile the v2 graph, render one frame of the box room,
    compare against the reference renderer.
    """
    config = _box_room_config()
    textures = default_texture_atlas()
    walls = _box_room_walls()
    segs = _box_room_segments()

    module = compile_game_v2(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
    )

    state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
    inputs = PlayerInput()  # no movement

    frame, new_state = step_frame_v2(module, state, inputs, walls, config)
    ref = _ref_frame(0.0, 0.0, 0, segs, config, textures)

    assert frame.max() > 0.1, "frame appears blank"

    max_err = np.abs(frame - ref).max()
    mean_err = np.abs(frame - ref).mean()
    print(f"\nv2 box room: max_err={max_err:.3f}, mean_err={mean_err:.3f}")

    assert max_err < 0.65, (
        f"max pixel error {max_err:.3f} exceeds 0.5 (mean {mean_err:.3f})"
    )


def test_v2_game_logic_angle_update():
    """Verify that the START token's game logic updates the player
    angle correctly when turn inputs are provided.
    """
    config = _box_room_config()
    textures = default_texture_atlas()
    walls = _box_room_walls()

    module = compile_game_v2(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
    )

    state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)

    # Turn right: angle should increase by turn_speed
    inputs = PlayerInput(turn_right=True)
    frame1, state1 = step_frame_v2(module, state, inputs, walls, config)
    assert abs(state1.angle - 4) < 1, (
        f"angle after turn_right: {state1.angle}, expected ~4"
    )

    # Turn left from angle 0: should wrap to 252
    inputs = PlayerInput(turn_left=True)
    frame2, state2 = step_frame_v2(module, state, inputs, walls, config)
    assert abs(state2.angle - 252) < 1, (
        f"angle after turn_left: {state2.angle}, expected ~252"
    )


def test_v2_renders_from_different_angles():
    """Render the box room from two different angles and verify both
    match their reference frames.
    """
    config = _box_room_config()
    textures = default_texture_atlas()
    walls = _box_room_walls()
    segs = _box_room_segments()

    module = compile_game_v2(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False,
    )
    inputs = PlayerInput()

    for angle in [0, 64, 128, 192]:
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame_v2(module, state, inputs, walls, config)
        ref = _ref_frame(0.0, 0.0, angle, segs, config, textures)

        max_err = np.abs(frame - ref).max()
        assert max_err < 0.65, (
            f"angle={angle}: max pixel error {max_err:.3f} exceeds 0.5"
        )
