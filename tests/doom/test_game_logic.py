"""Tests for Phase 3: player movement, collision detection, and state carry."""

import math

import pytest

from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.collision import (
    point_segment_distance,
    resolve_collision,
)
from torchwright.reference_renderer.scenes import box_room, multi_room
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import Segment


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def box_segments():
    return box_room()


@pytest.fixture
def multi_segments():
    return multi_room()


# ── point_segment_distance ───────────────────────────────────────────


@pytest.mark.parametrize(
    "ax,ay,bx,by,px,py,expected",
    [
        (0, 0, 10, 0, 5, 3, 3.0),  # perpendicular to midpoint
        (0, 0, 10, 0, 5, 0, 0.0),  # on segment
        (0, 0, 10, 0, -3, 4, 5.0),  # nearest is endpoint A
        (0, 0, 10, 0, 13, 4, 5.0),  # nearest is endpoint B
        (0, 0, 3, 4, 0, 0, 0.0),  # diagonal segment, on endpoint A
        (3, 4, 3, 4, 0, 0, 5.0),  # degenerate zero-length
        (5, 0, 5, 10, 2, 5, 3.0),  # perpendicular to vertical segment
    ],
    ids=[
        "perpendicular_to_midpoint",
        "on_segment",
        "nearest_is_endpoint_a",
        "nearest_is_endpoint_b",
        "diagonal_segment",
        "degenerate_zero_length",
        "perpendicular_to_vertical_segment",
    ],
)
def test_point_segment_distance(ax, ay, bx, by, px, py, expected):
    seg = Segment(ax=ax, ay=ay, bx=bx, by=by, color=(1, 0, 0))
    assert point_segment_distance(px, py, seg) == pytest.approx(expected)


# ── resolve_collision ────────────────────────────────────────────────


class TestResolveCollision:
    def test_no_collision_passes_through(self):
        segments = [Segment(ax=5, ay=-10, bx=5, by=10, color=(1, 0, 0))]
        rx, ry = resolve_collision(0, 0, 1, 0, segments, margin=0.2)
        assert (rx, ry) == (1.0, 0.0)

    def test_blocked_by_wall(self):
        segments = [Segment(ax=2, ay=-10, bx=2, by=10, color=(1, 0, 0))]
        rx, ry = resolve_collision(0, 0, 1.9, 0, segments, margin=0.2)
        # X move puts us within margin of wall at x=2, Y is unchanged
        assert rx == 0.0  # X blocked
        assert ry == 0.0  # Y was 0, stays 0

    def test_wall_sliding(self):
        # Wall along x=2. Moving diagonally into it should slide along Y.
        segments = [Segment(ax=2, ay=-10, bx=2, by=10, color=(1, 0, 0))]
        rx, ry = resolve_collision(0, 0, 1.9, 1.0, segments, margin=0.2)
        assert rx == 0.0  # X blocked
        assert ry == 1.0  # Y slides freely

    def test_corner_blocked_both_axes(self):
        # Walls on both axes
        segments = [
            Segment(ax=1, ay=-10, bx=1, by=10, color=(1, 0, 0)),
            Segment(ax=-10, ay=1, bx=10, by=1, color=(0, 1, 0)),
        ]
        rx, ry = resolve_collision(0, 0, 0.9, 0.9, segments, margin=0.2)
        assert rx == 0.0
        assert ry == 0.0


# ── Movement (no collision) ─────────────────────────────────────────


class TestMovement:
    @pytest.mark.parametrize(
        "angle,input_kwargs,expected_dx,expected_dy",
        [
            (0, {}, 0.0, 0.0),  # no input
            (0, {"forward": True}, 1.0, 0.0),  # angle 0: +x
            (64, {"forward": True}, 0.0, 1.0),  # angle 64: +y
            (0, {"backward": True}, -1.0, 0.0),  # backward is opposite of forward
            (0, {"strafe_left": True}, 0.0, -1.0),  # strafe left at angle 0: -y
            (0, {"strafe_right": True}, 0.0, 1.0),  # strafe right at angle 0: +y
        ],
        ids=[
            "no_input_no_change",
            "forward_angle_0",
            "forward_angle_64",
            "backward_is_opposite",
            "strafe_left_perpendicular",
            "strafe_right_perpendicular",
        ],
    )
    def test_position_change(
        self, trig_table, angle, input_kwargs, expected_dx, expected_dy
    ):
        state = GameState(x=0, y=0, angle=angle, move_speed=1.0)
        new = update_state(state, PlayerInput(**input_kwargs), [], trig_table)
        assert new.x == pytest.approx(expected_dx, abs=1e-6)
        assert new.y == pytest.approx(expected_dy, abs=1e-6)

    @pytest.mark.parametrize(
        "start_angle,input_kwargs,turn_speed,expected_angle",
        [
            (10, {"turn_left": True}, 4, 6),  # turn left decreases by 4
            (10, {"turn_right": True}, 4, 14),  # turn right increases by 4
            (2, {"turn_left": True}, 4, 254),  # wraps below zero
            (254, {"turn_right": True}, 4, 2),  # wraps above 255
        ],
        ids=["turn_left", "turn_right", "wraps_below_zero", "wraps_above_255"],
    )
    def test_turn(
        self, trig_table, start_angle, input_kwargs, turn_speed, expected_angle
    ):
        state = GameState(x=0, y=0, angle=start_angle, turn_speed=turn_speed)
        new = update_state(state, PlayerInput(**input_kwargs), [], trig_table)
        assert new.angle == expected_angle

    def test_full_rotation_returns_to_start(self, trig_table):
        state = GameState(x=0, y=0, angle=0, turn_speed=1)
        for _ in range(256):
            state = update_state(state, PlayerInput(turn_right=True), [], trig_table)
        assert state.angle == 0


# ── Movement with collision ──────────────────────────────────────────


class TestMovementWithCollision:
    def test_walk_into_wall_stops(self, trig_table, box_segments):
        """Walking toward a wall in box_room stops at the margin."""
        state = GameState(x=0, y=0, angle=0, move_speed=0.5)
        for _ in range(100):
            state = update_state(
                state,
                PlayerInput(forward=True),
                box_segments,
                trig_table,
            )
        # Should be near the east wall (x=5) minus margin
        assert state.x < 5.0
        assert state.x > 4.0

    def test_cannot_escape_box_room(self, trig_table, box_segments):
        """Player cannot walk through any wall of box_room."""
        for start_angle in range(0, 256, 8):
            state = GameState(x=0, y=0, angle=start_angle, move_speed=0.5)
            for _ in range(100):
                state = update_state(
                    state,
                    PlayerInput(forward=True),
                    box_segments,
                    trig_table,
                )
            assert -5.0 < state.x < 5.0, f"Escaped at angle {start_angle}"
            assert -5.0 < state.y < 5.0, f"Escaped at angle {start_angle}"

    def test_walk_through_doorway(self, trig_table, multi_segments):
        """Can walk from room A through corridor to room B."""
        # Start in room A, face east (angle=0), walk forward
        state = GameState(x=-8, y=0, angle=0, move_speed=0.3)
        for _ in range(200):
            state = update_state(
                state,
                PlayerInput(forward=True),
                multi_segments,
                trig_table,
            )
        # Should have passed through corridor into room B
        assert state.x > 4.0, f"Didn't reach room B: x={state.x}"


# ── State carry ──────────────────────────────────────────────────────


class TestStateCarry:
    def test_multi_frame_position_accumulates(self, trig_table):
        """Position accumulates correctly across multiple frames."""
        state = GameState(x=0, y=0, angle=0, move_speed=0.5)
        n_frames = 10
        for _ in range(n_frames):
            state = update_state(
                state,
                PlayerInput(forward=True),
                [],
                trig_table,
            )
        expected_x = 0.5 * n_frames * float(trig_table[0, 0])
        assert state.x == pytest.approx(expected_x, abs=1e-6)
