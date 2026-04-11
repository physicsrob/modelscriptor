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


class TestPointSegmentDistance:
    def test_perpendicular_to_midpoint(self):
        seg = Segment(ax=0, ay=0, bx=10, by=0, color=(1, 0, 0))
        assert point_segment_distance(5, 3, seg) == pytest.approx(3.0)

    def test_on_segment(self):
        seg = Segment(ax=0, ay=0, bx=10, by=0, color=(1, 0, 0))
        assert point_segment_distance(5, 0, seg) == pytest.approx(0.0)

    def test_nearest_is_endpoint_a(self):
        seg = Segment(ax=0, ay=0, bx=10, by=0, color=(1, 0, 0))
        # Point is behind endpoint A
        assert point_segment_distance(-3, 4, seg) == pytest.approx(5.0)

    def test_nearest_is_endpoint_b(self):
        seg = Segment(ax=0, ay=0, bx=10, by=0, color=(1, 0, 0))
        # Point is past endpoint B
        assert point_segment_distance(13, 4, seg) == pytest.approx(5.0)

    def test_diagonal_segment(self):
        seg = Segment(ax=0, ay=0, bx=3, by=4, color=(1, 0, 0))
        # Point at origin is on endpoint A
        assert point_segment_distance(0, 0, seg) == pytest.approx(0.0)

    def test_degenerate_zero_length(self):
        seg = Segment(ax=3, ay=4, bx=3, by=4, color=(1, 0, 0))
        assert point_segment_distance(0, 0, seg) == pytest.approx(5.0)

    def test_perpendicular_to_vertical_segment(self):
        seg = Segment(ax=5, ay=0, bx=5, by=10, color=(1, 0, 0))
        assert point_segment_distance(2, 5, seg) == pytest.approx(3.0)


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
    def test_no_input_no_change(self, trig_table):
        state = GameState(x=0, y=0, angle=0)
        new = update_state(state, PlayerInput(), [], trig_table)
        assert new.x == 0.0
        assert new.y == 0.0
        assert new.angle == 0

    def test_forward_angle_0(self, trig_table):
        """Angle 0 = +x direction (cos=1, sin=0)."""
        state = GameState(x=0, y=0, angle=0, move_speed=1.0)
        new = update_state(state, PlayerInput(forward=True), [], trig_table)
        assert new.x == pytest.approx(1.0, abs=1e-6)
        assert new.y == pytest.approx(0.0, abs=1e-6)

    def test_forward_angle_64(self, trig_table):
        """Angle 64 = +y direction (cos=0, sin=1)."""
        state = GameState(x=0, y=0, angle=64, move_speed=1.0)
        new = update_state(state, PlayerInput(forward=True), [], trig_table)
        assert new.x == pytest.approx(0.0, abs=1e-6)
        assert new.y == pytest.approx(1.0, abs=1e-6)

    def test_backward_is_opposite(self, trig_table):
        state = GameState(x=0, y=0, angle=0, move_speed=1.0)
        new = update_state(state, PlayerInput(backward=True), [], trig_table)
        assert new.x == pytest.approx(-1.0, abs=1e-6)
        assert new.y == pytest.approx(0.0, abs=1e-6)

    def test_strafe_left_perpendicular(self, trig_table):
        """Strafe left at angle 0 should move in +y direction."""
        state = GameState(x=0, y=0, angle=0, move_speed=1.0)
        new = update_state(state, PlayerInput(strafe_left=True), [], trig_table)
        # At angle 0: cos=1, sin=0. Strafe left: dx += sin=0, dy -= cos=-1
        assert new.x == pytest.approx(0.0, abs=1e-6)
        assert new.y == pytest.approx(-1.0, abs=1e-6)

    def test_strafe_right_perpendicular(self, trig_table):
        """Strafe right at angle 0 should move in -y direction."""
        state = GameState(x=0, y=0, angle=0, move_speed=1.0)
        new = update_state(state, PlayerInput(strafe_right=True), [], trig_table)
        assert new.x == pytest.approx(0.0, abs=1e-6)
        assert new.y == pytest.approx(1.0, abs=1e-6)

    def test_turn_left(self, trig_table):
        state = GameState(x=0, y=0, angle=10, turn_speed=4)
        new = update_state(state, PlayerInput(turn_left=True), [], trig_table)
        assert new.angle == 6

    def test_turn_right(self, trig_table):
        state = GameState(x=0, y=0, angle=10, turn_speed=4)
        new = update_state(state, PlayerInput(turn_right=True), [], trig_table)
        assert new.angle == 14

    def test_angle_wraps_below_zero(self, trig_table):
        state = GameState(x=0, y=0, angle=2, turn_speed=4)
        new = update_state(state, PlayerInput(turn_left=True), [], trig_table)
        assert new.angle == 254

    def test_angle_wraps_above_255(self, trig_table):
        state = GameState(x=0, y=0, angle=254, turn_speed=4)
        new = update_state(state, PlayerInput(turn_right=True), [], trig_table)
        assert new.angle == 2

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
                state, PlayerInput(forward=True), box_segments, trig_table,
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
                    state, PlayerInput(forward=True), box_segments, trig_table,
                )
            assert -5.0 < state.x < 5.0, f"Escaped at angle {start_angle}"
            assert -5.0 < state.y < 5.0, f"Escaped at angle {start_angle}"

    def test_walk_through_doorway(self, trig_table, multi_segments):
        """Can walk from room A through corridor to room B."""
        # Start in room A, face east (angle=0), walk forward
        state = GameState(x=-8, y=0, angle=0, move_speed=0.3)
        for _ in range(200):
            state = update_state(
                state, PlayerInput(forward=True), multi_segments, trig_table,
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
                state, PlayerInput(forward=True), [], trig_table,
            )
        expected_x = 0.5 * n_frames * float(trig_table[0, 0])
        assert state.x == pytest.approx(expected_x, abs=1e-6)
