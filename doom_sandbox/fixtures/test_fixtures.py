"""Tests for the fixture loader."""

from __future__ import annotations

import pytest

from . import (
    assert_pose_clear_of_planes,
    available_fixtures,
    load_fixture,
)
from ..phases.phase1_bsp_ranks.reference import expected_bsp_ranks
from ..types import BSPNode, GameState, MapSubset


# --- Loader basics ---

def test_box_room_is_available():
    assert "box_room" in available_fixtures()


def test_load_unknown_raises_with_available_list():
    with pytest.raises(FileNotFoundError) as exc:
        load_fixture("does_not_exist")
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "box_room" in msg  # available list mentioned


def test_box_room_loads_as_map_subset():
    scene = load_fixture("box_room")
    assert isinstance(scene, MapSubset)


# --- box_room shape sanity ---

def test_box_room_has_4_segments():
    scene = load_fixture("box_room")
    assert len(scene.segments) == 4


def test_box_room_has_3_bsp_nodes():
    """Balanced BSP over 4 leaves has 3 internal nodes."""
    scene = load_fixture("box_room")
    assert len(scene.bsp_nodes) == 3


def test_box_room_bsp_coefficient_shape_matches():
    scene = load_fixture("box_room")
    n = len(scene.segments)
    m = len(scene.bsp_nodes)
    assert len(scene.seg_bsp_coeffs) == n
    assert all(len(row) == m for row in scene.seg_bsp_coeffs)
    assert len(scene.seg_bsp_consts) == n


def test_box_room_obeys_phase1_bounds():
    """Phase 1 commits to N <= 8 walls and M <= 16 BSP nodes."""
    scene = load_fixture("box_room")
    assert len(scene.segments) <= 8
    assert len(scene.bsp_nodes) <= 16


# --- Reference computes valid ranks on the loaded fixture ---

def test_box_room_ranks_form_a_permutation():
    """For any player position inside the room, the BSP-rank vector
    must be a permutation of [0..N-1] — every wall gets a distinct
    rank."""
    scene = load_fixture("box_room")
    for x, y in [(0.0, 0.0), (3.0, 1.0), (-2.5, -4.0), (4.9, 4.9)]:
        state = GameState(x=x, y=y, angle=0)
        ranks = expected_bsp_ranks(scene, state)
        assert sorted(ranks) == list(range(len(scene.segments))), (
            f"ranks at ({x}, {y}) = {ranks} not a permutation"
        )


# --- Test poses ---

def test_box_room_has_test_poses():
    scene = load_fixture("box_room")
    assert len(scene.test_poses) >= 1


def test_box_room_test_poses_are_clear_of_planes():
    """Every committed test_pose must land at least 0.1 from every
    BSP plane — otherwise the assertion helper would have caught it."""
    scene = load_fixture("box_room")
    for state in scene.test_poses:
        assert_pose_clear_of_planes(scene, state)


def test_box_room_test_poses_cover_distinct_side_p_patterns():
    """Test poses should exercise different `side_P_vec` patterns;
    otherwise the test only ever sees one branch of the rank formula."""
    scene = load_fixture("box_room")
    seen = set()
    for state in scene.test_poses:
        side_p = tuple(
            1 if (n.nx * state.x + n.ny * state.y + n.d) > 0 else 0
            for n in scene.bsp_nodes
        )
        seen.add(side_p)
    assert len(seen) >= 2, (
        f"test_poses cover only {len(seen)} side_P pattern(s); add more "
        f"poses to exercise different branches"
    )


def test_assert_pose_clear_of_planes_passes_well_clear():
    scene = load_fixture("box_room")
    assert_pose_clear_of_planes(scene, GameState(x=2.0, y=0.0, angle=0))


def test_assert_pose_clear_of_planes_raises_on_plane():
    scene = load_fixture("box_room")
    # Player exactly on the x=0 plane.
    with pytest.raises(AssertionError) as exc:
        assert_pose_clear_of_planes(
            scene, GameState(x=0.0, y=0.0, angle=0)
        )
    msg = str(exc.value)
    assert "plane 0" in msg
    assert "0.0" in msg or "0.000000" in msg


def test_assert_pose_clear_of_planes_respects_atol():
    """A pose with clearance 0.5 passes at default atol=0.1 but fails at atol=1.0."""
    scene = MapSubset(
        segments=[],
        bsp_nodes=[BSPNode(nx=1.0, ny=0.0, d=0.0)],
        seg_bsp_coeffs=[],
        seg_bsp_consts=[],
    )
    state = GameState(x=0.5, y=0.0, angle=0)
    assert_pose_clear_of_planes(scene, state, atol=0.1)
    with pytest.raises(AssertionError):
        assert_pose_clear_of_planes(scene, state, atol=1.0)


def test_box_room_ranks_match_known_player_position():
    """Spot check the exact rank vector at a known player position. A
    sign error or coefficient permutation in the fixture-generation
    pipeline would silently pass a permutation-only check, so we pin
    the concrete rank vector here.

    At player (0.5, 0.5) inside the box_room (segments at x=±5, y=±5):
    side_P_vec = [1, 1, 0] (FRONT of x=0 plane, FRONT of y=-2.5
    plane, BACK of y=+2.5 plane). With the committed
    seg_bsp_coeffs / seg_bsp_consts, the resulting rank vector is
    [0, 2, 3, 1]."""
    scene = load_fixture("box_room")
    state = GameState(x=0.5, y=0.5, angle=0)
    ranks = expected_bsp_ranks(scene, state)
    assert ranks == [0, 2, 3, 1]
    # Determinism — re-running yields the same result.
    assert expected_bsp_ranks(scene, state) == ranks
