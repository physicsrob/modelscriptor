"""Tests for the fixture loader."""

from __future__ import annotations

import pytest

from doom_sandbox.fixtures import available_fixtures, load_fixture
from doom_sandbox.phases.phase1_bsp_ranks.reference import expected_bsp_ranks
from doom_sandbox.types import GameState, MapSubset


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
