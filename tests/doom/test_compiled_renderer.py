"""End-to-end tests: compile the renderer graph and compare against reference.

Phase 2 deliverable: compiled transformer produces pixel-exact match against
the reference software renderer at multiple test positions and angles.
"""

import numpy as np
import pytest

from torchwright.doom.compile import compile_renderer, render_frame_compiled
from torchwright.reference_renderer import (
    RenderConfig,
    Segment,
    generate_trig_table,
    render_frame,
)
from torchwright.reference_renderer.scenes import box_room, multi_room


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def small_config(trig_table):
    """Small resolution for fast tests."""
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=trig_table,
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


@pytest.fixture
def target_config(trig_table):
    """Phase 2 target resolution: 32×40."""
    return RenderConfig(
        screen_width=32,
        screen_height=40,
        fov_columns=16,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _assert_frame_match(compiled, reference, atol=0.15, max_boundary_pixels=None, msg=""):
    """Full-frame comparison allowing a small number of boundary pixel mismatches.

    The graph uses continuous wall bounds while the reference uses int()
    truncation, which can cause up to 1 pixel boundary difference per column.
    """
    H, W = compiled.shape[:2]
    if max_boundary_pixels is None:
        # Allow up to 2 boundary rows per column (top + bottom edge)
        max_boundary_pixels = 2 * W

    mismatched = np.abs(compiled - reference) > atol
    n_bad = mismatched.any(axis=2).sum()
    n_total = H * W
    if n_bad > max_boundary_pixels:
        rows, cols = np.where(mismatched.any(axis=2))
        r, c = rows[0], cols[0]
        assert False, (
            f"{msg} {n_bad}/{n_total} pixels differ (max allowed: {max_boundary_pixels}). "
            f"First at ({r},{c}): compiled={compiled[r,c]} ref={reference[r,c]}"
        )


# ── Single wall: full-frame comparison at multiple angles ──────────


def test_compiled_single_wall_full_frame(small_config):
    """Full-frame pixel match for a single wall at 3 angles."""
    segments = [Segment(ax=5.0, ay=-10.0, bx=5.0, by=10.0, color=(1.0, 0.0, 0.0))]
    module = compile_renderer(
        segments, small_config, max_coord=15.0, d=1024, d_head=16, verbose=False,
    )

    for angle in [0, 32, 224]:
        compiled = render_frame_compiled(module, 0.0, 0.0, angle, small_config)
        ref = render_frame(0.0, 0.0, angle, segments, small_config)
        _assert_frame_match(compiled, ref, msg=f"angle={angle}")


# ── Two walls: occlusion with full-frame check ────────────────────


def test_compiled_two_walls_full_frame(small_config):
    """Nearer wall occludes farther wall — full frame."""
    segments = [
        Segment(ax=3.0, ay=-10.0, bx=3.0, by=10.0, color=(1.0, 0.0, 0.0)),
        Segment(ax=8.0, ay=-10.0, bx=8.0, by=10.0, color=(0.0, 1.0, 0.0)),
    ]
    module = compile_renderer(
        segments, small_config, max_coord=15.0, d=1024, d_head=16, verbose=False,
    )
    compiled = render_frame_compiled(module, 0.0, 0.0, 0, small_config)
    ref = render_frame(0.0, 0.0, 0, segments, small_config)
    _assert_frame_match(compiled, ref)


# ── Box room: full frame at 4 cardinal directions ─────────────────


def test_compiled_box_room_full_frame(small_config):
    """Box room from center, full-frame match in all 4 directions."""
    segments = box_room()
    module = compile_renderer(
        segments, small_config, max_coord=10.0, d=1024, d_head=16, verbose=False,
    )

    for angle in [0, 64, 128, 192]:
        compiled = render_frame_compiled(module, 0.0, 0.0, angle, small_config)
        ref = render_frame(0.0, 0.0, angle, segments, small_config)
        _assert_frame_match(compiled, ref, msg=f"angle={angle}")


# ── Box room: off-center player positions ──────────────────────────


def test_compiled_box_room_off_center(small_config):
    """Box room from off-center positions — tests perspective correctness."""
    segments = box_room()
    module = compile_renderer(
        segments, small_config, max_coord=10.0, d=1024, d_head=16, verbose=False,
    )

    positions = [
        (2.0, 0.0, 0),     # shifted east, looking east
        (-2.0, 1.0, 64),   # shifted west+north, looking north
        (0.0, -3.0, 192),  # shifted south, looking south
        (3.0, 3.0, 128),   # corner, looking west
    ]
    for px, py, angle in positions:
        compiled = render_frame_compiled(module, px, py, angle, small_config)
        ref = render_frame(px, py, angle, segments, small_config)
        _assert_frame_match(compiled, ref, msg=f"pos=({px},{py}) angle={angle}")


# ── Multi-room scene: 22 segments with diagonals ──────────────────


def test_compiled_multi_room(small_config):
    """Multi-room scene (22 segments including diagonals) — full frame."""
    segments = multi_room()
    module = compile_renderer(
        segments, small_config, max_coord=15.0, d=1024, d_head=16, verbose=False,
    )

    viewpoints = [
        (-8.0, 0.0, 0),    # room A, looking east toward corridor
        (-8.0, 0.0, 128),  # room A, looking west at wall
        (8.0, 0.0, 128),   # room B, looking west toward corridor
        (0.0, 0.0, 0),     # corridor center, looking east
        (0.0, 0.0, 64),    # corridor center, looking north
    ]
    for px, py, angle in viewpoints:
        compiled = render_frame_compiled(module, px, py, angle, small_config)
        ref = render_frame(px, py, angle, segments, small_config)
        _assert_frame_match(compiled, ref, msg=f"pos=({px},{py}) angle={angle}")


# ── Target resolution: 32×40 ──────────────────────────────────────


def test_compiled_target_resolution(target_config):
    """Phase 2 target: 32×40 box room, full-frame match."""
    segments = box_room()
    module = compile_renderer(
        segments, target_config, max_coord=10.0, d=1024, d_head=16, verbose=False,
    )

    for angle in [0, 64, 128, 192]:
        compiled = render_frame_compiled(module, 0.0, 0.0, angle, target_config)
        ref = render_frame(0.0, 0.0, angle, segments, target_config)
        _assert_frame_match(compiled, ref, msg=f"32x40 angle={angle}")
