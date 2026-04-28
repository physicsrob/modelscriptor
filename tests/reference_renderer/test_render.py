import numpy as np
import pytest

from torchwright.reference_renderer import (
    RenderConfig,
    Segment,
    generate_trig_table,
    render_frame,
    save_png,
    box_room,
)
from torchwright.reference_renderer.render import intersect_ray_segment


@pytest.fixture
def config():
    """Small screen for fast, auditable tests.

    These tests use a "small-world" scale: walls 2 world units tall
    centred on the eye (``_wall_seg`` builds them with floor=-1,
    ceiling=1) so the geometry stays a few units across without
    pixels overflowing the screen.  The renderer code is the same as
    for DOOM-scale scenes; only the chosen world units differ.
    """
    return RenderConfig(
        screen_width=32,
        screen_height=24,
        fov_columns=64,  # 90° HFOV, DOOM canonical
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
        player_eye_z=0.0,
    )


WALL_COLOR = (1.0, 0.0, 0.0)
WALL2_COLOR = (0.0, 1.0, 0.0)


def _wall_seg(ax, ay, bx, by, color, *, texture_id=-1):
    """Test helper: build a one-sided sector-aware Segment.

    Small-world scale (floor=-1, ceiling=1 — 2 world units tall
    centred on eye=0).  Caller orients the seg so FRONT (right of
    a→b) faces the player at the origin; the renderer back-face culls
    segs whose front faces away.
    """
    return Segment(
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        color=color,
        texture_id=texture_id,
        front_floor=-1.0,
        front_ceiling=1.0,
    )


# ── Test 1: Empty scene ────────────────────────────────────────────


def test_empty_scene(config):
    """No segments → ceiling above center, floor below."""
    frame = render_frame(0.0, 0.0, 0, [], config)
    assert frame.shape == (24, 32, 3)

    center = config.screen_height // 2
    # Every column: ceiling above, floor below
    for col in range(config.screen_width):
        for row in range(center):
            np.testing.assert_array_equal(frame[row, col], config.ceiling_color)
        for row in range(center, config.screen_height):
            np.testing.assert_array_equal(frame[row, col], config.floor_color)


# ── Test 2: Head-on wall ───────────────────────────────────────────


def test_head_on_wall(config):
    """Wall perpendicular to view direction, directly ahead.

    Player at (0, 0) facing angle 0 (east / +x).
    Wall segment at x=5, spanning y=-10 to y=10.
    """
    # Wall east of player; FRONT must face west (player side) → seg
    # goes from north to south (right of south = west).
    seg = _wall_seg(5.0, 10.0, 5.0, -10.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg], config)

    # Center column (col=16) should hit the wall
    center_col = config.screen_width // 2
    center_row = config.screen_height // 2

    # The wall color should appear at the screen center
    np.testing.assert_array_equal(frame[center_row, center_col], WALL_COLOR)

    # Ceiling should still be visible at the top
    np.testing.assert_array_equal(frame[0, center_col], config.ceiling_color)

    # Floor should still be visible at the bottom
    np.testing.assert_array_equal(frame[-1, center_col], config.floor_color)


# ── Test 3: Angled wall ───────────────────────────────────────────


def test_angled_wall(config):
    """Diagonal segment — verify it produces wall pixels in multiple columns.

    Player at (0, 0) facing angle 0 (east).
    Wall runs diagonally from (4, -3) to (8, 3).
    """
    # Diagonal wall; FRONT must face the player (player at origin, west of wall).
    # cross = dy*(px-ax) - dx*(py-ay) > 0 ⇒ player on FRONT.
    # For (8,3) → (4,-3): dx=-4 dy=-6 → cross = -6*(0-8) - (-4)*(0-3) = 36 > 0. ✓
    seg = _wall_seg(8.0, 3.0, 4.0, -3.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg], config)

    center_row = config.screen_height // 2
    # At least some columns should show wall color at the center row
    wall_cols = [
        col
        for col in range(config.screen_width)
        if np.array_equal(frame[center_row, col], WALL_COLOR)
    ]
    assert len(wall_cols) > 1, "Angled wall should hit multiple columns"


# ── Test 4: Corner view ───────────────────────────────────────────


def test_corner_view(config):
    """Two walls meeting at a corner — nearest is rendered per column.

    Player at (0, 0) facing east. Two walls form a V:
    - Wall A at x=3, y=-10 to y=0 (left side, closer)
    - Wall B at x=6, y=0 to y=10  (right side, farther)
    """
    # Both walls east of player; FRONT must face west → seg goes south.
    seg_a = _wall_seg(3.0, 0.0, 3.0, -10.0, WALL_COLOR)
    seg_b = _wall_seg(6.0, 10.0, 6.0, 0.0, WALL2_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg_a, seg_b], config)

    center_row = config.screen_height // 2
    # Columns pointing left of center (negative y direction) should hit seg_a (closer)
    # Columns pointing right of center (positive y direction) should hit seg_b
    has_red = False
    has_green = False
    for col in range(config.screen_width):
        pixel = frame[center_row, col]
        if np.array_equal(pixel, WALL_COLOR):
            has_red = True
        if np.array_equal(pixel, WALL2_COLOR):
            has_green = True
    assert has_red, "Should see wall A"
    assert has_green, "Should see wall B"


# ── Test 5: Through doorway ───────────────────────────────────────


def test_through_doorway(config):
    """Gap between two segments — gap columns show ceiling/floor only.

    Player at (0, 0) facing east.
    Wall left:  x=5, y=-10 to y=-1
    Wall right: x=5, y=1 to y=10
    Gap at y in (-1, 1).
    """
    # Both walls east of player; FRONT must face west → seg goes south.
    left = _wall_seg(5.0, -1.0, 5.0, -10.0, WALL_COLOR)
    right = _wall_seg(5.0, 10.0, 5.0, 1.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [left, right], config)

    # Center column (angle 0, pointing straight east along y=0) goes through the gap
    center_col = config.screen_width // 2
    center_row = config.screen_height // 2
    # Should see ceiling or floor, not wall
    pixel = tuple(frame[center_row, center_col].tolist())
    assert (
        pixel == config.ceiling_color or pixel == config.floor_color
    ), f"Center column through doorway should not show wall, got {pixel}"


# ── Test 6: Parallel ray ──────────────────────────────────────────


def test_parallel_ray(config):
    """Segment exactly parallel to a ray — no intersection.

    Player at (0, 0) facing east. Segment runs along the x-axis at y=0.
    The center ray (angle 0, direction +x) is parallel to this segment.
    """
    # Even with sector data, the ray-segment intersection rejects
    # exactly-parallel rays.  Orientation here is arbitrary — only the
    # geometric reject matters.  Make FRONT face the player so the
    # cull doesn't pre-empt the parallel-rejection test.
    seg = _wall_seg(8.0, 0.0, 3.0, 0.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg], config)

    center_col = config.screen_width // 2
    center_row = config.screen_height // 2
    # Center column should show ceiling/floor, not wall
    pixel = tuple(frame[center_row, center_col].tolist())
    assert (
        pixel == config.floor_color
    ), f"Parallel ray should not hit segment, got {pixel}"


# ── Test 7: Behind player ─────────────────────────────────────────


def test_behind_player(config):
    """Segment entirely behind the player — should not render.

    Player at (0, 0) facing east (+x). Wall at x=-5.
    """
    # Wall west of player at x=-5; FRONT faces east → going north.
    seg = _wall_seg(-5.0, -10.0, -5.0, 10.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg], config)

    center_col = config.screen_width // 2
    center_row = config.screen_height // 2
    pixel = tuple(frame[center_row, center_col].tolist())
    assert (
        pixel == config.floor_color
    ), f"Wall behind player should not render, got {pixel}"


# ── Test 8: Multiple depths ───────────────────────────────────────


def test_multiple_depths(config):
    """Two walls at different distances — nearer one wins.

    Player at (0, 0) facing east.
    Near wall at x=3, far wall at x=8, both spanning y=-10 to y=10.
    """
    # Both walls east of player; FRONT must face west → seg goes south.
    near = _wall_seg(3.0, 10.0, 3.0, -10.0, WALL_COLOR)
    far = _wall_seg(8.0, 10.0, 8.0, -10.0, WALL2_COLOR)
    frame = render_frame(0.0, 0.0, 0, [near, far], config)

    center_col = config.screen_width // 2
    center_row = config.screen_height // 2
    np.testing.assert_array_equal(
        frame[center_row, center_col],
        WALL_COLOR,
        err_msg="Nearer wall should occlude farther wall",
    )


# ── Test 9: intersect_ray_segment unit test ────────────────────────


def test_intersect_direct_hit():
    """Ray pointing +x hits a vertical segment at x=5."""
    seg = _wall_seg(5.0, -1.0, 5.0, 1.0, WALL_COLOR)
    hit = intersect_ray_segment(0.0, 0.0, 1.0, 0.0, seg)
    assert hit is not None
    t, u = hit
    assert abs(t - 5.0) < 1e-10


def test_intersect_miss_behind():
    """Segment behind the ray origin returns None."""
    seg = _wall_seg(-5.0, -1.0, -5.0, 1.0, WALL_COLOR)
    hit = intersect_ray_segment(0.0, 0.0, 1.0, 0.0, seg)
    assert hit is None


def test_intersect_miss_parallel():
    """Ray parallel to segment returns None."""
    seg = _wall_seg(3.0, 0.0, 8.0, 0.0, WALL_COLOR)
    hit = intersect_ray_segment(0.0, 0.0, 1.0, 0.0, seg)
    assert hit is None


def test_intersect_miss_outside_segment():
    """Ray would hit the line but misses the segment (u outside [0, 1])."""
    # Segment from (5, 2) to (5, 3) — ray along y=0 misses
    seg = _wall_seg(5.0, 2.0, 5.0, 3.0, WALL_COLOR)
    hit = intersect_ray_segment(0.0, 0.0, 1.0, 0.0, seg)
    assert hit is None


# ── Box room tests ─────────────────────────────────────────────────


def test_box_room():
    """box_room returns 4 segments with distinct colors."""
    segments = box_room()
    assert len(segments) == 4
    colors = {seg.color for seg in segments}
    assert len(colors) == 4, "Each wall should have a distinct color"


def test_box_room_render():
    """Render from centre of box room — rotating through 4 directions sees all 4 walls.

    Uses a DOOM-scale config (eye=41 sits between floor=0 and
    ceiling=128 of the 256-unit box) so the ray hits the wall mid-
    height regardless of facing.
    """
    cfg = RenderConfig(
        screen_width=32,
        screen_height=24,
        fov_columns=64,
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
        player_eye_z=41.0,
    )
    segments = box_room()

    colors_seen = set()
    for angle in [0, 64, 128, 192]:  # east, north, west, south
        frame = render_frame(0.0, 0.0, angle, segments, cfg)
        center_col = cfg.screen_width // 2
        center_row = cfg.screen_height // 2
        pixel = tuple(frame[center_row, center_col].tolist())
        if pixel != cfg.ceiling_color and pixel != cfg.floor_color:
            colors_seen.add(pixel)

    wall_colors = {seg.color for seg in segments}
    assert colors_seen == wall_colors, f"Expected all 4 wall colors, saw {colors_seen}"


# ── PNG output tests ───────────────────────────────────────────────


def test_save_png(config, tmp_path):
    """save_png writes a valid PNG file."""
    seg = _wall_seg(5.0, 10.0, 5.0, -10.0, WALL_COLOR)
    frame = render_frame(0.0, 0.0, 0, [seg], config)

    path = tmp_path / "test.png"
    save_png(frame, str(path))

    assert path.exists()
    assert path.stat().st_size > 0

    from PIL import Image

    img = Image.open(path)
    assert img.size == (config.screen_width, config.screen_height)
    assert img.mode == "RGB"
