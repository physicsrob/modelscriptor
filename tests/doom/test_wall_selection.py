"""Tests for wall sorting, visibility masks, and compiled rendering accuracy.

Structural tests that catch real rendering failures:
- Wall height must match reference within 2 rows
- Texture colors at the center column must match
- Different walls must produce different textures
- Wall coverage (fraction of non-ceiling/floor pixels) must match
"""

import math

import numpy as np
import pytest
import torch

from torchwright.reference_renderer.render import render_frame, intersect_ray_segment
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


TRIG = generate_trig_table()


def _test_textures(tex_size=8):
    """High-contrast solid-color textures for compiled tests.

    Every texel is far from both ceiling (0.2, 0.2, 0.2) and floor
    (0.4, 0.4, 0.4), so _wall_band never misclassifies wall pixels.
    """
    colors = [
        (0.9, 0.1, 0.1),  # east: red
        (0.1, 0.9, 0.1),  # west: green
        (0.1, 0.1, 0.9),  # north: blue
        (0.9, 0.9, 0.1),  # south: yellow
    ]
    textures = []
    for r, g, b in colors:
        tex = np.full((tex_size, tex_size, 3), [r, g, b], dtype=np.float64)
        textures.append(tex)
    return textures


def _config(W=32, H=24, fov=8):
    return RenderConfig(
        screen_width=W,
        screen_height=H,
        fov_columns=fov,
        trig_table=TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _ref_frame(segments, textures, config, px, py, angle):
    return np.array(
        render_frame(px, py, angle, segments, config, textures=textures),
        dtype=np.float32,
    )


def _wall_band(column, ceiling_color, floor_color, tol=0.05):
    """Find the wall band [top, bottom) in a rendered column.

    Returns (top_row, bottom_row) of the wall region, or None if no wall.
    The wall region is where pixels differ from both ceiling and floor.
    """
    H = len(column)
    ceil_c = np.array(ceiling_color)
    floor_c = np.array(floor_color)
    is_wall = np.array([
        np.linalg.norm(column[r] - ceil_c) > tol
        and np.linalg.norm(column[r] - floor_c) > tol
        for r in range(H)
    ])
    if not is_wall.any():
        return None
    indices = np.where(is_wall)[0]
    return int(indices[0]), int(indices[-1]) + 1


# ---------------------------------------------------------------------------
# Reference-only: central ray sort ordering
# ---------------------------------------------------------------------------


class TestCentralRaySortOrder:

    def _ref_dists(self, segments, px, py, angle):
        cos_a = TRIG[angle % 256, 0]
        sin_a = TRIG[angle % 256, 1]
        return [
            (intersect_ray_segment(px, py, cos_a, sin_a, s) or (float("inf"),))[0]
            for s in segments
        ]

    def test_facing_east(self):
        segs = [
            Segment(ax=5, ay=-5, bx=5, by=5, color=(1, 0, 0)),
            Segment(ax=-5, ay=5, bx=-5, by=-5, color=(0, 1, 0)),
        ]
        d = self._ref_dists(segs, 0, 0, 0)
        assert d[0] < d[1]

    def test_facing_north(self):
        segs = [
            Segment(ax=5, ay=-5, bx=5, by=5, color=(1, 0, 0)),
            Segment(ax=-5, ay=5, bx=5, by=5, color=(0, 0, 1)),
        ]
        d = self._ref_dists(segs, 0, 0, 64)
        assert d[1] < d[0]

    def test_off_center(self):
        segs = [
            Segment(ax=5, ay=-5, bx=5, by=5, color=(1, 0, 0)),
            Segment(ax=-5, ay=5, bx=-5, by=-5, color=(0, 1, 0)),
        ]
        d = self._ref_dists(segs, 4, 0, 0)
        assert d[0] == pytest.approx(1.0, abs=0.1)

    def test_parallel_infinite(self):
        segs = [Segment(ax=-5, ay=5, bx=5, by=5, color=(0, 0, 1))]
        d = self._ref_dists(segs, 0, 0, 0)
        assert d[0] == float("inf")


# ---------------------------------------------------------------------------
# Reference-only: column visibility
# ---------------------------------------------------------------------------


class TestColumnVisibility:

    def _vis_cols(self, seg, px, py, angle, config):
        W, fov = config.screen_width, config.fov_columns
        cols = []
        for col in range(W):
            ao = (col - W // 2) * fov // W
            ra = (angle + ao) % 256
            if intersect_ray_segment(px, py, TRIG[ra, 0], TRIG[ra, 1], seg):
                cols.append(col)
        return cols

    def test_full_span(self):
        cfg = _config(32, 24, 8)
        v = self._vis_cols(Segment(ax=5, ay=-5, bx=5, by=5, color=(1, 0, 0)),
                           0, 0, 0, cfg)
        assert len(v) == 32

    def test_behind_invisible(self):
        cfg = _config(32, 24, 8)
        v = self._vis_cols(Segment(ax=-5, ay=-5, bx=-5, by=5, color=(0, 1, 0)),
                           0, 0, 0, cfg)
        assert len(v) == 0

    def test_contiguous(self):
        cfg = RenderConfig(screen_width=64, screen_height=24, fov_columns=32,
                           trig_table=TRIG, ceiling_color=(0, 0, 0),
                           floor_color=(0.5, 0.5, 0.5))
        v = self._vis_cols(Segment(ax=5, ay=-1, bx=5, by=1, color=(1, 0, 0)),
                           0, 0, 0, cfg)
        if len(v) >= 2:
            assert v[-1] - v[0] + 1 == len(v)

    def test_partial_span(self):
        cfg = RenderConfig(screen_width=64, screen_height=48, fov_columns=32,
                           trig_table=TRIG, ceiling_color=(0, 0, 0),
                           floor_color=(0.5, 0.5, 0.5))
        v = self._vis_cols(Segment(ax=5, ay=-1, bx=5, by=1, color=(1, 0, 0)),
                           0, 0, 0, cfg)
        assert 0 < len(v) < 64


# ---------------------------------------------------------------------------
# Structural: reference renderer sanity (these document what "correct" means)
# ---------------------------------------------------------------------------


class TestReferenceStructure:
    """Verify structural properties of the reference renderer that the
    compiled graph must also satisfy."""

    @pytest.fixture(scope="class")
    def box_data(self):
        segs, _ = box_room_textured(size=10.0, wad_path="doom1.wad", tex_size=8)
        texs = _test_textures(tex_size=8)
        cfg = _config(W=32, H=24, fov=8)
        return segs, texs, cfg

    def test_wall_height_at_center(self, box_data):
        """Facing east at center: wall at dist 5, height = H/5 ≈ 4.8 rows."""
        segs, texs, cfg = box_data
        frame = _ref_frame(segs, texs, cfg, 0, 0, 0)
        band = _wall_band(frame[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band is not None, "Should see a wall at center column"
        height = band[1] - band[0]
        expected = cfg.screen_height / 5.0
        assert abs(height - expected) < 2, f"Wall height {height} != expected ~{expected:.1f}"

    def test_wall_height_near_wall(self, box_data):
        """Near east wall (dist 2): wall should be taller than at center."""
        segs, texs, cfg = box_data
        frame_far = _ref_frame(segs, texs, cfg, 0, 0, 0)
        frame_near = _ref_frame(segs, texs, cfg, 3, 0, 0)
        band_far = _wall_band(frame_far[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        band_near = _wall_band(frame_near[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band_far is not None and band_near is not None
        h_far = band_far[1] - band_far[0]
        h_near = band_near[1] - band_near[0]
        assert h_near > h_far, f"Near wall ({h_near}px) should be taller than far ({h_far}px)"

    def test_different_walls_different_textures(self, box_data):
        """Facing east vs north: different walls have visually different pixels."""
        segs, texs, cfg = box_data
        frame_e = _ref_frame(segs, texs, cfg, 0, 0, 0)
        frame_n = _ref_frame(segs, texs, cfg, 0, 0, 64)
        # Extract center column wall pixels
        band_e = _wall_band(frame_e[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        band_n = _wall_band(frame_n[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band_e is not None and band_n is not None
        wall_e = frame_e[band_e[0]:band_e[1], 16, :]
        wall_n = frame_n[band_n[0]:band_n[1], 16, :]
        # Average color should differ (different textures)
        mean_e = wall_e.mean(axis=0)
        mean_n = wall_n.mean(axis=0)
        diff = np.linalg.norm(mean_e - mean_n)
        assert diff > 0.05, f"Different walls should have different textures, diff={diff:.4f}"

    def test_wall_coverage_fraction(self, box_data):
        """At center of box room, wall should cover roughly H/5 / H ≈ 20% of each column."""
        segs, texs, cfg = box_data
        frame = _ref_frame(segs, texs, cfg, 0, 0, 0)
        H = cfg.screen_height
        wall_rows = 0
        for col in range(cfg.screen_width):
            band = _wall_band(frame[:, col, :], cfg.ceiling_color, cfg.floor_color)
            if band:
                wall_rows += band[1] - band[0]
        frac = wall_rows / (H * cfg.screen_width)
        assert 0.1 < frac < 0.5, f"Wall coverage {frac:.2f} out of expected range"


# ---------------------------------------------------------------------------
# Compiled graph: structural correctness
# ---------------------------------------------------------------------------


class TestCompiledStructure:
    """Compiled graph must match reference renderer on structural properties,
    not just per-pixel error."""

    @pytest.fixture(scope="class")
    def box_data(self):
        segs, _ = box_room_textured(size=10.0, wad_path="doom1.wad", tex_size=8)
        texs = _test_textures(tex_size=8)
        cfg = _config(W=32, H=24, fov=8)
        return segs, texs, cfg

    @pytest.fixture(scope="class")
    def module(self, box_data):
        from torchwright.doom.compile import compile_game
        segs, texs, cfg = box_data
        return compile_game(cfg, texs, max_walls=8, max_coord=10.0, d=2048, verbose=False)

    def _compiled_frame(self, module, box_data, px, py, angle):
        from torchwright.doom.compile import step_frame
        from torchwright.doom.game import GameState
        from torchwright.doom.input import PlayerInput
        from torchwright.doom.map_subset import build_scene_subset
        segs, texs, cfg = box_data
        subset = build_scene_subset(segs, texs)
        state = GameState(x=px, y=py, angle=angle)
        frame, _ = step_frame(module, state, PlayerInput(), subset, cfg,
                              textures=texs)
        return frame

    def test_wall_height_center_column(self, module, box_data):
        """Wall height at center column must match reference within 2 rows."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 0)
        comp = self._compiled_frame(module, box_data, 0, 0, 0)

        ref_band = _wall_band(ref[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        comp_band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)

        assert ref_band is not None, "Reference should show wall"
        assert comp_band is not None, "Compiled should show wall"

        ref_h = ref_band[1] - ref_band[0]
        comp_h = comp_band[1] - comp_band[0]
        assert abs(ref_h - comp_h) <= 2, \
            f"Wall height: compiled={comp_h} vs reference={ref_h}"

    def test_wall_top_bottom_match(self, module, box_data):
        """Wall top and bottom row must match reference within 2 rows."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 0)
        comp = self._compiled_frame(module, box_data, 0, 0, 0)

        ref_band = _wall_band(ref[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        comp_band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)

        assert ref_band is not None and comp_band is not None
        assert abs(ref_band[0] - comp_band[0]) <= 2, \
            f"Wall top: compiled={comp_band[0]} vs reference={ref_band[0]}"
        assert abs(ref_band[1] - comp_band[1]) <= 2, \
            f"Wall bottom: compiled={comp_band[1]} vs reference={ref_band[1]}"

    def test_correct_texture_facing_east(self, module, box_data):
        """Center column wall color must match reference (correct texture)."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 0)
        comp = self._compiled_frame(module, box_data, 0, 0, 0)

        ref_band = _wall_band(ref[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        comp_band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert ref_band is not None and comp_band is not None

        # Compare average wall color (robust to slight height differences)
        ref_wall = ref[ref_band[0]:ref_band[1], 16, :]
        comp_wall = comp[comp_band[0]:comp_band[1], 16, :]
        ref_mean = ref_wall.mean(axis=0)
        comp_mean = comp_wall.mean(axis=0)
        diff = np.linalg.norm(ref_mean - comp_mean)
        assert diff < 0.15, \
            f"Wall texture color mismatch: ref={ref_mean} comp={comp_mean} diff={diff:.3f}"

    def test_correct_texture_facing_north(self, module, box_data):
        """Facing north: texture should match reference (different from east)."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 64)
        comp = self._compiled_frame(module, box_data, 0, 0, 64)

        ref_band = _wall_band(ref[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        comp_band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert ref_band is not None and comp_band is not None

        ref_wall = ref[ref_band[0]:ref_band[1], 16, :]
        comp_wall = comp[comp_band[0]:comp_band[1], 16, :]
        ref_mean = ref_wall.mean(axis=0)
        comp_mean = comp_wall.mean(axis=0)
        diff = np.linalg.norm(ref_mean - comp_mean)
        assert diff < 0.15, \
            f"Wall texture color mismatch: ref={ref_mean} comp={comp_mean} diff={diff:.3f}"

    def test_different_walls_different_textures(self, module, box_data):
        """Compiled east vs north frames must produce different wall textures."""
        segs, texs, cfg = box_data
        frame_e = self._compiled_frame(module, box_data, 0, 0, 0)
        frame_n = self._compiled_frame(module, box_data, 0, 0, 64)

        band_e = _wall_band(frame_e[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        band_n = _wall_band(frame_n[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band_e is not None and band_n is not None

        mean_e = frame_e[band_e[0]:band_e[1], 16, :].mean(axis=0)
        mean_n = frame_n[band_n[0]:band_n[1], 16, :].mean(axis=0)
        diff = np.linalg.norm(mean_e - mean_n)
        assert diff > 0.05, \
            f"East and north walls should have different textures, diff={diff:.4f}"

    def test_wall_coverage_matches_reference(self, module, box_data):
        """Total wall pixel fraction must match reference within 10%."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 0)
        comp = self._compiled_frame(module, box_data, 0, 0, 0)

        def _coverage(frame):
            total = 0
            for col in range(cfg.screen_width):
                band = _wall_band(frame[:, col, :], cfg.ceiling_color, cfg.floor_color)
                if band:
                    total += band[1] - band[0]
            return total / (cfg.screen_height * cfg.screen_width)

        ref_cov = _coverage(ref)
        comp_cov = _coverage(comp)
        assert abs(ref_cov - comp_cov) < 0.1, \
            f"Wall coverage: compiled={comp_cov:.3f} vs reference={ref_cov:.3f}"

    def test_per_column_wall_height(self, module, box_data):
        """Every column's wall height must match reference within 3 rows."""
        segs, texs, cfg = box_data
        ref = _ref_frame(segs, texs, cfg, 0, 0, 0)
        comp = self._compiled_frame(module, box_data, 0, 0, 0)

        mismatches = []
        for col in range(cfg.screen_width):
            ref_band = _wall_band(ref[:, col, :], cfg.ceiling_color, cfg.floor_color)
            comp_band = _wall_band(comp[:, col, :], cfg.ceiling_color, cfg.floor_color)
            if ref_band is None and comp_band is None:
                continue
            if ref_band is None or comp_band is None:
                mismatches.append((col, ref_band, comp_band))
                continue
            ref_h = ref_band[1] - ref_band[0]
            comp_h = comp_band[1] - comp_band[0]
            if abs(ref_h - comp_h) > 3:
                mismatches.append((col, ref_h, comp_h))

        assert len(mismatches) == 0, \
            f"{len(mismatches)} columns with wall height mismatch: {mismatches[:5]}"

    def test_near_wall_fills_screen(self, module, box_data):
        """Standing 2 units from east wall: wall should fill most of screen."""
        segs, texs, cfg = box_data
        comp = self._compiled_frame(module, box_data, 3, 0, 0)
        band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band is not None, "Should see a wall"
        height = band[1] - band[0]
        assert height >= cfg.screen_height * 0.4, \
            f"Near wall should fill >=40% of screen, got {height}/{cfg.screen_height}"

    @pytest.mark.parametrize("angle,wall_name", [
        (0, "east"),
        (64, "north"),
        (128, "west"),
        (192, "south"),
    ])
    def test_four_directions_wall_visible(self, module, box_data, angle, wall_name):
        """Each cardinal direction should show a wall at the center column."""
        segs, texs, cfg = box_data
        comp = self._compiled_frame(module, box_data, 0, 0, angle)
        band = _wall_band(comp[:, 16, :], cfg.ceiling_color, cfg.floor_color)
        assert band is not None, f"Should see {wall_name} wall when facing {wall_name}"
        assert band[1] - band[0] >= 3, \
            f"{wall_name} wall too short: {band[1] - band[0]} rows"
