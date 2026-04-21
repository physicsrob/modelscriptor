"""Pipeline-trace tests for the DOOM renderer.

Compiles the full game at walkthrough resolution (64x80, fov=32), runs
step_frame with trace capture, and compares intermediate outputs at each
stage boundary against reference values.  When something breaks, the
first failing boundary tells you exactly which stage diverged.
"""

import numpy as np
import pytest

from tests._utils.image_compare import compare_images
from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.doom.trace import FrameTrace
from torchwright.reference_renderer.render import (
    project_wall,
    render_frame,
    render_wall_column,
)
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TRIG = generate_trig_table()


def _box_room_config():
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room_segments(half=5.0):
    return [
        Segment(ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0),
        Segment(ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1),
        Segment(ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2),
        Segment(ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3),
    ]


def _ref_bsp_ranks(subset, px, py):
    """Compute reference BSP ranks for each segment at player position (px, py)."""
    n_cols = subset.seg_bsp_coeffs.shape[1]
    side_P_vec = np.zeros(n_cols)
    for i, node in enumerate(subset.bsp_nodes):
        val = node.nx * px + node.ny * py + node.d
        side_P_vec[i] = 1.0 if val > 0 else 0.0
    ranks = subset.seg_bsp_coeffs @ side_P_vec + subset.seg_bsp_consts
    return ranks


class TestPipeline:
    """Full pipeline tests at walkthrough resolution (64x80)."""

    @pytest.fixture(scope="class")
    def scene(self):
        config = _box_room_config()
        textures = default_texture_atlas()
        segs = _box_room_segments()
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs

    @pytest.fixture(scope="class")
    def module(self, scene):
        config, textures, subset, segs = scene
        return compile_game(
            config,
            textures,
            max_walls=8,
            d=2048,
            d_head=32,
            verbose=False,
        )

    def _run_frame(self, module, scene, px, py, angle, forward=False):
        config, textures, subset, segs = scene
        state = GameState(x=px, y=py, angle=angle, move_speed=0.3, turn_speed=4)
        inp = PlayerInput(forward=forward)
        trace = FrameTrace()
        frame, new_state = step_frame(
            module, state, inp, subset, config, textures=textures, trace=trace
        )
        return frame, new_state, trace

    # ── EOS boundary ──────────────────────────────────────────────

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_eos_state_matches_reference_cardinal(self, module, scene, angle):
        config, textures, subset, segs = scene
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        inp = PlayerInput()
        ref = update_state(state, inp, segs, _TRIG)

        _, new_state, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        assert trace.eos_resolved_x == pytest.approx(ref.x, abs=0.15)
        assert trace.eos_resolved_y == pytest.approx(ref.y, abs=0.15)
        assert trace.eos_new_angle == pytest.approx(float(ref.angle), abs=1.5)

    @pytest.mark.parametrize("angle", [20, 45, 100, 160, 210])
    def test_eos_state_matches_reference_oblique(self, module, scene, angle):
        config, textures, subset, segs = scene
        state = GameState(x=0.0, y=0.0, angle=angle, move_speed=0.3, turn_speed=4)
        inp = PlayerInput()
        ref = update_state(state, inp, segs, _TRIG)

        _, new_state, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        assert trace.eos_resolved_x == pytest.approx(ref.x, abs=0.15)
        assert trace.eos_resolved_y == pytest.approx(ref.y, abs=0.15)
        assert trace.eos_new_angle == pytest.approx(float(ref.angle), abs=1.5)

    def test_eos_collision_blocks_wall(self, module, scene):
        config, textures, subset, segs = scene
        state = GameState(x=4.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        inp = PlayerInput(forward=True)
        ref = update_state(state, inp, segs, _TRIG)

        _, new_state, trace = self._run_frame(module, scene, 4.0, 0.0, 0, forward=True)

        assert trace.eos_resolved_x == pytest.approx(ref.x, abs=0.15)
        assert trace.eos_resolved_x < 5.0, f"Passed through wall: x={trace.eos_resolved_x}"

    # ── SORTED boundary ───────────────────────────────────────────

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_sort_order_matches_bsp_cardinal(self, module, scene, angle):
        """Walls picked in BSP rank order at cardinal angles."""
        config, textures, subset, segs = scene
        _, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        ref_ranks = _ref_bsp_ranks(subset, 0.0, 0.0)

        # Reference: which walls are renderable and their rank order
        renderable = []
        for i, seg in enumerate(segs):
            proj = project_wall(0.0, 0.0, angle, seg, config)
            if proj is not None:
                renderable.append((ref_ranks[i], i))
        renderable.sort()
        expected_order = [wall_i for _, wall_i in renderable]

        compiled_order = [s.selected_wall_index for s in trace.sort_steps]
        assert len(compiled_order) == len(expected_order), (
            f"n_renderable mismatch: compiled={len(compiled_order)}, ref={len(expected_order)}"
        )
        assert compiled_order == expected_order, (
            f"sort order mismatch: compiled={compiled_order}, expected={expected_order}"
        )

    @pytest.mark.parametrize("angle", [20, 45, 100, 160, 210])
    def test_sort_order_matches_bsp_oblique(self, module, scene, angle):
        """Walls picked in BSP rank order at oblique angles."""
        config, textures, subset, segs = scene
        _, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        ref_ranks = _ref_bsp_ranks(subset, 0.0, 0.0)

        renderable = []
        for i, seg in enumerate(segs):
            proj = project_wall(0.0, 0.0, angle, seg, config)
            if proj is not None:
                renderable.append((ref_ranks[i], i))
        renderable.sort()
        expected_order = [wall_i for _, wall_i in renderable]

        compiled_order = [s.selected_wall_index for s in trace.sort_steps]
        assert len(compiled_order) == len(expected_order), (
            f"n_renderable mismatch: compiled={len(compiled_order)}, ref={len(expected_order)}"
        )
        assert compiled_order == expected_order, (
            f"sort order mismatch: compiled={compiled_order}, expected={expected_order}"
        )

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_sort_visibility_matches_reference(self, module, scene, angle):
        """vis_lo/vis_hi per wall match reference projection."""
        config, textures, subset, segs = scene
        _, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        for step in trace.sort_steps:
            wall_i = step.selected_wall_index
            proj = project_wall(0.0, 0.0, angle, segs[wall_i], config)
            assert proj is not None, f"wall {wall_i} not renderable in reference"
            assert step.vis_lo == pytest.approx(proj.vis_lo, abs=2), (
                f"wall {wall_i} vis_lo: compiled={step.vis_lo}, ref={proj.vis_lo}"
            )
            assert step.vis_hi == pytest.approx(proj.vis_hi, abs=2), (
                f"wall {wall_i} vis_hi: compiled={step.vis_hi}, ref={proj.vis_hi}"
            )

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_sort_done_fires_correctly(self, module, scene, angle):
        """n_renderable count covers all reference-visible walls.

        The compiled renderer's renderability criterion (central ray hits
        wall and is in front) is broader than ``project_wall`` (at least
        one screen column's ray intersects).  So n_renderable >= ref count.
        """
        config, textures, subset, segs = scene
        _, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        ref_renderable = sum(
            1 for seg in segs if project_wall(0.0, 0.0, angle, seg, config) is not None
        )
        assert trace.n_renderable >= ref_renderable, (
            f"n_renderable: compiled={trace.n_renderable} < ref={ref_renderable}"
        )
        assert trace.n_renderable <= len(segs), (
            f"n_renderable: compiled={trace.n_renderable} > total walls={len(segs)}"
        )

    # ── Structural checks ─────────────────────────────────────────

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_walls_are_visible(self, module, scene, angle):
        """Wall pixels exist and cover >5% of frame."""
        config, textures, subset, segs = scene
        frame, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        ceil = np.array(config.ceiling_color)
        floor = np.array(config.floor_color)
        H, W = config.screen_height, config.screen_width

        wall_pixels = 0
        for y in range(H):
            for x in range(W):
                pix = frame[y, x]
                is_ceil = np.allclose(pix, ceil, atol=0.05)
                is_floor = np.allclose(pix, floor, atol=0.05)
                if not is_ceil and not is_floor:
                    wall_pixels += 1

        fraction = wall_pixels / (H * W)
        assert fraction > 0.05, (
            f"angle={angle}: only {fraction:.1%} wall pixels (expected >5%)"
        )

    # ── RENDER boundary ───────────────────────────────────────────

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_render_wall_heights(self, module, scene, angle):
        """Wall height at mid-column matches reference."""
        config, textures, subset, segs = scene
        _, _, trace = self._run_frame(module, scene, 0.0, 0.0, angle)

        for step in trace.sort_steps:
            wall_i = step.selected_wall_index
            proj = project_wall(0.0, 0.0, angle, segs[wall_i], config)
            if proj is None:
                continue
            mid_col = (proj.vis_lo + proj.vis_hi) // 2
            ref_result = render_wall_column(
                mid_col, proj, 0.0, 0.0, angle, config, textures
            )
            if ref_result is None:
                continue

            ref_height = ref_result.wall_bottom - ref_result.wall_top

            # Find the compiled wall height from render steps at this mid_col
            compiled_rows = set()
            for rs in trace.render_steps:
                if rs.wall_index == trace.sort_steps.index(step) and rs.col == mid_col:
                    for r in range(rs.length):
                        compiled_rows.add(rs.start + r)
            compiled_height = len(compiled_rows)

            if ref_height > 0:
                assert compiled_height == pytest.approx(ref_height, abs=3), (
                    f"wall {wall_i} col {mid_col}: "
                    f"compiled_height={compiled_height}, ref_height={ref_height}"
                )

    # ── Final frame ───────────────────────────────────────────────

    @pytest.mark.parametrize("angle", [0, 64, 128, 192])
    def test_frame_matches_reference_cardinal(self, module, scene, angle):
        config, textures, subset, segs = scene
        frame, _, _ = self._run_frame(module, scene, 0.0, 0.0, angle)
        ref = render_frame(0.0, 0.0, angle, segs, config, textures=textures)
        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.96, max_err=0.30
        )

    @pytest.mark.parametrize("angle", [20, 45, 100, 160, 210])
    def test_frame_matches_reference_oblique(self, module, scene, angle):
        config, textures, subset, segs = scene
        frame, _, _ = self._run_frame(module, scene, 0.0, 0.0, angle)
        ref = render_frame(0.0, 0.0, angle, segs, config, textures=textures)
        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.96, max_err=0.30
        )

    @pytest.mark.parametrize(
        "px,py,angle",
        [
            (3.0, 2.0, 20),
            (-2.0, 3.0, 240),
            (1.0, -3.0, 50),
        ],
    )
    def test_frame_matches_reference_off_center(self, module, scene, px, py, angle):
        config, textures, subset, segs = scene
        frame, _, _ = self._run_frame(module, scene, px, py, angle)
        ref = render_frame(px, py, angle, segs, config, textures=textures)
        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.96, max_err=0.30
        )

    # ── Multi-frame stability ─────────────────────────────────────

    def test_multi_frame_stability(self, module, scene):
        """Walls visible across 10 frames of walk-and-turn."""
        config, textures, subset, segs = scene
        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        ceil = np.array(config.ceiling_color)
        floor = np.array(config.floor_color)
        H, W = config.screen_height, config.screen_width

        actions = [
            PlayerInput(),
            PlayerInput(forward=True),
            PlayerInput(forward=True),
            PlayerInput(turn_right=True),
            PlayerInput(forward=True),
            PlayerInput(turn_right=True),
            PlayerInput(forward=True),
            PlayerInput(turn_left=True),
            PlayerInput(forward=True),
            PlayerInput(),
        ]

        for i, inp in enumerate(actions):
            trace = FrameTrace()
            frame, state = step_frame(
                module, state, inp, subset, config, textures=textures, trace=trace
            )

            wall_pixels = 0
            for y in range(H):
                for x in range(W):
                    pix = frame[y, x]
                    is_ceil = np.allclose(pix, ceil, atol=0.05)
                    is_floor = np.allclose(pix, floor, atol=0.05)
                    if not is_ceil and not is_floor:
                        wall_pixels += 1

            fraction = wall_pixels / (H * W)
            assert fraction > 0.05, (
                f"frame {i}: only {fraction:.1%} wall pixels (expected >5%)"
            )
