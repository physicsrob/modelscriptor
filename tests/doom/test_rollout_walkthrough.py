"""Walkthrough-parameter rollout tests.

Same compile config as ``make walkthrough`` (d=3072, 120×100,
box_room_textured, max_coord=10.0, chunk_size=20) and the first three
pre-step states the wall-following controller produces from the default
start of (0, 0, 0).  ``render_pixels=False`` because the tier-2
assertions only need (col, start, length) — the texture coordinate
block is unused at tier 2 and skipping it makes the 120×100 frames
tractable to run in CI.

Split out from ``test_rollout.py`` so that the walkthrough fixture
(~163s class compile) runs in its own Modal shard rather than serially
behind ``TestRollout``.
"""

from typing import List

import pytest

from tests.doom._rollout.reference import compute_reference
from tests.doom._rollout.runner import Rollout, run_rollout
from tests.doom._rollout.scenarios import WALKTHROUGH_SCENARIOS, Scenario
from torchwright.doom.compile import compile_game
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


_TRIG = generate_trig_table()
_MOVE_SPEED = 0.3
_TURN_SPEED = 4

_WALKTHROUGH_D = 3072
_WALKTHROUGH_WIDTH = 120
_WALKTHROUGH_HEIGHT = 100
_WALKTHROUGH_FOV = 32
_WALKTHROUGH_TEX_SIZE = 64
_WALKTHROUGH_CHUNK_SIZE = 20
_WALKTHROUGH_MAX_COORD = 10.0


def _walkthrough_config():
    return RenderConfig(
        screen_width=_WALKTHROUGH_WIDTH,
        screen_height=_WALKTHROUGH_HEIGHT,
        fov_columns=_WALKTHROUGH_FOV,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


class TestRolloutWalkthrough:
    """Token-stream + per-RENDER overflow checks at walkthrough resolution."""

    @pytest.fixture(scope="class")
    def scene(self):
        config = _walkthrough_config()
        segs, textures = box_room_textured(
            wad_path="doom1.wad",
            tex_size=_WALKTHROUGH_TEX_SIZE,
        )
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs

    @pytest.fixture(scope="class")
    def module(self, scene):
        config, textures, subset, segs = scene
        return compile_game(
            config,
            textures,
            max_walls=max(8, len(segs)),
            max_coord=_WALKTHROUGH_MAX_COORD,
            d=_WALKTHROUGH_D,
            chunk_size=_WALKTHROUGH_CHUNK_SIZE,
            verbose=False,
            render_pixels=False,
        )

    @pytest.fixture(
        scope="class",
        params=WALKTHROUGH_SCENARIOS,
        ids=lambda s: s.label,
    )
    def scenario(self, request) -> Scenario:
        return request.param

    @pytest.fixture(scope="class")
    def rollout(self, module, scene, scenario) -> Rollout:
        config, _textures, subset, _segs = scene
        return run_rollout(
            module=module,
            px=scenario.px,
            py=scenario.py,
            angle=scenario.angle,
            inputs=PlayerInput(**scenario.inputs),
            subset=subset,
            config=config,
            move_speed=_MOVE_SPEED,
            turn_speed=_TURN_SPEED,
        )

    @pytest.fixture(scope="class")
    def reference(self, scenario, scene):
        config, _textures, subset, _segs = scene
        return compute_reference(
            px=scenario.px,
            py=scenario.py,
            angle=scenario.angle,
            inputs=PlayerInput(**scenario.inputs),
            subset=subset,
            config=config,
            move_speed=_MOVE_SPEED,
            turn_speed=_TURN_SPEED,
            chunk_size=_WALKTHROUGH_CHUNK_SIZE,
        )

    # ------------------------------------------------------------
    # Contract VALUEs (mirror TestRollout)
    # ------------------------------------------------------------

    def test_per_wall_contract_values(self, rollout, reference):
        mismatches = []
        for wall_i, ref in enumerate(reference.walls):
            comp_bsp = rollout.per_wall_value(wall_i, "BSP_RANK")
            if abs(comp_bsp - ref.bsp_rank) > 0.5:
                mismatches.append(
                    f"wall {wall_i} BSP_RANK: ref={ref.bsp_rank}, compiled={comp_bsp:.3f}"
                )

            comp_rend = rollout.per_wall_bool(wall_i, "IS_RENDERABLE")
            if comp_rend != ref.is_renderable:
                mismatches.append(
                    f"wall {wall_i} IS_RENDERABLE: ref={ref.is_renderable}, "
                    f"compiled={comp_rend}"
                )

            if ref.is_renderable and ref.is_projected:
                comp_vlo = rollout.per_wall_value(wall_i, "VIS_LO")
                comp_vhi = rollout.per_wall_value(wall_i, "VIS_HI")
                if abs(comp_vlo - ref.vis_lo) > 2.5:
                    mismatches.append(
                        f"wall {wall_i} VIS_LO: ref={ref.vis_lo}, "
                        f"compiled={comp_vlo:.3f}"
                    )
                if abs(comp_vhi - ref.vis_hi) > 2.5:
                    mismatches.append(
                        f"wall {wall_i} VIS_HI: ref={ref.vis_hi}, "
                        f"compiled={comp_vhi:.3f}"
                    )

            comp_hf = rollout.per_wall_bool(wall_i, "HIT_FULL")
            comp_hx = rollout.per_wall_bool(wall_i, "HIT_X")
            comp_hy = rollout.per_wall_bool(wall_i, "HIT_Y")
            for name, ref_b, comp_b in [
                ("HIT_FULL", ref.hit_full, comp_hf),
                ("HIT_X", ref.hit_x, comp_hx),
                ("HIT_Y", ref.hit_y, comp_hy),
            ]:
                if ref_b != comp_b:
                    mismatches.append(
                        f"wall {wall_i} {name}: ref={ref_b}, compiled={comp_b}"
                    )
        assert not mismatches, "\n".join(mismatches)

    def test_resolved_state(self, rollout, reference):
        rx = rollout.resolved_value("RESOLVED_X")
        ry = rollout.resolved_value("RESOLVED_Y")
        ra = rollout.resolved_value("RESOLVED_ANGLE")
        assert rx == pytest.approx(reference.resolved_x, abs=0.15)
        assert ry == pytest.approx(reference.resolved_y, abs=0.15)
        assert ra == pytest.approx(reference.resolved_angle, abs=1.5)

    def test_sort_order(self, rollout, reference):
        compiled = rollout.sort_result_wall_indices()
        n = len(reference.sort_order)
        assert len(compiled) >= n, (
            f"sort_order: compiled has {len(compiled)} slots, "
            f"reference expects at least {n}"
        )
        assert compiled[:n] == reference.sort_order, (
            f"sort_order mismatch: compiled[:{n}]={compiled[:n]}, "
            f"reference={reference.sort_order}"
        )

    # ------------------------------------------------------------
    # Per-RENDER overflow (the tier-2 check the walkthrough symptom
    # would surface in: a wall_top_y / wall_bottom_y miscompute appears
    # here as a (start, length) mismatch on the affected columns).
    # ------------------------------------------------------------

    def test_render_chunk_per_column(self, rollout, reference):
        """Each RENDER token's (col, start, length) overflow matches
        the reference's chunk-fill arithmetic.

        Compares per sort slot.  Tolerances:
          * ±1 column on ``col`` (same glancing-angle ±2 rationale as
            VIS_LO/HI, halved because we're matching position-wise
            rather than range-wise).
          * ±1 row on ``start`` and ``length`` to absorb
            quantization-step rounding (the graph carries these as
            float residuals and the host int(round(...))s them).
        """
        actual_per_slot = rollout.render_steps_per_sort_slot()
        mismatches: List[str] = []
        for slot, wall_i in enumerate(reference.sort_order):
            wall_ref = reference.walls[wall_i]
            if not wall_ref.is_projected:
                continue
            expected = reference.render_tokens_per_sort_slot[slot]
            assert slot < len(actual_per_slot), (
                f"slot {slot}: compiled emitted no RENDER tokens "
                f"(reference expected {len(expected)})"
            )
            actual = actual_per_slot[slot]
            # Filter expected to length>0 — trace.render_steps drops
            # length-zero RENDERs (their bitblit is a no-op).
            expected_nonzero = [e for e in expected if e.length > 0]
            if abs(len(actual) - len(expected_nonzero)) > 2:
                mismatches.append(
                    f"slot {slot} (wall {wall_i}): compiled emitted "
                    f"{len(actual)} non-zero RENDERs, "
                    f"reference expected {len(expected_nonzero)}"
                )
                continue
            n = min(len(actual), len(expected_nonzero))
            for i in range(n):
                a = actual[i]
                e = expected_nonzero[i]
                if abs(a.col - e.col) > 1:
                    mismatches.append(
                        f"slot {slot} (wall {wall_i}) render {i}: "
                        f"col ref={e.col}, compiled={a.col}"
                    )
                if abs(a.start - e.start) > 1:
                    mismatches.append(
                        f"slot {slot} (wall {wall_i}) render {i} col {a.col}: "
                        f"start ref={e.start}, compiled={a.start}"
                    )
                if abs(a.length - e.length) > 1:
                    mismatches.append(
                        f"slot {slot} (wall {wall_i}) render {i} col {a.col}: "
                        f"length ref={e.length}, compiled={a.length}"
                    )
        assert not mismatches, "\n".join(mismatches)
