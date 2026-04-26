"""Rollout-based DOOM tests.

Replaces the slow-and-coarse pixel-comparison ``test_pipeline.py`` with
an introspective check on the full autoregressive token stream.  The
graph is compiled once with ``render_pixels=False`` (no texture
attention, no chunked-fill cost), one rollout per scenario is captured
into a :class:`Rollout`, and each assertion test reads the structured
view through that handle.

Per scenario we assert:

* The token sequence is structurally well-formed — markers and
  identifiers in the right order, plus the SORT_RESULT / RENDER /
  DONE pattern in the post-thinking stream.
* Every contract VALUE — those consumed by another stage — matches
  a Python reference: BSP_RANK, IS_RENDERABLE, VIS_LO, VIS_HI,
  HIT_FULL, HIT_X, HIT_Y per wall, plus RESOLVED_X/Y/ANGLE and
  per-sort-position SORT_RESULT.
* The number of RENDER tokens per sort position matches
  ``vis_hi - vis_lo + 1`` (the reference's column-iteration count).

Internal thinking-VALUE intermediates (CROSS_*, DOT_*, T_STAR_*,
T_LO/HI, COL_A/B) are NOT numerically referenced — they're private
to the line-clip factoring and the assert_in_range nodes already
guard their bounds.
"""

from typing import List

import pytest

from tests.doom._rollout.reference import compute_reference
from tests.doom._rollout.runner import Rollout, run_rollout
from tests.doom._rollout.scenarios import SCENARIOS, Scenario
from torchwright.doom.compile import compile_game
from torchwright.doom.embedding import IDENTIFIER_NAMES, vocab_id
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


_TRIG = generate_trig_table()
_MOVE_SPEED = 0.3
_TURN_SPEED = 4
_MAX_WALLS = 8

_STEPS_PER_WALL = 35  # 1 marker + 17 (id, value) pairs


def _box_room_config():
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room_segments(half=5.0):
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


class TestRollout:
    """Token-stream tests at headless 16×20 resolution."""

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
            max_walls=_MAX_WALLS,
            d=2048,
            d_head=32,
            verbose=False,
            render_pixels=False,
        )

    @pytest.fixture(
        scope="class",
        params=SCENARIOS,
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
        )

    # ------------------------------------------------------------
    # Structural
    # ------------------------------------------------------------

    def test_token_structure(self, rollout, reference):
        """Markers, identifier IDs, RESOLVED block, and SORTED hand-off
        sit at their fixed offsets."""
        log = rollout.token_id_log
        expected_min_len = _MAX_WALLS * _STEPS_PER_WALL + 7
        assert len(log) >= expected_min_len, (
            f"token_id_log length {len(log)} < expected_min {expected_min_len}"
        )

        for wall_i in range(_MAX_WALLS):
            base = wall_i * _STEPS_PER_WALL
            assert log[base] == vocab_id(f"THINKING_WALL_{wall_i}"), (
                f"wall {wall_i}: marker at pos {base} got "
                f"{log[base]} (expected {vocab_id(f'THINKING_WALL_{wall_i}')})"
            )
            for slot, name in enumerate(IDENTIFIER_NAMES[:17]):
                pos = base + 1 + 2 * slot
                assert log[pos] == vocab_id(name), (
                    f"wall {wall_i} slot {slot} ({name}): pos {pos} got "
                    f"{log[pos]} (expected {vocab_id(name)})"
                )

        resolved_base = _MAX_WALLS * _STEPS_PER_WALL
        assert log[resolved_base + 0] == vocab_id("RESOLVED_X")
        assert log[resolved_base + 2] == vocab_id("RESOLVED_Y")
        assert log[resolved_base + 4] == vocab_id("RESOLVED_ANGLE")
        assert log[resolved_base + 6] == vocab_id("SORTED_WALL")

    def test_per_wall_contract_values(self, rollout, reference):
        """Per-wall BSP_RANK, IS_RENDERABLE, VIS_LO/HI, HIT_*."""
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

            # VIS_LO/HI: only compare when project_wall produced a
            # projection.  When the graph thinks a wall is renderable
            # (broader central-ray gate) but no per-column ray hits,
            # project_wall returns None and the graph's vis values
            # come from FOV-clipping the wall plane — we have no
            # independent reference for those.
            if ref.is_renderable and ref.is_projected:
                comp_vlo = rollout.per_wall_value(wall_i, "VIS_LO")
                comp_vhi = rollout.per_wall_value(wall_i, "VIS_HI")
                # ±2-column tolerance at glancing angles.
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
        """Resolved x/y/angle from the RESOLVED tail tokens."""
        rx = rollout.resolved_value("RESOLVED_X")
        ry = rollout.resolved_value("RESOLVED_Y")
        ra = rollout.resolved_value("RESOLVED_ANGLE")
        assert rx == pytest.approx(reference.resolved_x, abs=0.15)
        assert ry == pytest.approx(reference.resolved_y, abs=0.15)
        assert ra == pytest.approx(reference.resolved_angle, abs=1.5)

    # ------------------------------------------------------------
    # SORT + RENDER stream
    # ------------------------------------------------------------

    def test_sort_order(self, rollout, reference):
        """Each renderable wall is picked once, in BSP-rank ascending
        order.

        The graph emits garbage SORT_RESULT VALUEs after the sort
        exhausts (existing behavior — the soft-averaged attention picks
        nonsensical wall_indices).  We only check the first
        ``len(reference.sort_order)`` slots.
        """
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

    def test_render_count_per_wall(self, rollout, reference):
        """One RENDER per visible column per renderable wall.

        Skips slots whose wall has no ``project_wall`` projection — the
        graph still emits RENDERs for those (FOV-clipped vis range) but
        we lack an independent column-iteration reference.
        """
        per_slot = rollout.render_positions_per_sort_slot()
        for slot, wall_i in enumerate(reference.sort_order):
            wall_ref = reference.walls[wall_i]
            if not wall_ref.is_projected:
                continue
            assert slot < len(per_slot), (
                f"slot {slot}: compiled has only {len(per_slot)} slots, "
                f"reference expected at least {slot + 1}"
            )
            expected = reference.render_count_per_sort_slot[slot]
            actual = len(per_slot[slot])
            # Graph and reference can disagree by ±2 columns for the
            # same reason VIS_LO/HI can disagree (glancing-angle column
            # boundary).
            assert abs(actual - expected) <= 2, (
                f"slot {slot} (wall {wall_i}): compiled emitted "
                f"{actual} RENDERs, reference expected {expected}"
            )

