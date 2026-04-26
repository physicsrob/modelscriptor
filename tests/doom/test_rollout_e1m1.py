"""E1M1 rollout fixture — exercise the host-side coord shift on a real map.

E1M1's wall coordinates run to thousands of units from the WAD origin —
well outside the graph's declared ``max_coord`` envelope.  ``MapSubset``
carries a ``scene_origin`` (set to the player spawn by
``load_map_subset``) and ``step_frame`` subtracts it from every wall /
player coord before feeding the graph, then adds it back when reading
``RESOLVED_X``/``RESOLVED_Y``.  Without that shift the
``piecewise_linear_2d`` ops would clamp every coord to ±max_coord and
the per-wall geometry would be unrecoverable.

Asserted contract (per the plan): BSP_RANK and IS_RENDERABLE per wall
agree with the Python reference.  RESOLVED_X/Y comparisons are skipped
— the rollout decoder returns shifted-frame coords while the reference
is in world coords, so any meaningful comparison requires re-shifting
on one side.  The two contract values that don't depend on world-coord
scale are sufficient to confirm the shift is wired correctly end to end.

Split out from ``test_rollout.py`` so the fixture's class-compile
runs in its own Modal shard.

Why ``max_walls=4`` rather than the usual 8: the wire-format
``CROSS_A``/``DOT_A`` clamp at ±40 in the player-rotated frame.  Walls
selected by the closest-N rule that sit further than ~40 units from
spawn (rotated into the player's facing direction) saturate that
clamp, dropping enough angular resolution to flip ``IS_RENDERABLE``
on the affected wall.  This is an inherent limit of the host-shift
strategy noted in the plan's "Risks" section — gameplay-relevant
walls (the closest ~4-6) sit comfortably inside ±40, so the test
selects 4 to stay safely inside that envelope.  Larger ``max_walls``
on E1M1 would surface false ``IS_RENDERABLE=True`` on saturated walls
that shouldn't be visible.
"""

from typing import List

import pytest

from tests.doom._rollout.reference import compute_reference
from tests.doom._rollout.runner import Rollout, run_rollout
from torchwright.doom.compile import compile_game
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import load_map_subset
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


_TRIG = generate_trig_table()
_MOVE_SPEED = 0.3
_TURN_SPEED = 4
_MAX_WALLS = 4

# E1M1 player-1 start position in DOOM map units.  Used both as the
# subset selector (which 8 walls are nearest) and the scene_origin for
# the host-side coord shift — the spawn-centred frame keeps the player
# and any nearby walls inside the ``max_coord``-sized envelope.
_E1M1_SPAWN_X = 1056.0
_E1M1_SPAWN_Y = -3616.0


def _e1m1_config() -> RenderConfig:
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


# Scenarios are in world coords (DOOM map units).  ``run_rollout`` /
# ``step_frame`` apply the subset's ``scene_origin`` shift internally;
# the reference reads world-coord segments and world-coord (px, py)
# directly, so both sides see consistent input.
_E1M1_SCENARIOS = [
    # Player at the exact spawn, facing north (the canonical E1M1
    # facing direction).  ``angle=64`` rotates the player's frame off
    # cardinal east so no wall coord lands on a sort_den ≈ 0 boundary
    # — angle=0 with this scene puts wall 0 right at the renderability
    # margin where compiled FP noise (1e-5) can flip the bool.
    pytest.param(
        (_E1M1_SPAWN_X, _E1M1_SPAWN_Y, 64, {}),
        id="spawn_facing_north",
    ),
    # A few units away from spawn within the same room — typical
    # gameplay pose, still inside the local envelope.
    pytest.param(
        (_E1M1_SPAWN_X + 32.0, _E1M1_SPAWN_Y - 8.0, 64, {}),
        id="near_spawn",
    ),
]


class TestE1M1Subset:
    """E1M1 host-shift contract: BSP_RANK + IS_RENDERABLE per wall."""

    @pytest.fixture(scope="class")
    def scene(self):
        config = _e1m1_config()
        subset = load_map_subset(
            wad_path="doom1.wad",
            map_name="E1M1",
            px=_E1M1_SPAWN_X,
            py=_E1M1_SPAWN_Y,
            max_walls=_MAX_WALLS,
        )
        return config, subset

    @pytest.fixture(scope="class")
    def module(self, scene):
        config, subset = scene
        return compile_game(
            config,
            subset.textures,
            max_walls=_MAX_WALLS,
            max_coord=100.0,
            d=2048,
            d_head=32,
            verbose=False,
            render_pixels=False,
        )

    @pytest.fixture(
        scope="class",
        params=_E1M1_SCENARIOS,
    )
    def scenario(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def rollout(self, module, scene, scenario) -> Rollout:
        config, subset = scene
        px, py, angle, inputs = scenario
        return run_rollout(
            module=module,
            px=px,
            py=py,
            angle=angle,
            inputs=PlayerInput(**inputs),
            subset=subset,
            config=config,
            move_speed=_MOVE_SPEED,
            turn_speed=_TURN_SPEED,
        )

    @pytest.fixture(scope="class")
    def reference(self, scenario, scene):
        config, subset = scene
        px, py, angle, inputs = scenario
        return compute_reference(
            px=px,
            py=py,
            angle=angle,
            inputs=PlayerInput(**inputs),
            subset=subset,
            config=config,
            move_speed=_MOVE_SPEED,
            turn_speed=_TURN_SPEED,
        )

    def test_bsp_rank_and_is_renderable(self, rollout, reference):
        """BSP_RANK and IS_RENDERABLE per wall agree with the reference.

        Both contract values are scale-invariant: BSP_RANK is a
        coefficient dot product over the BSP side decisions
        (integer-valued); IS_RENDERABLE is a sign / margin test on the
        central-ray sort_den.  Neither depends on the absolute
        magnitude of wall coords, so they hold across the host-side
        shift without further adjustment.
        """
        mismatches: List[str] = []
        for wall_i, ref in enumerate(reference.walls):
            comp_bsp = rollout.per_wall_value(wall_i, "BSP_RANK")
            if abs(comp_bsp - ref.bsp_rank) > 0.5:
                mismatches.append(
                    f"wall {wall_i} BSP_RANK: ref={ref.bsp_rank}, "
                    f"compiled={comp_bsp:.3f}"
                )
            comp_rend = rollout.per_wall_bool(wall_i, "IS_RENDERABLE")
            if comp_rend != ref.is_renderable:
                mismatches.append(
                    f"wall {wall_i} IS_RENDERABLE: ref={ref.is_renderable}, "
                    f"compiled={comp_rend}"
                )
        assert not mismatches, "\n".join(mismatches)
