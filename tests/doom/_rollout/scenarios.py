"""Scenario list for rollout-based DOOM tests.

Each scenario is one (player state, input) pair.  The rollout fixture
in ``test_rollout.py`` runs ``step_frame`` once per scenario and shares
the resulting :class:`Rollout` across every assertion test.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Scenario:
    label: str
    px: float
    py: float
    angle: int
    inputs: Dict[str, bool] = field(default_factory=dict)


SCENARIOS = [
    # Center, facing east (angle 0), no movement: every wall renderable
    # but no collision firing.
    Scenario(label="center_east_idle", px=0.0, py=0.0, angle=0),
    # Center, oblique angle (45°): tests off-axis projection — a
    # different sort order than the cardinal scenarios.
    Scenario(label="center_oblique_45", px=0.0, py=0.0, angle=45),
    # Up against the east wall, moving forward into it: HIT_FULL/HIT_X
    # fire on wall 0 (east wall).
    Scenario(
        label="east_wall_collide",
        px=4.9,
        py=0.0,
        angle=0,
        inputs={"forward": True},
    ),
    # Off-center, oblique: stresses the resolved-state and per-wall
    # geometry path with a non-symmetric scene.
    Scenario(label="off_center_3_2_20", px=3.0, py=2.0, angle=20),
    # Strafing into the east wall: HIT_X fires via the strafe-axis ray,
    # not the forward ray.
    Scenario(
        label="strafe_into_east",
        px=4.9,
        py=0.0,
        angle=64,
        inputs={"strafe_right": True},
    ),
]
