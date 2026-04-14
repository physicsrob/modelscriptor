"""Unit tests for the WALL stage (torchwright.doom.stages.wall).

These tests compile a subgraph rooted at ``build_wall``'s outputs and
exercise the per-WALL collision + sort-score computations in isolation.
Expected behavior is checked against the reference math in
``torchwright.reference_renderer.collision`` — not against observed
output — so discrepancies flagged here are real bugs in the graph.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.collision import _ray_hits_segment
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

from torchwright.doom.stages.wall import WallInputs, build_wall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MAX_COORD = 20.0
_MAX_WALLS = 4


def _tiny_config() -> RenderConfig:
    # Small screen to keep compile fast; collision doesn't depend on W/H.
    return RenderConfig(
        screen_width=16, screen_height=16, fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


@pytest.fixture(scope="module")
def wall_collision_module():
    """Compile just the collision-flag outputs of build_wall."""
    pos = create_pos_encoding()

    # Same names build_wall expects via WallInputs.
    is_wall = create_input("is_wall", 1)
    move_cos = create_input("move_cos", 1)
    move_sin = create_input("move_sin", 1)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    vel_dx = create_input("vel_dx", 1)
    vel_dy = create_input("vel_dy", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    wall_index = create_input("wall_index", 1)
    wall_tex_id = create_input("wall_tex_id", 1)

    outputs = build_wall(
        WallInputs(
            wall_ax=wall_ax, wall_ay=wall_ay,
            wall_bx=wall_bx, wall_by=wall_by,
            wall_tex_id=wall_tex_id, wall_index=wall_index,
            player_x=player_x, player_y=player_y,
            is_wall=is_wall,
            vel_dx=vel_dx, vel_dy=vel_dy,
            move_cos=move_cos, move_sin=move_sin,
        ),
        config=_tiny_config(),
        max_walls=_MAX_WALLS,
        max_coord=_MAX_COORD,
    )
    out = Concatenate([
        outputs.collision.hit_full,
        outputs.collision.hit_x,
        outputs.collision.hit_y,
    ])
    return compile_headless(
        out, pos, d=1024, d_head=16, max_layers=50, verbose=False,
    )


def _pack(module, values: dict) -> torch.Tensor:
    """Pack ``values`` into the tensor layout the compiled ``module`` expects.

    compile_headless only surfaces inputs that are ancestors of the output
    node — so the shape depends on which outputs we compiled.  Reading
    ``module._input_specs`` gives us the canonical order.
    """
    d_input = max(s + w for _, s, w in module._input_specs)
    row = torch.zeros(1, d_input, dtype=torch.float32)
    for name, start, width in module._input_specs:
        row[0, start:start + width] = torch.tensor(
            values[name], dtype=torch.float32,
        ).reshape(width)
    return row


# ---------------------------------------------------------------------------
# Collision-flag behavior
# ---------------------------------------------------------------------------

# Velocities must stay within VEL_BP = [-0.7, 0.7] — the compiled graph's
# domain is per-frame movement (move_speed ≈ 0.3), not arbitrary rays.
@pytest.mark.parametrize("scenario", [
    # Wall right in front, within one frame's reach.
    dict(
        name="head_on_hit",
        px=0.0, py=0.0, vel_dx=0.3, vel_dy=0.0,
        ax=0.2, ay=-1.0, bx=0.2, by=1.0,
        # t = 0.2/0.3 = 0.67 (hit); u = 0.5 (hit).
        expect_full=True, expect_x=True, expect_y=False,
    ),
    # Wall further than the velocity reaches: miss.
    dict(
        name="miss_too_far",
        px=0.0, py=0.0, vel_dx=0.3, vel_dy=0.0,
        ax=5.0, ay=-1.0, bx=5.0, by=1.0,
        expect_full=False, expect_x=False, expect_y=False,
    ),
    # Wall segment off to the side: u out of [0,1].
    dict(
        name="miss_wall_sideways",
        px=0.0, py=0.0, vel_dx=0.3, vel_dy=0.0,
        ax=0.2, ay=3.0, bx=0.2, by=5.0,
        expect_full=False, expect_x=False, expect_y=False,
    ),
    # Diagonal velocity through a vertical wall close in.
    dict(
        name="oblique_hit",
        px=0.0, py=0.0, vel_dx=0.3, vel_dy=0.3,
        ax=0.15, ay=-1.0, bx=0.15, by=1.0,
        # den = 0.6; num_t = 0.3; t = 0.5.  num_u = 0.345; u = 0.575 (hit).
        expect_full=True, expect_x=True, expect_y=False,
    ),
    # Moving straight back: shouldn't hit a wall in front.
    dict(
        name="backward_no_hit",
        px=0.0, py=0.0, vel_dx=-0.3, vel_dy=0.0,
        ax=0.2, ay=-1.0, bx=0.2, by=1.0,
        expect_full=False, expect_x=False, expect_y=False,
    ),
])
def test_collision_flags_match_reference(wall_collision_module, scenario):
    """Each (hit_full, hit_x, hit_y) flag must agree with reference _ray_hits_segment."""
    seg = Segment(
        ax=scenario["ax"], ay=scenario["ay"],
        bx=scenario["bx"], by=scenario["by"],
        color=(1, 0, 0),
    )
    ref_full = _ray_hits_segment(
        scenario["px"], scenario["py"],
        scenario["vel_dx"], scenario["vel_dy"], seg,
    )
    ref_x = _ray_hits_segment(
        scenario["px"], scenario["py"],
        scenario["vel_dx"], 0.0, seg,
    )
    ref_y = _ray_hits_segment(
        scenario["px"], scenario["py"],
        0.0, scenario["vel_dy"], seg,
    )
    # Sanity: the parametrized "expect_*" flags match the reference.
    assert ref_full == scenario["expect_full"]
    assert ref_x == scenario["expect_x"]
    assert ref_y == scenario["expect_y"]

    inputs = _pack(wall_collision_module, {
        "is_wall": 1.0,
        "player_x": scenario["px"], "player_y": scenario["py"],
        "vel_dx": scenario["vel_dx"], "vel_dy": scenario["vel_dy"],
        "wall_ax": scenario["ax"], "wall_ay": scenario["ay"],
        "wall_bx": scenario["bx"], "wall_by": scenario["by"],
    })
    with torch.no_grad():
        out = wall_collision_module(inputs)[0]
    hit_full = out[0].item() > 0.0
    hit_x = out[1].item() > 0.0
    hit_y = out[2].item() > 0.0

    assert hit_full == ref_full, (
        f"{scenario['name']}: hit_full={hit_full} but reference={ref_full}"
    )
    assert hit_x == ref_x, (
        f"{scenario['name']}: hit_x={hit_x} but reference={ref_x}"
    )
    assert hit_y == ref_y, (
        f"{scenario['name']}: hit_y={hit_y} but reference={ref_y}"
    )


def test_non_wall_positions_always_miss(wall_collision_module):
    """With is_wall=0, all three hit flags should be gated to -1 (miss)."""
    inputs = _pack(wall_collision_module, {
        "is_wall": 0.0,
        "player_x": 0.0, "player_y": 0.0,
        "vel_dx": 0.3, "vel_dy": 0.0,
        "wall_ax": 0.2, "wall_ay": -1.0,
        "wall_bx": 0.2, "wall_by": 1.0,
    })
    with torch.no_grad():
        out = wall_collision_module(inputs)[0]
    for i, name in enumerate(("hit_full", "hit_x", "hit_y")):
        assert out[i].item() < 0.0, (
            f"{name} should be gated to miss at non-WALL positions, "
            f"got {out[i].item():+.2f}"
        )
