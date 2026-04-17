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
        screen_width=16,
        screen_height=16,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


_MAX_BSP_NODES = 4


@pytest.fixture(scope="module")
def wall_collision_module():
    """Compile collision flags and is_renderable from build_wall in one module.

    Output layout: [hit_full, hit_x, hit_y, is_renderable].

    BSP inputs are passed in but their contribution doesn't reach the
    collision outputs, so they won't actually be surfaced as ancestors
    of the compiled subgraph.
    """
    pos = create_pos_encoding()

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
    wall_bsp_coeffs = create_input("wall_bsp_coeffs", _MAX_BSP_NODES)
    wall_bsp_const = create_input("wall_bsp_const", 1)
    side_P_vec = create_input("side_P_vec", _MAX_BSP_NODES)

    outputs = build_wall(
        WallInputs(
            wall_ax=wall_ax,
            wall_ay=wall_ay,
            wall_bx=wall_bx,
            wall_by=wall_by,
            wall_tex_id=wall_tex_id,
            wall_index=wall_index,
            player_x=player_x,
            player_y=player_y,
            is_wall=is_wall,
            vel_dx=vel_dx,
            vel_dy=vel_dy,
            move_cos=move_cos,
            move_sin=move_sin,
            wall_bsp_coeffs=wall_bsp_coeffs,
            wall_bsp_const=wall_bsp_const,
            side_P_vec=side_P_vec,
        ),
        config=_tiny_config(),
        max_walls=_MAX_WALLS,
        max_coord=_MAX_COORD,
        max_bsp_nodes=_MAX_BSP_NODES,
    )
    out = Concatenate(
        [
            outputs.collision.hit_full,
            outputs.collision.hit_x,
            outputs.collision.hit_y,
            outputs.is_renderable,
        ]
    )
    return compile_headless(
        out,
        pos,
        d=1024,
        d_head=16,
        max_layers=50,
        verbose=False,
    )


@pytest.fixture(scope="module")
def wall_sort_value_module():
    """Compile the packed ``sort_value`` output of build_wall.

    Used to answer the angle-192 diagnostic question: does
    ``pack_wall_payload(wall_ax, wall_ay, wall_bx, wall_by, ...)``
    emit clean values at the WALL token's residual-stream slot, or
    does the WALL stage drift ax/by relative to their host-fed inputs?
    """
    pos = create_pos_encoding()

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
    wall_bsp_coeffs = create_input("wall_bsp_coeffs", _MAX_BSP_NODES)
    wall_bsp_const = create_input("wall_bsp_const", 1)
    side_P_vec = create_input("side_P_vec", _MAX_BSP_NODES)

    outputs = build_wall(
        WallInputs(
            wall_ax=wall_ax,
            wall_ay=wall_ay,
            wall_bx=wall_bx,
            wall_by=wall_by,
            wall_tex_id=wall_tex_id,
            wall_index=wall_index,
            player_x=player_x,
            player_y=player_y,
            is_wall=is_wall,
            vel_dx=vel_dx,
            vel_dy=vel_dy,
            move_cos=move_cos,
            move_sin=move_sin,
            wall_bsp_coeffs=wall_bsp_coeffs,
            wall_bsp_const=wall_bsp_const,
            side_P_vec=side_P_vec,
        ),
        config=_tiny_config(),
        max_walls=_MAX_WALLS,
        max_coord=_MAX_COORD,
        max_bsp_nodes=_MAX_BSP_NODES,
    )
    return compile_headless(
        outputs.sort_value,
        pos,
        d=1024,
        d_head=32,
        max_layers=80,
        verbose=False,
    )


def _pack(module, values: dict) -> torch.Tensor:
    """Pack ``values`` into the tensor layout the compiled ``module`` expects.

    compile_headless only surfaces inputs that are ancestors of the output
    node — so the shape depends on which outputs we compiled.  Reading
    ``module._input_specs`` gives us the canonical order.  Inputs not
    present in ``values`` default to zero.
    """
    d_input = max(s + w for _, s, w in module._input_specs)
    row = torch.zeros(1, d_input, dtype=torch.float32)
    for name, start, width in module._input_specs:
        if name not in values:
            continue
        row[0, start : start + width] = torch.tensor(
            values[name],
            dtype=torch.float32,
        ).reshape(width)
    return row


# ---------------------------------------------------------------------------
# Collision-flag behavior
# ---------------------------------------------------------------------------


# Velocities must stay within VEL_BP = [-0.7, 0.7] — the compiled graph's
# domain is per-frame movement (move_speed ≈ 0.3), not arbitrary rays.
@pytest.mark.parametrize(
    "scenario",
    [
        # Wall right in front, within one frame's reach.
        dict(
            name="head_on_hit",
            px=0.0,
            py=0.0,
            vel_dx=0.3,
            vel_dy=0.0,
            ax=0.2,
            ay=-1.0,
            bx=0.2,
            by=1.0,
            # t = 0.2/0.3 = 0.67 (hit); u = 0.5 (hit).
            expect_full=True,
            expect_x=True,
            expect_y=False,
        ),
        # Wall further than the velocity reaches: miss.
        dict(
            name="miss_too_far",
            px=0.0,
            py=0.0,
            vel_dx=0.3,
            vel_dy=0.0,
            ax=5.0,
            ay=-1.0,
            bx=5.0,
            by=1.0,
            expect_full=False,
            expect_x=False,
            expect_y=False,
        ),
        # Wall segment off to the side: u out of [0,1].
        dict(
            name="miss_wall_sideways",
            px=0.0,
            py=0.0,
            vel_dx=0.3,
            vel_dy=0.0,
            ax=0.2,
            ay=3.0,
            bx=0.2,
            by=5.0,
            expect_full=False,
            expect_x=False,
            expect_y=False,
        ),
        # Diagonal velocity through a vertical wall close in.
        dict(
            name="oblique_hit",
            px=0.0,
            py=0.0,
            vel_dx=0.3,
            vel_dy=0.3,
            ax=0.15,
            ay=-1.0,
            bx=0.15,
            by=1.0,
            # den = 0.6; num_t = 0.3; t = 0.5.  num_u = 0.345; u = 0.575 (hit).
            expect_full=True,
            expect_x=True,
            expect_y=False,
        ),
        # Moving straight back: shouldn't hit a wall in front.
        dict(
            name="backward_no_hit",
            px=0.0,
            py=0.0,
            vel_dx=-0.3,
            vel_dy=0.0,
            ax=0.2,
            ay=-1.0,
            bx=0.2,
            by=1.0,
            expect_full=False,
            expect_x=False,
            expect_y=False,
        ),
    ],
)
def test_collision_flags_match_reference(wall_collision_module, scenario):
    """Each (hit_full, hit_x, hit_y) flag must agree with reference _ray_hits_segment."""
    seg = Segment(
        ax=scenario["ax"],
        ay=scenario["ay"],
        bx=scenario["bx"],
        by=scenario["by"],
        color=(1, 0, 0),
    )
    ref_full = _ray_hits_segment(
        scenario["px"],
        scenario["py"],
        scenario["vel_dx"],
        scenario["vel_dy"],
        seg,
    )
    ref_x = _ray_hits_segment(
        scenario["px"],
        scenario["py"],
        scenario["vel_dx"],
        0.0,
        seg,
    )
    ref_y = _ray_hits_segment(
        scenario["px"],
        scenario["py"],
        0.0,
        scenario["vel_dy"],
        seg,
    )
    # Sanity: the parametrized "expect_*" flags match the reference.
    assert ref_full == scenario["expect_full"]
    assert ref_x == scenario["expect_x"]
    assert ref_y == scenario["expect_y"]

    inputs = _pack(
        wall_collision_module,
        {
            "is_wall": 1.0,
            "player_x": scenario["px"],
            "player_y": scenario["py"],
            "vel_dx": scenario["vel_dx"],
            "vel_dy": scenario["vel_dy"],
            "wall_ax": scenario["ax"],
            "wall_ay": scenario["ay"],
            "wall_bx": scenario["bx"],
            "wall_by": scenario["by"],
        },
    )
    with torch.no_grad():
        out = wall_collision_module(inputs)[0]
    hit_full = out[0].item() > 0.0
    hit_x = out[1].item() > 0.0
    hit_y = out[2].item() > 0.0

    assert (
        hit_full == ref_full
    ), f"{scenario['name']}: hit_full={hit_full} but reference={ref_full}"
    assert hit_x == ref_x, f"{scenario['name']}: hit_x={hit_x} but reference={ref_x}"
    assert hit_y == ref_y, f"{scenario['name']}: hit_y={hit_y} but reference={ref_y}"


def test_is_renderable_output(wall_collision_module):
    """``is_renderable`` is +1 for a head-on wall, -1 for a parallel wall,
    and -1 at non-WALL token positions.
    """
    # Head-on wall (vertical wall in front, player facing +x).
    head_on = _pack(
        wall_collision_module,
        {
            "is_wall": 1.0,
            "player_x": 0.0,
            "player_y": 0.0,
            "move_cos": 1.0,
            "move_sin": 0.0,
            "vel_dx": 0.0,
            "vel_dy": 0.0,
            "wall_ax": 2.0,
            "wall_ay": -1.0,
            "wall_bx": 2.0,
            "wall_by": 1.0,
            "wall_tex_id": 0.0,
            "wall_index": 0.0,
            "wall_bsp_coeffs": [0.0] * _MAX_BSP_NODES,
            "wall_bsp_const": 0.0,
            "side_P_vec": [0.0] * _MAX_BSP_NODES,
        },
    )
    with torch.no_grad():
        out = wall_collision_module(head_on)[0]
    assert (
        out[3].item() > 0.5
    ), f"head-on wall should be renderable, got {out[3].item():+.3f}"

    # Parallel wall (horizontal wall in front, player facing +x).
    # Wall runs along x-axis in front of player; sort_den ≈ 0.
    parallel = _pack(
        wall_collision_module,
        {
            "is_wall": 1.0,
            "player_x": 0.0,
            "player_y": 0.0,
            "move_cos": 1.0,
            "move_sin": 0.0,
            "vel_dx": 0.0,
            "vel_dy": 0.0,
            "wall_ax": 2.0,
            "wall_ay": 2.0,
            "wall_bx": 4.0,
            "wall_by": 2.0,
            "wall_tex_id": 0.0,
            "wall_index": 0.0,
            "wall_bsp_coeffs": [0.0] * _MAX_BSP_NODES,
            "wall_bsp_const": 0.0,
            "side_P_vec": [0.0] * _MAX_BSP_NODES,
        },
    )
    with torch.no_grad():
        out = wall_collision_module(parallel)[0]
    assert (
        out[3].item() < -0.5
    ), f"parallel wall should be non-renderable, got {out[3].item():+.3f}"

    # Non-WALL token position (is_wall=0).
    non_wall = _pack(
        wall_collision_module,
        {
            "is_wall": 0.0,
            "player_x": 0.0,
            "player_y": 0.0,
            "move_cos": 1.0,
            "move_sin": 0.0,
            "vel_dx": 0.0,
            "vel_dy": 0.0,
            "wall_ax": 2.0,
            "wall_ay": -1.0,
            "wall_bx": 2.0,
            "wall_by": 1.0,
            "wall_tex_id": 0.0,
            "wall_index": 0.0,
            "wall_bsp_coeffs": [0.0] * _MAX_BSP_NODES,
            "wall_bsp_const": 0.0,
            "side_P_vec": [0.0] * _MAX_BSP_NODES,
        },
    )
    with torch.no_grad():
        out = wall_collision_module(non_wall)[0]
    assert (
        out[3].item() < -0.5
    ), f"non-WALL position should be non-renderable, got {out[3].item():+.3f}"


def test_non_wall_positions_always_miss(wall_collision_module):
    """With is_wall=0, all three hit flags should be gated to -1 (miss)."""
    inputs = _pack(
        wall_collision_module,
        {
            "is_wall": 0.0,
            "player_x": 0.0,
            "player_y": 0.0,
            "vel_dx": 0.3,
            "vel_dy": 0.0,
            "wall_ax": 0.2,
            "wall_ay": -1.0,
            "wall_bx": 0.2,
            "wall_by": 1.0,
        },
    )
    with torch.no_grad():
        out = wall_collision_module(inputs)[0]
    for i, name in enumerate(("hit_full", "hit_x", "hit_y")):
        assert out[i].item() < 0.0, (
            f"{name} should be gated to miss at non-WALL positions, "
            f"got {out[i].item():+.2f}"
        )


# ---------------------------------------------------------------------------
# Angle-192 diagnostic — does WALL's sort_value drift ax/by for the north wall?
# ---------------------------------------------------------------------------


def test_sort_value_north_wall_clean_at_angle_192(wall_sort_value_module):
    """Boiled-down reproduction of the angle-192 render bug.

    At angle=192 (facing south), the observed compiled output at
    ``sort[2]`` shows the north wall payload as
    ``[4.86, 5.00, -5.00, 4.86]`` instead of the expected
    ``[5.00, 5.00, -5.00, 5.00]``.  That's a 0.14 drift on ax and by
    (both +5) but not on ay (+5) or bx (-5).  Because the SORTED-stage
    argmin was cleared by the isolated unit test
    (``test_angle_192_sentinel_ties_clean_pick``), the drift must
    originate at or before the WALL stage.

    This test compiles just ``build_wall``'s ``sort_value`` output,
    feeds the north wall's inputs with move_cos=0 / move_sin=-1
    (angle=192), and checks that wall_ax / wall_ay / wall_bx / wall_by
    read back clean — i.e., the WALL stage's pack_wall_payload doesn't
    introduce per-field drift.

    If this test PASSES, the drift is introduced downstream of WALL —
    most likely in how sort_value is stored in the residual stream and
    routed to the SORTED stage's attention.  If it FAILS, the WALL
    stage itself miscomputes one of these fields at angle=192.
    """
    # North wall of the synthetic box room: (5, 5) → (-5, 5), wall_index=1.
    # Player at origin; at DOOM angle=192 (facing south, -y direction),
    # cos(270°)=0, sin(270°)=-1.
    inputs = _pack(
        wall_sort_value_module,
        {
            "is_wall": 1.0,
            "player_x": 0.0,
            "player_y": 0.0,
            "move_cos": 0.0,
            "move_sin": -1.0,
            "vel_dx": 0.0,
            "vel_dy": 0.0,
            "wall_ax": 5.0,
            "wall_ay": 5.0,
            "wall_bx": -5.0,
            "wall_by": 5.0,
            "wall_tex_id": 1.0,
            "wall_index": 1.0,
            "wall_bsp_coeffs": [0.0] * _MAX_BSP_NODES,
            "wall_bsp_const": 0.0,
            "side_P_vec": [0.0] * _MAX_BSP_NODES,
        },
    )
    with torch.no_grad():
        out = wall_sort_value_module(inputs)[0]

    # sort_value layout (from wall_payload.pack_wall_payload):
    #   [0]: wall_ax
    #   [1]: wall_ay
    #   [2]: wall_bx
    #   [3]: wall_by
    #   [4]: wall_tex_id
    #   [5..9]: render precomp (sort_den, C, D, E, H_inv)
    #   [10]: bsp_rank
    #   [11..12]: visibility columns (vis_lo, vis_hi)
    #   [13..13+max_walls-1]: position_onehot
    got = out[:5].tolist()
    expected = [5.0, 5.0, -5.0, 5.0, 1.0]

    drifts = [abs(g - e) for g, e in zip(got, expected)]
    max_drift = max(drifts)
    assert max_drift < 0.05, (
        f"WALL-stage sort_value drift at angle=192, north wall:\n"
        f"  got      = {got}\n"
        f"  expected = {expected}\n"
        f"  per-field drift = {drifts}\n"
        f"If any of ax/by specifically drift (≈0.14) while ay/bx don't,\n"
        f"that matches the integration-test pattern and confirms the\n"
        f"WALL stage as the source."
    )
