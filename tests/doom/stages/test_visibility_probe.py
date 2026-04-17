"""Probe ``_compute_visibility_columns`` on the exact scenes that
``test_game_graph.py`` marks as the residual failures.

Purpose: verify the hypothesis that the pixel-level disagreement at
``test_renders_oblique_angle[20]`` and
``test_renders_off_center_oblique[1.0,-3.0,50]`` traces back to
column drift in ``_compute_visibility_columns`` (frustum clip +
endpoint projection) rather than something downstream in the render
pipeline.

For each failing scene we reconstruct the exact WALL-stage input
the integration test produces — player state + one wall — compile the
visibility-column subgraph, and compare the compiled ``(vis_lo,
vis_hi)`` against an exact Python oracle that implements the same
frustum clip + ``atan(cross/dot)`` projection.

If the compiled columns match the oracle, the failing pixel is *not*
caused by the projection — look elsewhere (render attention, wall
selection, texture lookup).  If they drift, the drift magnitude
tells us how much of the pixel error is attributable to the
projection.
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

from torchwright.doom.stages.wall import _compute_visibility_columns

_MAX_COORD = 20.0
_W = 16
_H = 20
_FOV = 16


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=_W,
        screen_height=_H,
        fov_columns=_FOV,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


@pytest.fixture(scope="module")
def vis_module():
    """Compile just the visibility-columns subgraph.

    Wall geometry (``wall_ax``, ``wall_ay``, etc.) and player state
    (``player_x``, ``player_y``, ``move_cos``, ``move_sin``) are fed
    as 1-wide scalars.  ``is_renderable`` is set to +1 (valid) so the
    cond_gate passes through the raw columns.
    """
    pos = create_pos_encoding()
    wall_ax = create_input("wall_ax", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    wall_ay = create_input("wall_ay", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    wall_bx = create_input("wall_bx", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    wall_by = create_input("wall_by", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    player_x = create_input("player_x", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    player_y = create_input("player_y", 1, value_range=(-_MAX_COORD, _MAX_COORD))
    move_cos = create_input("move_cos", 1, value_range=(-1.0, 1.0))
    move_sin = create_input("move_sin", 1, value_range=(-1.0, 1.0))
    is_renderable = create_input("is_renderable", 1, value_range=(-1.0, 1.0))

    vis_lo, vis_hi = _compute_visibility_columns(
        wall_ax,
        wall_ay,
        wall_bx,
        wall_by,
        player_x,
        player_y,
        move_cos,
        move_sin,
        is_renderable,
        config=_config(),
        max_coord=_MAX_COORD,
    )
    output = Concatenate([vis_lo, vis_hi])
    return compile_headless(
        output,
        pos,
        d=1024,
        d_head=32,
        max_layers=60,
        verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start : start + width] = torch.tensor(
                row[name],
                dtype=torch.float32,
            ).reshape(width)
    return t


def _oracle_vis_columns(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    px: float,
    py: float,
    angle_idx: int,
) -> tuple[float, float]:
    """Exact Python implementation of ``_compute_visibility_columns``.

    Does the frustum clip against the two FOV half-planes, then
    projects clipped endpoints either via ``atan(cross/dot)`` (for
    endpoints still inside the cone) or to exactly ``col=0``/``col=W``
    (for endpoints snapped to an FOV boundary).  Matches the integer
    semantics of the compiled version: a DOOM-unit angle, ``trig[]``
    table lookup, and the same L/R boundary conventions as
    ``torchwright.doom.stages.sorted._compute_visibility_columns``.
    """
    trig = generate_trig_table()
    cos_p = float(trig[angle_idx % 256, 0])
    sin_p = float(trig[angle_idx % 256, 1])

    def rot(x: float, y: float) -> tuple[float, float]:
        dx = x - px
        dy = y - py
        cross = cos_p * dy - sin_p * dx
        dot = cos_p * dx + sin_p * dy
        return cross, dot

    cross_a, dot_a = rot(ax, ay)
    cross_b, dot_b = rot(bx, by)

    fov_rad = _FOV * math.pi / 128.0
    half_fov = fov_rad / 2.0
    sh = math.sin(half_fov)
    ch = math.cos(half_fov)

    def f_L(cr, dt):
        return sh * dt - ch * cr

    def f_R(cr, dt):
        return sh * dt + ch * cr

    fLa, fLb = f_L(cross_a, dot_a), f_L(cross_b, dot_b)
    fRa, fRb = f_R(cross_a, dot_a), f_R(cross_b, dot_b)

    def plane_contrib(fa, fb):
        denom = fa - fb
        if abs(denom) < 1e-12:
            # Parallel to plane — treat as no-clip for that plane.
            t_star = 0.5
        else:
            t_star = fa / denom
        t_lo_contrib = 0.0 if fa >= 0 else t_star
        t_hi_contrib = 1.0 if fb >= 0 else t_star
        return t_lo_contrib, t_hi_contrib

    t_lo_L, t_hi_L = plane_contrib(fLa, fLb)
    t_lo_R, t_hi_R = plane_contrib(fRa, fRb)

    t_lo = max(0.0, t_lo_L, t_lo_R)
    t_hi = min(1.0, t_hi_L, t_hi_R)

    W = float(_W)
    col_lo, col_hi = -2.0, W + 2.0
    half_W = W / 2.0
    col_scale = W / fov_rad

    def project(cr, dt):
        if dt > 0:
            col = math.atan(cr / dt) * col_scale + half_W
        elif dt < 0:
            col_front = math.atan(cr / (-dt)) * col_scale + half_W
            col = W - col_front
        else:
            col = half_W
        return max(col_lo, min(col_hi, col))

    # Endpoint A: clipped on L → col=W, clipped on R → col=0.
    a_inside_L = fLa >= 0
    a_inside_R = fRa >= 0
    a_clipped_on_L = t_lo_L > t_lo_R
    if a_inside_L and a_inside_R:
        col_A = project(cross_a, dot_a)
    elif a_clipped_on_L:
        col_A = W
    else:
        col_A = 0.0

    b_inside_L = fLb >= 0
    b_inside_R = fRb >= 0
    # Matches the code's ``compare(t_hi_R − t_hi_L, 0)``: B is clipped
    # on L iff the L plane exits before R (smaller t_hi_L).
    b_clipped_on_L = t_hi_L < t_hi_R
    if b_inside_L and b_inside_R:
        col_B = project(cross_b, dot_b)
    elif b_clipped_on_L:
        col_B = W
    else:
        col_B = 0.0

    if t_lo > t_hi:
        sentinel = W + 2.0
        return sentinel, sentinel

    return min(col_A, col_B), max(col_A, col_B)


# Failing scenes from ``test_game_graph.py`` (confirmed).  Each entry
# names a wall that's in-FOV at the given player pose — i.e., at
# least one endpoint projects via ``_endpoint_to_column`` rather than
# being frustum-snapped.
#
# Box-room walls:
#   east  = (5, -5) → (5, 5)     tex 0
#   west  = (-5, 5) → (-5, -5)   tex 1
#   north = (-5, 5) → (5, 5)     tex 2
#   south = (-5, -5) → (5, -5)   tex 3
_CASES = [
    # angle[20] — player (0, 0) facing 20.  Test fails at pixel
    # (row=9, col=0).  east wall covers full screen per sort log;
    # north wall is off-screen to the right.
    ("angle20_east", 0.0, 0.0, 20, (5.0, -5.0, 5.0, 5.0)),
    ("angle20_north", 0.0, 0.0, 20, (-5.0, 5.0, 5.0, 5.0)),
    # off_center[1.0,-3.0,50] — player (1, -3) facing 50.
    ("oc50_east", 1.0, -3.0, 50, (5.0, -5.0, 5.0, 5.0)),
    ("oc50_north", 1.0, -3.0, 50, (-5.0, 5.0, 5.0, 5.0)),
    ("oc50_west", 1.0, -3.0, 50, (-5.0, 5.0, -5.0, -5.0)),
    ("oc50_south", 1.0, -3.0, 50, (-5.0, -5.0, 5.0, -5.0)),
]


@pytest.mark.parametrize("name,px,py,angle,wall", _CASES)
def test_visibility_columns_match_oracle(vis_module, name, px, py, angle, wall):
    """Compiled ``(vis_lo, vis_hi)`` must match the Python oracle within
    0.3 columns — tight enough to catch the 1-col drift class but loose
    enough to tolerate the known piecewise_linear_2d pinv residual."""
    ax, ay, bx, by = wall
    trig = generate_trig_table()
    cos_p = float(trig[angle % 256, 0])
    sin_p = float(trig[angle % 256, 1])
    row = {
        "wall_ax": ax,
        "wall_ay": ay,
        "wall_bx": bx,
        "wall_by": by,
        "player_x": px,
        "player_y": py,
        "move_cos": cos_p,
        "move_sin": sin_p,
        "is_renderable": 1.0,
    }
    # compile_headless wants at least 2 positions for the sequence to be
    # non-degenerate; the probe reads from position 0.
    pad = {k: 0.0 for k in row}
    inputs = _pack(vis_module, [row, pad])
    with torch.no_grad():
        out = vis_module(inputs)
    vis_lo_got = out[0, 0].item()
    vis_hi_got = out[0, 1].item()

    vis_lo_ref, vis_hi_ref = _oracle_vis_columns(
        ax,
        ay,
        bx,
        by,
        px,
        py,
        angle,
    )

    lo_err = abs(vis_lo_got - vis_lo_ref)
    hi_err = abs(vis_hi_got - vis_hi_ref)
    msg = (
        f"{name}: wall=({ax},{ay})-({bx},{by}) player=({px},{py}) angle={angle}\n"
        f"  compiled: vis_lo={vis_lo_got:.3f}, vis_hi={vis_hi_got:.3f}\n"
        f"  oracle:   vis_lo={vis_lo_ref:.3f}, vis_hi={vis_hi_ref:.3f}\n"
        f"  drift:    lo={lo_err:.3f}, hi={hi_err:.3f}"
    )
    assert lo_err < 0.3 and hi_err < 0.3, msg
