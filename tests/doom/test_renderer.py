"""Unit tests for the DOOM renderer graph stages.

Tests graph logic via .compute() against reference renderer math.
"""

import numpy as np
import torch
import pytest

from torchwright.doom.renderer import (
    BIG_DISTANCE,
    _column_fill,
    _segment_distance,
    _segment_intersection,
    build_renderer_graph,
    trig_lookup,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer import (
    RenderConfig,
    Segment,
    generate_trig_table,
    render_frame,
)
from torchwright.reference_renderer.render import intersect_ray_segment


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def config(trig_table):
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=trig_table,
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


# ── Stage 1: Trig lookup ────────────────────────────────────────────


def test_trig_lookup(trig_table):
    """map_to_table trig lookup matches generate_trig_table for all 256 entries."""
    ray_angle = create_input("ray_angle", 1)
    ray_cos, ray_sin = trig_lookup(ray_angle)

    for i in range(256):
        vals = {"ray_angle": torch.tensor([[float(i)]])}
        cos_val = ray_cos.compute(1, vals).item()
        sin_val = ray_sin.compute(1, vals).item()
        assert abs(cos_val - trig_table[i, 0]) < 1e-4, f"cos mismatch at angle {i}"
        assert abs(sin_val - trig_table[i, 1]) < 1e-4, f"sin mismatch at angle {i}"


# ── Stage 3: Segment intersection math ──────────────────────────────


def test_intersection_values():
    """den, num_t, num_u match manual computation for a known case."""
    from torchwright.graph import Concatenate

    px = create_input("player_x", 1)
    py = create_input("player_y", 1)
    rc = create_input("ray_cos", 1)
    rs = create_input("ray_sin", 1)
    pxs = create_input("px_sin", 1)
    pyc = create_input("py_cos", 1)

    cos_sin = Concatenate([rc, rs])
    px_py = Concatenate([px, py])
    trig_and_products = Concatenate([rc, rs, pxs, pyc])

    seg = Segment(ax=5.0, ay=-10.0, bx=5.0, by=10.0, color=(1.0, 0.0, 0.0))
    den, num_t, num_u = _segment_intersection(cos_sin, px_py, trig_and_products, seg)

    # Player at origin, facing east (cos=1, sin=0)
    vals = {
        "player_x": torch.tensor([[0.0]]),
        "player_y": torch.tensor([[0.0]]),
        "ray_cos": torch.tensor([[1.0]]),
        "ray_sin": torch.tensor([[0.0]]),
        "px_sin": torch.tensor([[0.0]]),   # 0 * 0
        "py_cos": torch.tensor([[0.0]]),   # 0 * 1
    }

    # ex=0, ey=20
    # den = cos*ey - sin*ex = 1*20 - 0*0 = 20
    # num_t = (5*20 - (-10)*0) + 0*0 - 20*0 = 100
    # num_u = (5*0 - (-10)*1) + (0 - 0) = 10
    assert abs(den.compute(1, vals).item() - 20.0) < 1e-4
    assert abs(num_t.compute(1, vals).item() - 100.0) < 1e-4
    assert abs(num_u.compute(1, vals).item() - 10.0) < 1e-4

    # t = 100/20 = 5.0 — matches intersect_ray_segment
    hit = intersect_ray_segment(0.0, 0.0, 1.0, 0.0, seg)
    assert hit is not None
    ref_t, ref_u = hit
    assert abs(ref_t - 5.0) < 1e-10


# ── Stage 4: Validity + distance ────────────────────────────────────


def _compute_distance(px_val, py_val, cos_val, sin_val, seg, max_coord=20.0):
    """Helper: build intersection + distance graph and compute for one case.

    Uses the full pipeline including per-angle lookup, matching what
    build_renderer_graph does.
    """
    from torchwright.doom.renderer import _build_angle_lookup
    from torchwright.graph import Concatenate
    from torchwright.ops.arithmetic_ops import signed_multiply

    px = create_input("player_x", 1)
    py = create_input("player_y", 1)
    ray_angle = create_input("ray_angle", 1)
    rc = create_input("ray_cos", 1)
    rs = create_input("ray_sin", 1)
    pxs = create_input("px_sin", 1)
    pyc = create_input("py_cos", 1)

    cos_sin = Concatenate([rc, rs])
    px_py = Concatenate([px, py])
    trig_and_products = Concatenate([rc, rs, pxs, pyc])

    # Build angle lookup for this single segment
    angle_data = _build_angle_lookup(ray_angle, [seg])
    signed_inv_den, abs_den_node, sign_den = angle_data[0]

    _den, num_t, num_u = _segment_intersection(cos_sin, px_py, trig_and_products, seg)
    dist = _segment_distance(
        num_t, num_u, signed_inv_den, abs_den_node, sign_den, max_coord,
    )

    # Find the ray_angle that produces cos_val, sin_val
    import numpy as np
    angle = int(round(np.arctan2(sin_val, cos_val) * 256 / (2 * np.pi))) % 256

    vals = {
        "player_x": torch.tensor([[px_val]]),
        "player_y": torch.tensor([[py_val]]),
        "ray_angle": torch.tensor([[float(angle)]]),
        "ray_cos": torch.tensor([[cos_val]]),
        "ray_sin": torch.tensor([[sin_val]]),
        "px_sin": torch.tensor([[px_val * sin_val]]),
        "py_cos": torch.tensor([[py_val * cos_val]]),
    }
    return dist.compute(1, vals).item()


def test_distance_direct_hit():
    """Ray hitting a wall returns correct distance."""
    seg = Segment(ax=5.0, ay=-10.0, bx=5.0, by=10.0, color=(1.0, 0.0, 0.0))
    dist = _compute_distance(0.0, 0.0, 1.0, 0.0, seg)
    assert abs(dist - 5.0) < 0.5


def test_distance_behind_player():
    """Wall behind player returns BIG_DISTANCE."""
    seg = Segment(ax=-5.0, ay=-10.0, bx=-5.0, by=10.0, color=(1.0, 0.0, 0.0))
    dist = _compute_distance(0.0, 0.0, 1.0, 0.0, seg)
    assert dist > BIG_DISTANCE * 0.9


def test_distance_parallel():
    """Ray parallel to segment returns BIG_DISTANCE."""
    seg = Segment(ax=3.0, ay=0.0, bx=8.0, by=0.0, color=(1.0, 0.0, 0.0))
    dist = _compute_distance(0.0, 0.0, 1.0, 0.0, seg)
    assert dist > BIG_DISTANCE * 0.9


def test_distance_miss_outside_segment():
    """Ray misses segment (u outside [0,1]) returns BIG_DISTANCE."""
    seg = Segment(ax=5.0, ay=2.0, bx=5.0, by=3.0, color=(1.0, 0.0, 0.0))
    dist = _compute_distance(0.0, 0.0, 1.0, 0.0, seg)
    assert dist > BIG_DISTANCE * 0.9


# ── Full graph: single column ───────────────────────────────────────


def _render_column_graph(px_val, py_val, angle, col, segments, config):
    """Render one column via graph .compute() and return (H, 3) array."""
    output, pos_encoding = build_renderer_graph(segments, config)

    W = config.screen_width
    col_offset = col - W // 2
    ray_angle = (angle + col_offset * config.fov_columns // W) % 256
    angle_diff = (ray_angle - angle) % 256
    perp_cos = config.trig_table[angle_diff, 0]

    vals = {
        "perp_cos": torch.tensor([[float(perp_cos)]]),
        "player_x": torch.tensor([[float(px_val)]]),
        "player_y": torch.tensor([[float(py_val)]]),
        "ray_angle": torch.tensor([[float(ray_angle)]]),
    }
    result = output.compute(1, vals)  # (1, H*3)
    return result.squeeze(0).reshape(config.screen_height, 3).numpy()


def test_single_column_head_on(config):
    """Center column looking at a wall matches reference renderer.

    At most 1 pixel boundary difference at wall edges is acceptable
    since the graph uses continuous wall bounds while the reference
    uses int() truncation.
    """
    seg = Segment(ax=5.0, ay=-10.0, bx=5.0, by=10.0, color=(1.0, 0.0, 0.0))
    col = config.screen_width // 2

    graph_col = _render_column_graph(0.0, 0.0, 0, col, [seg], config)
    ref_frame = render_frame(0.0, 0.0, 0, [seg], config)
    ref_col = ref_frame[:, col, :]

    # Allow up to 1 pixel boundary mismatch
    mismatched = np.abs(graph_col - ref_col) > 0.15
    n_bad = mismatched.any(axis=1).sum()
    assert n_bad <= 1, (
        f"Too many mismatched rows ({n_bad}). Graph:\n{graph_col}\nRef:\n{ref_col}"
    )


def test_single_column_no_hit(config):
    """Column with no wall hit shows ceiling/floor only."""
    col = config.screen_width // 2
    graph_col = _render_column_graph(0.0, 0.0, 0, col, [], config)

    center = config.screen_height // 2
    # With no segments, wall height ≈ 0, so all rows should be ceiling or floor
    # split at center. Allow for the wall band to be ≈0 pixels.
    # Check top row is ceiling, bottom row is floor
    np.testing.assert_allclose(graph_col[0], config.ceiling_color, atol=0.15,
                               err_msg=f"Row 0 should be ceiling. Full col:\n{graph_col}")
    np.testing.assert_allclose(graph_col[-1], config.floor_color, atol=0.15,
                               err_msg=f"Last row should be floor. Full col:\n{graph_col}")
