"""End-to-end composition test: parametric wall → rendered pixels.

Proves that the walls-as-tokens pipeline composes correctly by taking
wall geometry as **runtime input nodes** (not baked constants), running
the full render pipeline (intersection → distance → wall height →
texture lookup → column fill), and verifying the output pixels match
the reference renderer.

This is the "composition test" that turns four individually-validated
mechanisms into one verified architecture:

    parametric intersection (51 ops, proven) →
    runtime den→angle data (new, ~4 ops) →
    distance masking (reused from renderer.py logic) →
    wall_height_lookup (reused) →
    u_norm_lookup (reused) →
    texture column lookup (baked textures, runtime key) →
    textured_column_fill (reused) →
    pixels (compared against reference renderer)
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    abs,
    add,
    add_const,
    compare,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    square_signed,
    subtract,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true
from torchwright.ops.map_select import select

from torchwright.doom.renderer import (
    _textured_column_fill,
    _u_norm_lookup,
    _wall_height_lookup,
)
from torchwright.reference_renderer.render import render_column
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

from tests._utils.image_compare import compare_images


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_COORD = 20.0
BIG_DISTANCE = 1000.0
_SQUARE_MAX_ABS = 40.0
_SQRT_BREAKPOINTS = [
    0.0, 0.25, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
    64.0, 100.0, 225.0, 400.0, 900.0, 1600.0, 3200.0,
]
_DIFF_BREAKPOINTS = [
    -40.0, -30.0, -20.0, -15.0, -10.0, -7.0, -5.0, -3.0, -2.0, -1.0, -0.5,
    0.0,
    0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0,
]
_TRIG_BREAKPOINTS = [
    -1.0, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0,
]


# ---------------------------------------------------------------------------
# Parametric intersection (copied from test_parametric_intersection.py)
# ---------------------------------------------------------------------------


def _parametric_segment_intersection(
    player_x, player_y, ray_cos, ray_sin,
    wall_ax, wall_ay, wall_bx, wall_by,
):
    """(den, num_t, num_u) from runtime wall endpoints."""
    ex = subtract(wall_bx, wall_ax)
    ey = subtract(wall_by, wall_ay)
    dx = subtract(wall_ax, player_x)
    dy = subtract(player_y, wall_ay)

    ey_cos = piecewise_linear_2d(ey, ray_cos, _DIFF_BREAKPOINTS, _TRIG_BREAKPOINTS,
                                  lambda a, b: a * b, name="ey_cos")
    ex_sin = piecewise_linear_2d(ex, ray_sin, _DIFF_BREAKPOINTS, _TRIG_BREAKPOINTS,
                                  lambda a, b: a * b, name="ex_sin")
    den = subtract(ey_cos, ex_sin)

    ey_dx = piecewise_linear_2d(ey, dx, _DIFF_BREAKPOINTS, _DIFF_BREAKPOINTS,
                                 lambda a, b: a * b, name="ey_dx")
    ex_dy = piecewise_linear_2d(ex, dy, _DIFF_BREAKPOINTS, _DIFF_BREAKPOINTS,
                                 lambda a, b: a * b, name="ex_dy")
    num_t = add(ey_dx, ex_dy)

    dx_sin = piecewise_linear_2d(dx, ray_sin, _DIFF_BREAKPOINTS, _TRIG_BREAKPOINTS,
                                  lambda a, b: a * b, name="dx_sin")
    dy_cos = piecewise_linear_2d(dy, ray_cos, _DIFF_BREAKPOINTS, _TRIG_BREAKPOINTS,
                                  lambda a, b: a * b, name="dy_cos")
    num_u = add(dx_sin, dy_cos)

    return den, num_t, num_u


# ---------------------------------------------------------------------------
# Den → angle data (new: replaces _build_angle_lookup for one wall)
# ---------------------------------------------------------------------------


def _den_to_angle_data(den: Node) -> Tuple[Node, Node, Node]:
    """Compute (signed_inv_den, abs_den, sign_den) from runtime den.

    Replaces the baked ``_build_angle_lookup`` which precomputes these
    as a function of ray_angle with wall endpoints as constants.

    Cost: ~4 MLP sublayers (compare + abs + reciprocal + select).
    """
    epsilon = 0.01
    sign_den = compare(den, 0.0)          # +1 if den > 0, -1 if < 0
    abs_den = abs(den)
    inv_abs_den = reciprocal(
        abs_den,
        min_value=epsilon,
        max_value=2.0 * MAX_COORD,
    )
    signed_inv_den = select(sign_den, inv_abs_den, negate(inv_abs_den))
    return signed_inv_den, abs_den, sign_den


# ---------------------------------------------------------------------------
# Parametric distance + texture metadata
# ---------------------------------------------------------------------------


def _parametric_distance_and_texinfo(
    num_t: Node,
    num_u: Node,
    signed_inv_den: Node,
    abs_den: Node,
    sign_den: Node,
    tex_id_node: Node,
    max_coord: float = MAX_COORD,
) -> Tuple[Node, Node, Node, Node, Node]:
    """Distance pipeline with runtime texture_id.

    Mirrors ``_segment_distance_and_texinfo`` from renderer.py but
    accepts ``tex_id_node`` as a runtime Node instead of a compile-time
    int.

    Returns (dist, adj_num_u, abs_den, tex_id_node, is_valid).
    """
    max_num_t = 2.0 * max_coord * max_coord
    max_inv_den = 1.0 / 0.01  # 100
    epsilon = 0.05

    adj_num_t = select(sign_den, num_t, negate(num_t))
    adj_num_u = select(sign_den, num_u, negate(num_u))

    is_den_nonzero = compare(abs_den, epsilon)
    is_t_pos = compare(adj_num_t, epsilon)
    is_u_ge_0 = compare(adj_num_u, -epsilon)
    u_minus_den = subtract(abs_den, adj_num_u)
    is_u_le_den = compare(u_minus_den, -epsilon)

    abs_inv_den = select(sign_den, signed_inv_den, negate(signed_inv_den))
    dist = signed_multiply(
        adj_num_t, abs_inv_den,
        max_abs1=max_num_t, max_abs2=max_inv_den,
        step=1.0, max_abs_output=BIG_DISTANCE,
    )

    is_valid = bool_all_true([is_den_nonzero, is_t_pos, is_u_ge_0, is_u_le_den])
    big = LiteralValue(torch.tensor([BIG_DISTANCE]), name="big_dist")
    dist = select(is_valid, dist, big)

    return dist, adj_num_u, abs_den, tex_id_node, is_valid


# ---------------------------------------------------------------------------
# Full parametric render graph
# ---------------------------------------------------------------------------


def build_parametric_render_graph(
    config: RenderConfig,
    textures: List[np.ndarray],
    max_coord: float = MAX_COORD,
    rows_per_patch: Optional[int] = None,
) -> Tuple[Node, None]:
    """Build a graph that renders one column from runtime wall params.

    Inputs: player_x, player_y, ray_cos, ray_sin, perp_cos,
            wall_ax, wall_ay, wall_bx, wall_by, wall_tex_id

    Output: H*3 pixel values (or rp*3 if rows_per_patch is set).

    Returns ``(output_node, None)`` — no pos_encoding needed since
    this graph has no cross-position attention.
    """
    H = config.screen_height
    rp = rows_per_patch if rows_per_patch is not None else H
    tex_w = textures[0].shape[0]
    tex_h = textures[0].shape[1]

    # Inputs
    player_x = create_input("player_x", 1, value_range=(-max_coord, max_coord))
    player_y = create_input("player_y", 1, value_range=(-max_coord, max_coord))
    ray_cos = create_input("ray_cos", 1, value_range=(-1.0, 1.0))
    ray_sin = create_input("ray_sin", 1, value_range=(-1.0, 1.0))
    perp_cos = create_input("perp_cos", 1, value_range=(-1.0, 1.0))
    wall_ax = create_input("wall_ax", 1, value_range=(-max_coord, max_coord))
    wall_ay = create_input("wall_ay", 1, value_range=(-max_coord, max_coord))
    wall_bx = create_input("wall_bx", 1, value_range=(-max_coord, max_coord))
    wall_by = create_input("wall_by", 1, value_range=(-max_coord, max_coord))
    wall_tex_id = create_input("wall_tex_id", 1, value_range=(0.0, 255.0))

    # Stage 1-2: Parametric intersection
    den, num_t, num_u = _parametric_segment_intersection(
        player_x, player_y, ray_cos, ray_sin,
        wall_ax, wall_ay, wall_bx, wall_by,
    )

    # Stage 3: Den → angle data
    signed_inv_den, abs_den, sign_den = _den_to_angle_data(den)

    # Stage 4: Distance + validity
    dist, adj_num_u, abs_den_out, tex_id_out, is_valid = _parametric_distance_and_texinfo(
        num_t, num_u, signed_inv_den, abs_den, sign_den,
        wall_tex_id, max_coord,
    )

    # Stage 5: Wall height
    wall_top, wall_bottom, wall_height = _wall_height_lookup(
        dist, perp_cos, config, max_coord,
    )

    # Stage 6: u normalization → texture column index
    tex_col_idx = _u_norm_lookup(adj_num_u, abs_den_out, tex_w, max_coord)

    # Stage 7: Texture column lookup (baked textures, runtime key)
    num_tex = len(textures)
    n_keys = num_tex * tex_w
    flat_key = add(multiply_const(tex_id_out, float(tex_w)), tex_col_idx)

    def _tex_column_values(flat_idx):
        k = int(round(flat_idx))
        if 0 <= k < n_keys:
            tid = k // tex_w
            col = k % tex_w
            return [float(v) for v in textures[tid][col].flatten()]
        return [0.0] * (tex_h * 3)

    tex_column_colors = piecewise_linear(
        flat_key,
        breakpoints=[float(k) for k in range(n_keys)],
        fn=_tex_column_values,
        name="texture_column_lookup",
    )

    # Stage 8: Textured column fill
    pixels = _textured_column_fill(
        wall_top, wall_bottom, wall_height,
        tex_column_colors, tex_h, config,
        max_coord=max_coord,
        rows_per_patch=rp,
    )

    return pixels, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _test_config(screen_height=40, screen_width=40, fov=32):
    trig_table = generate_trig_table()
    return RenderConfig(
        screen_width=screen_width,
        screen_height=screen_height,
        fov_columns=fov,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _make_inputs(
    px, py, ray_angle, config, wall,
) -> Tuple[dict, float, float]:
    """Build input dict for one position + return (ray_cos, ray_sin, perp_cos)."""
    trig = config.trig_table
    rc = float(trig[ray_angle % 256, 0])
    rs = float(trig[ray_angle % 256, 1])

    # perp_cos for fish-eye correction
    player_angle = 0  # assume player faces angle 0 for simplicity
    angle_diff = (ray_angle - player_angle) % 256
    pc = float(trig[angle_diff, 0])

    inputs = {
        "player_x": torch.tensor([[px]]),
        "player_y": torch.tensor([[py]]),
        "ray_cos": torch.tensor([[rc]]),
        "ray_sin": torch.tensor([[rs]]),
        "perp_cos": torch.tensor([[pc]]),
        "wall_ax": torch.tensor([[wall["ax"]]]),
        "wall_ay": torch.tensor([[wall["ay"]]]),
        "wall_bx": torch.tensor([[wall["bx"]]]),
        "wall_by": torch.tensor([[wall["by"]]]),
        "wall_tex_id": torch.tensor([[wall["tex_id"]]]),
    }
    return inputs, rc, rs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parametric_render_hits_wall():
    """Render one column where the ray hits a wall. Verify pixels match
    the reference renderer within the piecewise-linear approximation
    tolerance.
    """
    config = _test_config(screen_height=40, screen_width=40, fov=32)
    textures = default_texture_atlas()
    tex_w = textures[0].shape[0]
    tex_h = textures[0].shape[1]

    output_node, _ = build_parametric_render_graph(config, textures)

    # Player at origin, wall at x=5 (perpendicular to view direction).
    px, py = 0.0, 0.0
    wall = {"ax": 5.0, "ay": -3.0, "bx": 5.0, "by": 3.0, "tex_id": 0.0}
    seg = Segment(ax=wall["ax"], ay=wall["ay"], bx=wall["bx"], by=wall["by"],
                  color=(0.8, 0.2, 0.1), texture_id=int(wall["tex_id"]))

    # Ray straight ahead (column = center of screen)
    col = config.screen_width // 2
    ray_angle = 0  # player_angle + offset = 0

    inputs, rc, rs = _make_inputs(px, py, ray_angle, config, wall)

    # Graph output
    cache = reference_eval(output_node, inputs, n_pos=1)
    got_pixels = cache[output_node][0].numpy()  # (H*3,)
    got_frame = got_pixels.reshape(config.screen_height, 3)

    # Reference output
    ref_frame = render_column(
        col, px, py, 0, [seg], config, textures=textures,
    )

    # Wall region should be textured, not blank
    H = config.screen_height
    wall_region = got_frame[H // 4 : 3 * H // 4]
    assert wall_region.max() > 0.01, (
        "wall region appears blank — intersection might have missed"
    )

    compare_images(got_frame[:, None, :], ref_frame[:, None, :]).assert_matches()


def test_parametric_render_misses_wall():
    """Ray that misses the wall should produce pure ceiling/floor."""
    config = _test_config(screen_height=20)
    textures = default_texture_atlas()

    output_node, _ = build_parametric_render_graph(config, textures)

    # Wall far to the right, ray straight ahead — miss.
    px, py = 0.0, 0.0
    wall = {"ax": 10.0, "ay": 10.0, "bx": 12.0, "by": 10.0, "tex_id": 0.0}
    ray_angle = 0

    inputs, _, _ = _make_inputs(px, py, ray_angle, config, wall)
    cache = reference_eval(output_node, inputs, n_pos=1)
    got = cache[output_node][0].numpy().reshape(config.screen_height, 3)

    # Expect ceiling above center, floor below center
    H = config.screen_height
    ceiling = np.array(config.ceiling_color)
    floor = np.array(config.floor_color)

    # Top quarter should be ceiling-ish, bottom quarter floor-ish.
    top = got[:H // 4]
    bottom = got[3 * H // 4:]
    assert np.allclose(top, ceiling, atol=0.1), (
        f"top pixels {top[0]} don't match ceiling {ceiling}"
    )
    assert np.allclose(bottom, floor, atol=0.1), (
        f"bottom pixels {bottom[0]} don't match floor {floor}"
    )


def test_parametric_render_two_columns():
    """Render two different columns from the same wall at the same
    position (as two batch positions). Verify each matches its
    reference.
    """
    config = _test_config(screen_height=40, screen_width=40, fov=32)
    textures = default_texture_atlas()

    output_node, _ = build_parametric_render_graph(config, textures)

    px, py = 0.0, 0.0
    wall = {"ax": 4.0, "ay": -5.0, "bx": 4.0, "by": 5.0, "tex_id": 1.0}
    seg = Segment(ax=wall["ax"], ay=wall["ay"], bx=wall["bx"], by=wall["by"],
                  color=(0.8, 0.2, 0.1), texture_id=int(wall["tex_id"]))

    trig = config.trig_table
    # Two columns: center and slightly off-center
    cols = [config.screen_width // 2, config.screen_width // 2 + 3]
    n_pos = len(cols)

    # Build batched inputs
    inputs = {name: torch.zeros(n_pos, 1) for name in [
        "player_x", "player_y", "ray_cos", "ray_sin", "perp_cos",
        "wall_ax", "wall_ay", "wall_bx", "wall_by", "wall_tex_id",
    ]}
    for j, col in enumerate(cols):
        col_offset = col - config.screen_width // 2
        ray_angle = (0 + col_offset * config.fov_columns // config.screen_width) % 256
        angle_diff = (ray_angle - 0) % 256
        inputs["player_x"][j, 0] = px
        inputs["player_y"][j, 0] = py
        inputs["ray_cos"][j, 0] = float(trig[ray_angle, 0])
        inputs["ray_sin"][j, 0] = float(trig[ray_angle, 1])
        inputs["perp_cos"][j, 0] = float(trig[angle_diff, 0])
        inputs["wall_ax"][j, 0] = wall["ax"]
        inputs["wall_ay"][j, 0] = wall["ay"]
        inputs["wall_bx"][j, 0] = wall["bx"]
        inputs["wall_by"][j, 0] = wall["by"]
        inputs["wall_tex_id"][j, 0] = wall["tex_id"]

    cache = reference_eval(output_node, inputs, n_pos)
    out = cache[output_node]  # (n_pos, H*3)

    for j, col in enumerate(cols):
        got = out[j].numpy().reshape(config.screen_height, 3)
        ref = render_column(col, px, py, 0, [seg], config, textures=textures)
        compare_images(got[:, None, :], ref[:, None, :]).assert_matches()


def test_probe_parametric_render():
    """Compile the parametric render graph and verify oracle-vs-compiled
    agreement at every materialised node.
    """
    config = _test_config(screen_height=20)
    textures = default_texture_atlas()

    output_node, _ = build_parametric_render_graph(config, textures)

    px, py = 0.0, 0.0
    wall = {"ax": 5.0, "ay": -3.0, "bx": 5.0, "by": 3.0, "tex_id": 0.0}

    inputs, _, _ = _make_inputs(px, py, 0, config, wall)

    report = probe_graph(
        output_node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=1,
        d=2048,
        d_head=16,
        verbose=False,
        atol=1.0,
    )
    assert report.first_divergent is None, (
        f"probe reported divergence:\n{report.format_short()}"
    )
