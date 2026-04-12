"""Full render pipeline test: wall coords + player pos + ray → pixels.

Compiles everything from the parametric intersection through
_textured_column_fill as a standalone subgraph, bypassing the
sort/render attention. This isolates whether the pixel output is
correct when the wall coords are fed directly.

Outputs intermediate values so we can see exactly where precision
loss occurs.
"""

import builtins
import math

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.renderer import (
    _textured_column_fill,
    _u_norm_lookup,
    _wall_height_lookup,
    trig_lookup,
)
from torchwright.graph import Concatenate, Linear
from torchwright.ops.arithmetic_ops import (
    abs as node_abs,
    add,
    add_const,
    clamp,
    compare,
    multiply_const,
    negate,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    subtract,
)
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.logic_ops import bool_all_true
from torchwright.ops.map_select import select
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


_DIFF_BP = [
    -40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5,
    0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40,
]
_TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
BIG_DISTANCE = 1000.0
MAX_COORD = 10.0
W, H, FOV = 32, 24, 8


def _test_textures(tex_size=8):
    """High-contrast solid-color textures — far from ceiling/floor."""
    colors = [
        (0.9, 0.1, 0.1),  # east: red
        (0.1, 0.9, 0.1),  # west: green
        (0.1, 0.1, 0.9),  # north: blue
        (0.9, 0.9, 0.1),  # south: yellow
    ]
    textures = []
    for r, g, b in colors:
        tex = np.full((tex_size, tex_size, 3), [r, g, b], dtype=np.float64)
        textures.append(tex)
    return textures


def _config():
    return RenderConfig(
        screen_width=W, screen_height=H, fov_columns=FOV,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _wall_band(column, tol=0.05):
    """Find wall rows (differs from both ceiling and floor)."""
    ceil_c = np.array([0.2, 0.2, 0.2])
    floor_c = np.array([0.4, 0.4, 0.4])
    is_wall = np.array([
        np.linalg.norm(column[r] - ceil_c) > tol
        and np.linalg.norm(column[r] - floor_c) > tol
        for r in range(len(column))
    ])
    if not is_wall.any():
        return None
    indices = np.where(is_wall)[0]
    return int(indices[0]), int(indices[-1]) + 1


# ---------------------------------------------------------------------------
# Test: Intermediates — compile the intersection + wall height, check values
# ---------------------------------------------------------------------------


class TestRenderIntermediate:
    """Compile intersection + height as one subgraph, check all intermediates."""

    @pytest.fixture(scope="class")
    def module(self):
        config = _config()
        pos = create_pos_encoding()

        # Inputs: wall coords, player pos, ray direction, perp_cos
        wall_ax = create_input("wall_ax", 1)
        wall_ay = create_input("wall_ay", 1)
        wall_bx = create_input("wall_bx", 1)
        wall_by = create_input("wall_by", 1)
        player_x = create_input("player_x", 1)
        player_y = create_input("player_y", 1)
        ray_cos = create_input("ray_cos", 1)
        ray_sin = create_input("ray_sin", 1)
        perp_cos = create_input("perp_cos", 1)

        # --- Parametric intersection (from game_graph.py lines 733-773) ---
        ex = subtract(wall_bx, wall_ax)
        ey = subtract(wall_by, wall_ay)
        dx_r = subtract(wall_ax, player_x)
        dy_r = subtract(player_y, wall_ay)

        ey_cos = piecewise_linear_2d(ey, ray_cos, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_ey_cos")
        ex_sin = piecewise_linear_2d(ex, ray_sin, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_ex_sin")
        den = subtract(ey_cos, ex_sin)

        ey_dx = piecewise_linear_2d(ey, dx_r, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="r_ey_dx")
        ex_dy = piecewise_linear_2d(ex, dy_r, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="r_ex_dy")
        num_t = add(ey_dx, ex_dy)

        dx_sin = piecewise_linear_2d(dx_r, ray_sin, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_dx_sin")
        dy_cos = piecewise_linear_2d(dy_r, ray_cos, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_dy_cos")
        num_u = add(dx_sin, dy_cos)

        sign_den = compare(den, 0.0)
        abs_den = node_abs(den)
        inv_abs_den = reciprocal(abs_den, min_value=0.01,
                                 max_value=2.0 * MAX_COORD)
        signed_inv_den = select(sign_den, inv_abs_den, negate(inv_abs_den))

        adj_num_t = select(sign_den, num_t, negate(num_t))
        adj_num_u = select(sign_den, num_u, negate(num_u))
        is_den_nz = compare(abs_den, 0.05)
        is_t_pos = compare(adj_num_t, 0.05)
        is_u_ge0 = compare(adj_num_u, -0.05)
        u_minus_den = subtract(abs_den, adj_num_u)
        is_u_le_den = compare(u_minus_den, -0.05)
        abs_inv = select(sign_den, signed_inv_den, negate(signed_inv_den))
        dist_r = signed_multiply(
            adj_num_t, abs_inv,
            max_abs1=2.0 * MAX_COORD * MAX_COORD,
            max_abs2=1.0 / 0.01,
            step=1.0, max_abs_output=BIG_DISTANCE,
        )
        is_valid = bool_all_true([is_den_nz, is_t_pos, is_u_ge0, is_u_le_den])
        big = create_literal_value(torch.tensor([BIG_DISTANCE]), name="big")
        dist_r_gated = select(is_valid, dist_r, big)

        # --- Wall height ---
        wall_top, wall_bottom, wall_height = _wall_height_lookup(
            dist_r_gated, perp_cos, config, MAX_COORD,
        )

        # Output: all intermediates + final height
        output = Concatenate([
            den, num_t, num_u,           # 3: intersection params
            dist_r, dist_r_gated,        # 2: raw and gated distance
            is_valid,                     # 1: validity flag
            wall_top, wall_bottom,        # 2: screen positions
            wall_height,                  # 1: height
        ])
        return compile_headless(output, pos, d=2048, d_head=32,
                                max_layers=100, verbose=False)

    def _run(self, module, wall_ax, wall_ay, wall_bx, wall_by,
             player_x, player_y, ray_cos, ray_sin, perp_cos):
        # Alphabetical: perp_cos, player_x, player_y, ray_cos, ray_sin,
        #               wall_ax, wall_ay, wall_bx, wall_by
        inputs = torch.tensor([[
            perp_cos, player_x, player_y, ray_cos, ray_sin,
            wall_ax, wall_ay, wall_bx, wall_by,
        ]])
        with torch.no_grad():
            out = module(inputs)[0]
        return {
            "den": out[0].item(),
            "num_t": out[1].item(),
            "num_u": out[2].item(),
            "dist_r": out[3].item(),
            "dist_r_gated": out[4].item(),
            "is_valid": out[5].item(),
            "wall_top": out[6].item(),
            "wall_bottom": out[7].item(),
            "wall_height": out[8].item(),
        }

    def test_center_dist5(self, module):
        """Player at (0,0) facing east, east wall: dist=5, height≈4.8."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=0, player_y=0,
                      ray_cos=1.0, ray_sin=0.0, perp_cos=1.0)
        print(f"  center dist5: {r}")
        assert builtins.abs(r["dist_r_gated"] - 5.0) < 0.5
        assert builtins.abs(r["wall_height"] - 4.8) < 1.5

    def test_near_dist2(self, module):
        """Player at (3,0) facing east, east wall: dist=2, height=12."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=3, player_y=0,
                      ray_cos=1.0, ray_sin=0.0, perp_cos=1.0)
        print(f"  near dist2: {r}")
        assert r["is_valid"] > 0, f"intersection should be valid, got is_valid={r['is_valid']}"
        assert builtins.abs(r["den"] - 10) < 0.5, f"den should be 10, got {r['den']:.3f}"
        assert builtins.abs(r["num_t"] - 20) < 1.0, f"num_t should be 20, got {r['num_t']:.3f}"
        assert builtins.abs(r["dist_r_gated"] - 2.0) < 0.5, \
            f"dist_r should be 2.0, got {r['dist_r_gated']:.3f}"
        assert builtins.abs(r["wall_height"] - 12.0) < 1.5, \
            f"wall_height should be 12, got {r['wall_height']:.3f}"
        assert builtins.abs(r["wall_top"] - 6.0) < 1.5, \
            f"wall_top should be ~6, got {r['wall_top']:.3f}"
        assert builtins.abs(r["wall_bottom"] - 18.0) < 1.5, \
            f"wall_bottom should be ~18, got {r['wall_bottom']:.3f}"

    def test_very_near_dist1(self, module):
        """Player at (4,0) facing east, east wall: dist=1, height=24."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=4, player_y=0,
                      ray_cos=1.0, ray_sin=0.0, perp_cos=1.0)
        print(f"  very near dist1: {r}")
        assert builtins.abs(r["dist_r_gated"] - 1.0) < 0.5
        assert r["wall_height"] > 20, f"wall should fill screen, got {r['wall_height']:.1f}"


# ---------------------------------------------------------------------------
# Test: Full pixel output — intersection + height + texture + column fill
# ---------------------------------------------------------------------------


class TestRenderPixels:
    """Compile the full render subgraph through to pixel output."""

    @pytest.fixture(scope="class")
    def module_and_textures(self):
        config = _config()
        texs = _test_textures(tex_size=8)
        tex_w, tex_h = texs[0].shape[0], texs[0].shape[1]
        num_tex = len(texs)
        n_keys = num_tex * tex_w

        pos = create_pos_encoding()

        wall_ax = create_input("wall_ax", 1)
        wall_ay = create_input("wall_ay", 1)
        wall_bx = create_input("wall_bx", 1)
        wall_by = create_input("wall_by", 1)
        wall_tex = create_input("wall_tex", 1)
        player_x = create_input("player_x", 1)
        player_y = create_input("player_y", 1)
        ray_cos = create_input("ray_cos", 1)
        ray_sin = create_input("ray_sin", 1)
        perp_cos = create_input("perp_cos", 1)

        # Parametric intersection
        ex = subtract(wall_bx, wall_ax)
        ey = subtract(wall_by, wall_ay)
        dx_r = subtract(wall_ax, player_x)
        dy_r = subtract(player_y, wall_ay)

        ey_cos = piecewise_linear_2d(ey, ray_cos, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_ey_cos")
        ex_sin = piecewise_linear_2d(ex, ray_sin, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_ex_sin")
        den = subtract(ey_cos, ex_sin)

        ey_dx = piecewise_linear_2d(ey, dx_r, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="r_ey_dx")
        ex_dy = piecewise_linear_2d(ex, dy_r, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="r_ex_dy")
        num_t = add(ey_dx, ex_dy)

        dx_sin = piecewise_linear_2d(dx_r, ray_sin, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_dx_sin")
        dy_cos = piecewise_linear_2d(dy_r, ray_cos, _DIFF_BP, _TRIG_BP,
                                     lambda a, b: a * b, name="r_dy_cos")
        num_u = add(dx_sin, dy_cos)

        sign_den = compare(den, 0.0)
        abs_den = node_abs(den)
        inv_abs_den = reciprocal(abs_den, min_value=0.01,
                                 max_value=2.0 * MAX_COORD)
        signed_inv_den = select(sign_den, inv_abs_den, negate(inv_abs_den))

        adj_num_t = select(sign_den, num_t, negate(num_t))
        adj_num_u = select(sign_den, num_u, negate(num_u))
        is_den_nz = compare(abs_den, 0.05)
        is_t_pos = compare(adj_num_t, 0.05)
        is_u_ge0 = compare(adj_num_u, -0.05)
        u_minus_den = subtract(abs_den, adj_num_u)
        is_u_le_den = compare(u_minus_den, -0.05)
        abs_inv = select(sign_den, signed_inv_den, negate(signed_inv_den))
        dist_r = signed_multiply(
            adj_num_t, abs_inv,
            max_abs1=2.0 * MAX_COORD * MAX_COORD,
            max_abs2=1.0 / 0.01,
            step=1.0, max_abs_output=BIG_DISTANCE,
        )
        is_valid = bool_all_true([is_den_nz, is_t_pos, is_u_ge0, is_u_le_den])
        big = create_literal_value(torch.tensor([BIG_DISTANCE]), name="big")
        dist_r = select(is_valid, dist_r, big)

        # Wall height
        wall_top, wall_bottom, wall_height = _wall_height_lookup(
            dist_r, perp_cos, config, MAX_COORD,
        )

        # Texture column lookup
        tex_col_idx = _u_norm_lookup(adj_num_u, abs_den, tex_w, MAX_COORD)

        def _tex_col_vals(flat_idx):
            k = int(round(flat_idx))
            if 0 <= k < n_keys:
                tid = k // tex_w
                col = k % tex_w
                return [float(v) for v in texs[tid][col].flatten()]
            return [0.0] * (tex_h * 3)

        flat_key = add(multiply_const(wall_tex, float(tex_w)), tex_col_idx)
        tex_column_colors = piecewise_linear(
            flat_key, [float(k) for k in range(n_keys)],
            _tex_col_vals, name="tex_col_lookup",
        )

        # Column fill (no patching — full height)
        pixels = _textured_column_fill(
            wall_top, wall_bottom, wall_height,
            tex_column_colors, tex_h, config, max_coord=MAX_COORD,
        )

        output = Concatenate([wall_height, wall_top, wall_bottom, pixels])
        module = compile_headless(output, pos, d=2048, d_head=32,
                                  max_layers=200, verbose=False)
        return module, texs

    def _run(self, module, wall_ax, wall_ay, wall_bx, wall_by, wall_tex,
             player_x, player_y, ray_cos, ray_sin, perp_cos):
        # Alphabetical: perp_cos, player_x, player_y, ray_cos, ray_sin,
        #               wall_ax, wall_ay, wall_bx, wall_by, wall_tex
        inputs = torch.tensor([[
            perp_cos, player_x, player_y, ray_cos, ray_sin,
            wall_ax, wall_ay, wall_bx, wall_by, wall_tex,
        ]])
        with torch.no_grad():
            out = module(inputs)[0]
        wall_height = out[0].item()
        wall_top = out[1].item()
        wall_bottom = out[2].item()
        pixels = out[3:].numpy().reshape(H, 3)
        return wall_height, wall_top, wall_bottom, pixels

    def test_near_wall_pixel_output(self, module_and_textures):
        """Player at (3,0) facing east: wall should span ~12 rows."""
        module, texs = module_and_textures
        wall_height, wall_top, wall_bottom, pixels = self._run(
            module,
            wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5, wall_tex=0,
            player_x=3, player_y=0,
            ray_cos=1.0, ray_sin=0.0, perp_cos=1.0,
        )
        print(f"\n  wall_height={wall_height:.2f} top={wall_top:.2f} bottom={wall_bottom:.2f}")

        # Print every row's RGB and classification
        ceil_c = np.array([0.2, 0.2, 0.2])
        floor_c = np.array([0.4, 0.4, 0.4])
        for r in range(H):
            rgb = pixels[r]
            d_ceil = np.linalg.norm(rgb - ceil_c)
            d_floor = np.linalg.norm(rgb - floor_c)
            label = "CEIL" if d_ceil < 0.05 else ("FLOOR" if d_floor < 0.05 else "WALL")
            print(f"  row {r:2d}: [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]  "
                  f"d_ceil={d_ceil:.3f} d_floor={d_floor:.3f}  {label}")

        band = _wall_band(pixels)
        detected_height = band[1] - band[0] if band else 0
        print(f"\n  detected band: {band}  height={detected_height}")

        assert builtins.abs(wall_height - 12.0) < 2.0, \
            f"wall_height should be ~12, got {wall_height:.2f}"
        assert detected_height >= 10, \
            f"detected wall band should be >= 10 rows, got {detected_height}"

    def test_center_wall_pixel_output(self, module_and_textures):
        """Player at (0,0) facing east: wall should span ~5 rows."""
        module, texs = module_and_textures
        wall_height, wall_top, wall_bottom, pixels = self._run(
            module,
            wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5, wall_tex=0,
            player_x=0, player_y=0,
            ray_cos=1.0, ray_sin=0.0, perp_cos=1.0,
        )
        print(f"\n  wall_height={wall_height:.2f} top={wall_top:.2f} bottom={wall_bottom:.2f}")

        band = _wall_band(pixels)
        detected_height = band[1] - band[0] if band else 0
        print(f"  detected band: {band}  height={detected_height}")

        assert builtins.abs(wall_height - 4.8) < 1.5, \
            f"wall_height should be ~4.8, got {wall_height:.2f}"
        assert detected_height >= 3, \
            f"detected wall band should be >= 3 rows, got {detected_height}"
