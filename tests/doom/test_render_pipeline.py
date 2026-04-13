"""Full render pipeline test: precomputed values + angle offset → pixels.

Tests the optimized render pipeline that precomputes wall data in the
player's angular frame at sort time. The standalone subgraph computes
sort_den, C, D, E, H_inv_num_t from raw inputs, then feeds them through
the same wall-height and u-coordinate pipeline used in game_graph.py.

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
    subtract,
)
from torchwright.ops.inout_nodes import create_input, create_literal_value, create_pos_encoding
from torchwright.ops.map_select import select
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


_DIFF_BP = [
    -40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5,
    0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40,
]
_TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
_PERP_COS_BP = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
_PERP_SIN_BP = [-0.5, -0.35, -0.25, -0.15, -0.08, 0, 0.08, 0.15, 0.25, 0.35, 0.5]
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


def _build_precomp_graph(pos):
    """Build precomputed-value render subgraph (matches game_graph.py).

    Inputs: wall_ax, wall_ay, wall_bx, wall_by, player_x, player_y,
            player_cos (cos of player angle), player_sin (sin of player angle),
            angle_offset (per-column offset from center in angle units)

    Returns (output_node, config) where output_node concatenates
    intermediates + wall_height + wall_top + wall_bottom.
    """
    config = _config()

    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    player_cos = create_input("player_cos", 1)
    player_sin = create_input("player_sin", 1)
    angle_offset = create_input("angle_offset", 1)

    # --- Sort-time precomputation (from raw wall + player data) ---
    ex = subtract(wall_bx, wall_ax)
    ey = subtract(wall_by, wall_ay)
    fx = subtract(wall_ax, player_x)
    gy = subtract(player_y, wall_ay)

    ey_cos = piecewise_linear_2d(ey, player_cos, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="ey_cos_p")
    ex_sin = piecewise_linear_2d(ex, player_sin, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="ex_sin_p")
    sort_den = subtract(ey_cos, ex_sin)

    ey_sin = piecewise_linear_2d(ey, player_sin, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="ey_sin_p")
    ex_cos = piecewise_linear_2d(ex, player_cos, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="ex_cos_p")
    C_val = add(ey_sin, ex_cos)

    fx_sin = piecewise_linear_2d(fx, player_sin, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="fx_sin_p")
    gy_cos = piecewise_linear_2d(gy, player_cos, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="gy_cos_p")
    D_val = add(fx_sin, gy_cos)

    fx_cos = piecewise_linear_2d(fx, player_cos, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="fx_cos_p")
    gy_sin = piecewise_linear_2d(gy, player_sin, _DIFF_BP, _TRIG_BP,
                                 lambda a, b: a * b, name="gy_sin_p")
    E_val = subtract(fx_cos, gy_sin)

    ey_fx = piecewise_linear_2d(ey, fx, _DIFF_BP, _DIFF_BP,
                                lambda a, b: a * b, name="ey_fx")
    ex_gy = piecewise_linear_2d(ex, gy, _DIFF_BP, _DIFF_BP,
                                lambda a, b: a * b, name="ex_gy")
    num_t = add(ey_fx, ex_gy)
    abs_num_t = node_abs(num_t)
    inv_abs_num_t = reciprocal(abs_num_t, min_value=0.3,
                               max_value=2.0 * MAX_COORD * MAX_COORD, step=1.0)
    H_inv_num_t = multiply_const(inv_abs_num_t, float(H))

    # --- Render-time pipeline (from precomputed values + angle offset) ---
    half_fov = FOV // 2
    tan_bp = [float(i) for i in range(-half_fov, half_fov + 1)]
    tan_o = piecewise_linear(
        angle_offset, tan_bp,
        lambda x: math.tan(x * 2.0 * math.pi / 256.0),
        name="tan_offset",
    )

    max_tan = math.tan(half_fov * 2.0 * math.pi / 256.0) * 1.1
    tan_val_bp = [-max_tan + i * (2 * max_tan / 10) for i in range(11)]
    C_tan = piecewise_linear_2d(
        C_val, tan_o, _DIFF_BP, tan_val_bp,
        lambda a, b: a * b, name="C_tan_o",
    )
    den_over_cos = subtract(sort_den, C_tan)
    abs_den_over_cos = node_abs(den_over_cos)

    max_h_inv = float(H) / 0.3
    h_inv_n = 16
    h_inv_ratio = (max_h_inv / 0.01) ** (1.0 / (h_inv_n - 1))
    height_inv_bp = [0.01 * (h_inv_ratio ** k) for k in range(h_inv_n)]
    height_inv_bp[0] = 0.0
    height_inv_bp[-1] = max_h_inv

    doc_max = 2.5 * MAX_COORD
    doc_bp = [doc_max * i / 15 for i in range(16)]

    wall_height_raw = piecewise_linear_2d(
        H_inv_num_t, abs_den_over_cos,
        height_inv_bp, doc_bp,
        lambda a, b: a * b, name="wall_height_raw",
    )
    wall_height = clamp(wall_height_raw, 0.0, float(H))

    center = float(H) / 2.0
    half_height = multiply_const(wall_height, 0.5)
    wall_top = Linear(
        half_height, torch.tensor([[-1.0]]),
        torch.tensor([center]), name="wall_top",
    )
    wall_bottom = Linear(
        half_height, torch.tensor([[1.0]]),
        torch.tensor([center]), name="wall_bottom",
    )

    # Output: intermediates + final height
    output = Concatenate([
        sort_den, C_val, num_t,             # 3: precomputed values
        den_over_cos, abs_den_over_cos,     # 2: per-column intermediates
        H_inv_num_t,                        # 1: height scale
        wall_top, wall_bottom,              # 2: screen positions
        wall_height,                        # 1: height
    ])
    return output, pos


def _precomp_inputs(wall_ax, wall_ay, wall_bx, wall_by,
                    player_x, player_y, player_angle_deg, angle_offset):
    """Build input tensor for the precomputed-value test graph.

    player_angle_deg: player facing angle in degrees (0=east, 90=north).
    angle_offset: per-column offset in angle units (256 = full circle).
    """
    pa_rad = player_angle_deg * 2.0 * math.pi / 256.0
    p_cos = math.cos(pa_rad)
    p_sin = math.sin(pa_rad)
    # Alphabetical: angle_offset, player_cos, player_sin, player_x, player_y,
    #               wall_ax, wall_ay, wall_bx, wall_by
    return torch.tensor([[
        angle_offset, p_cos, p_sin, player_x, player_y,
        wall_ax, wall_ay, wall_bx, wall_by,
    ]])


# ---------------------------------------------------------------------------
# Test: Intermediates — wall height from precomputed values
# ---------------------------------------------------------------------------


class TestRenderIntermediate:
    """Compile precomputed-value wall height, check intermediates."""

    @pytest.fixture(scope="class")
    def module(self):
        pos = create_pos_encoding()
        output, pos = _build_precomp_graph(pos)
        return compile_headless(output, pos, d=2048, d_head=32,
                                max_layers=100, verbose=False)

    def _run(self, module, wall_ax, wall_ay, wall_bx, wall_by,
             player_x, player_y, player_angle_deg, angle_offset=0):
        inputs = _precomp_inputs(
            wall_ax, wall_ay, wall_bx, wall_by,
            player_x, player_y, player_angle_deg, angle_offset,
        )
        with torch.no_grad():
            out = module(inputs)[0]
        return {
            "sort_den": out[0].item(),
            "C": out[1].item(),
            "num_t": out[2].item(),
            "den_over_cos": out[3].item(),
            "abs_den_over_cos": out[4].item(),
            "H_inv_num_t": out[5].item(),
            "wall_top": out[6].item(),
            "wall_bottom": out[7].item(),
            "wall_height": out[8].item(),
        }

    def test_center_dist5(self, module):
        """Player at (0,0) facing east, east wall at x=5: height≈4.8."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=0, player_y=0, player_angle_deg=0)
        print(f"  center dist5: {r}")
        # Expected: z_depth=5, wall_height = 24/5 = 4.8
        assert builtins.abs(r["wall_height"] - 4.8) < 1.5, \
            f"wall_height should be ~4.8, got {r['wall_height']:.3f}"

    def test_near_dist2(self, module):
        """Player at (3,0) facing east, east wall: dist=2, height=12."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=3, player_y=0, player_angle_deg=0)
        print(f"  near dist2: {r}")
        # Expected: z_depth=2, wall_height = 24/2 = 12
        assert builtins.abs(r["wall_height"] - 12.0) < 2.0, \
            f"wall_height should be ~12, got {r['wall_height']:.3f}"
        assert builtins.abs(r["wall_top"] - 6.0) < 2.0, \
            f"wall_top should be ~6, got {r['wall_top']:.3f}"
        assert builtins.abs(r["wall_bottom"] - 18.0) < 2.0, \
            f"wall_bottom should be ~18, got {r['wall_bottom']:.3f}"

    def test_very_near_dist1(self, module):
        """Player at (4,0) facing east, east wall: dist=1, height=24."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=4, player_y=0, player_angle_deg=0)
        print(f"  very near dist1: {r}")
        assert r["wall_height"] > 20, \
            f"wall should fill screen, got {r['wall_height']:.1f}"

    def test_oblique_wall(self, module):
        """Player at (0,0) facing east, oblique wall: height varies per column."""
        # Wall from (3, -1) to (5, 1) — not perpendicular to view.
        # Center ray (offset=0): hits at some point, z_depth depends on geometry.
        # This tests that the precomputed pipeline handles oblique walls correctly.
        r = self._run(module,
                      wall_ax=3, wall_ay=-1, wall_bx=5, wall_by=1,
                      player_x=0, player_y=0, player_angle_deg=0)
        print(f"  oblique wall: {r}")
        # Wall is close-ish (3-5 units away), should produce visible height
        assert r["wall_height"] > 2.0, \
            f"oblique wall should be visible, got {r['wall_height']:.1f}"
        assert r["wall_height"] < H, \
            f"oblique wall shouldn't fill screen, got {r['wall_height']:.1f}"


# ---------------------------------------------------------------------------
# Test: Full pixel output — precomputed pipeline + texture + column fill
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
        player_cos = create_input("player_cos", 1)
        player_sin = create_input("player_sin", 1)
        angle_offset = create_input("angle_offset", 1)
        perp_cos = create_input("perp_cos", 1)
        perp_sin = create_input("perp_sin", 1)

        # Sort-time precomputation
        ex = subtract(wall_bx, wall_ax)
        ey = subtract(wall_by, wall_ay)
        fx = subtract(wall_ax, player_x)
        gy = subtract(player_y, wall_ay)

        ey_cos_p = piecewise_linear_2d(ey, player_cos, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="ey_cos_p")
        ex_sin_p = piecewise_linear_2d(ex, player_sin, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="ex_sin_p")
        sort_den = subtract(ey_cos_p, ex_sin_p)

        ey_sin_p = piecewise_linear_2d(ey, player_sin, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="ey_sin_p")
        ex_cos_p = piecewise_linear_2d(ex, player_cos, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="ex_cos_p")
        C_val = add(ey_sin_p, ex_cos_p)

        fx_sin_p = piecewise_linear_2d(fx, player_sin, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="fx_sin_p")
        gy_cos_p = piecewise_linear_2d(gy, player_cos, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="gy_cos_p")
        D_val = add(fx_sin_p, gy_cos_p)

        fx_cos_p = piecewise_linear_2d(fx, player_cos, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="fx_cos_p")
        gy_sin_p = piecewise_linear_2d(gy, player_sin, _DIFF_BP, _TRIG_BP,
                                       lambda a, b: a * b, name="gy_sin_p")
        E_val = subtract(fx_cos_p, gy_sin_p)

        ey_fx = piecewise_linear_2d(ey, fx, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="ey_fx")
        ex_gy = piecewise_linear_2d(ex, gy, _DIFF_BP, _DIFF_BP,
                                    lambda a, b: a * b, name="ex_gy")
        num_t = add(ey_fx, ex_gy)
        abs_num_t = node_abs(num_t)
        inv_abs_num_t = reciprocal(abs_num_t, min_value=0.3,
                                   max_value=2.0 * MAX_COORD * MAX_COORD,
                                   step=1.0)
        H_inv_num_t = multiply_const(inv_abs_num_t, float(H))

        # Render-time: wall height
        half_fov = FOV // 2
        tan_bp = [float(i) for i in range(-half_fov, half_fov + 1)]
        tan_o = piecewise_linear(
            angle_offset, tan_bp,
            lambda x: math.tan(x * 2.0 * math.pi / 256.0),
            name="tan_offset",
        )
        max_tan = math.tan(half_fov * 2.0 * math.pi / 256.0) * 1.1
        tan_val_bp = [-max_tan + i * (2 * max_tan / 10) for i in range(11)]
        C_tan = piecewise_linear_2d(
            C_val, tan_o, _DIFF_BP, tan_val_bp,
            lambda a, b: a * b, name="C_tan_o",
        )
        den_over_cos = subtract(sort_den, C_tan)
        abs_den_over_cos = node_abs(den_over_cos)

        max_h_inv = float(H) / 0.3
        h_inv_n = 16
        h_inv_ratio = (max_h_inv / 0.01) ** (1.0 / (h_inv_n - 1))
        height_inv_bp = [0.01 * (h_inv_ratio ** k) for k in range(h_inv_n)]
        height_inv_bp[0] = 0.0
        height_inv_bp[-1] = max_h_inv

        doc_max = 2.5 * MAX_COORD
        doc_bp = [doc_max * i / 15 for i in range(16)]

        wall_height_raw = piecewise_linear_2d(
            H_inv_num_t, abs_den_over_cos,
            height_inv_bp, doc_bp,
            lambda a, b: a * b, name="wall_height_raw",
        )
        wall_height = clamp(wall_height_raw, 0.0, float(H))

        center = float(H) / 2.0
        half_height = multiply_const(wall_height, 0.5)
        wall_top = Linear(
            half_height, torch.tensor([[-1.0]]),
            torch.tensor([center]), name="wall_top",
        )
        wall_bottom = Linear(
            half_height, torch.tensor([[1.0]]),
            torch.tensor([center]), name="wall_bottom",
        )

        # Render-time: u-coordinate
        sign_den = compare(den_over_cos, 0.0)
        abs_den = piecewise_linear_2d(
            abs_den_over_cos, perp_cos, doc_bp, _PERP_COS_BP,
            lambda a, b: a * b, name="abs_den",
        )
        D_cos = piecewise_linear_2d(
            D_val, perp_cos, _DIFF_BP, _PERP_COS_BP,
            lambda a, b: a * b, name="D_cos",
        )
        E_sin = piecewise_linear_2d(
            E_val, perp_sin, _DIFF_BP, _PERP_SIN_BP,
            lambda a, b: a * b, name="E_sin",
        )
        num_u = add(D_cos, E_sin)
        adj_num_u = select(sign_den, num_u, negate(num_u))

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

        pixels = _textured_column_fill(
            wall_top, wall_bottom, wall_height,
            tex_column_colors, tex_h, config, max_coord=MAX_COORD,
        )

        output = Concatenate([wall_height, wall_top, wall_bottom, pixels])
        module = compile_headless(output, pos, d=2048, d_head=32,
                                  max_layers=200, verbose=False)
        return module, texs

    def _run(self, module, wall_ax, wall_ay, wall_bx, wall_by, wall_tex,
             player_x, player_y, player_angle_deg, angle_offset=0):
        pa_rad = player_angle_deg * 2.0 * math.pi / 256.0
        p_cos = math.cos(pa_rad)
        p_sin = math.sin(pa_rad)
        ao_rad = angle_offset * 2.0 * math.pi / 256.0
        pc = math.cos(ao_rad)
        ps = math.sin(ao_rad)
        # Alphabetical: angle_offset, perp_cos, perp_sin, player_cos, player_sin,
        #               player_x, player_y, wall_ax, wall_ay, wall_bx, wall_by, wall_tex
        inputs = torch.tensor([[
            angle_offset, pc, ps, p_cos, p_sin,
            player_x, player_y,
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
            player_x=3, player_y=0, player_angle_deg=0,
        )
        print(f"\n  wall_height={wall_height:.2f} top={wall_top:.2f} bottom={wall_bottom:.2f}")

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
            player_x=0, player_y=0, player_angle_deg=0,
        )
        print(f"\n  wall_height={wall_height:.2f} top={wall_top:.2f} bottom={wall_bottom:.2f}")

        band = _wall_band(pixels)
        detected_height = band[1] - band[0] if band else 0
        print(f"  detected band: {band}  height={detected_height}")

        assert builtins.abs(wall_height - 4.8) < 1.5, \
            f"wall_height should be ~4.8, got {wall_height:.2f}"
        assert detected_height >= 3, \
            f"detected wall band should be >= 3 rows, got {detected_height}"
