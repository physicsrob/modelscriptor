"""Full render pipeline test: precomputed values + angle offset → pixels.

Tests the optimized render pipeline that precomputes wall data in the
player's angular frame at sort time. The standalone subgraph computes
sort_den, C, D, E, H_inv_num_t from raw inputs, then feeds them through
the same wall-height and u-coordinate pipeline used in game_graph.py.

The u-coordinate uses the tan(offset) formulation:
    u = |D + E·tan(o)| / |A - C·tan(o)|
which eliminates the need for perp_cos/perp_sin entirely.
"""

import builtins
import math

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.renderer import (
    _textured_column_fill,
    trig_lookup,
)
from torchwright.graph import Concatenate, Linear
from torchwright.ops.arithmetic_ops import (
    abs as node_abs,
    add,
    add_const,
    clamp,
    compare,
    floor_int,
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


# Shared breakpoints for the render pipeline
_HALF_FOV = FOV // 2
_MAX_TAN = math.tan(_HALF_FOV * 2.0 * math.pi / 256.0) * 1.1
_TAN_BP = [float(i) for i in range(-_HALF_FOV, _HALF_FOV + 1)]
_TAN_VAL_BP = [-_MAX_TAN + i * (2 * _MAX_TAN / 10) for i in range(11)]
_DOC_MAX = 2.5 * MAX_COORD
_DOC_BP = [_DOC_MAX * i / 15 for i in range(16)]

_MAX_H_INV = float(H) / 0.3
_H_INV_N = 16
_H_INV_RATIO = (_MAX_H_INV / 0.01) ** (1.0 / (_H_INV_N - 1))
_HEIGHT_INV_BP = [0.01 * (_H_INV_RATIO ** k) for k in range(_H_INV_N)]
_HEIGHT_INV_BP[0] = 0.0
_HEIGHT_INV_BP[-1] = _MAX_H_INV


def _build_sort_precomp(pos):
    """Build sort-time precomputation from raw wall + player data.

    Returns (precomp_dict, pos) where precomp_dict has all precomputed
    nodes and input nodes needed for rendering.
    """
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    player_cos = create_input("player_cos", 1)
    player_sin = create_input("player_sin", 1)
    angle_offset = create_input("angle_offset", 1)

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

    return {
        "sort_den": sort_den, "C": C_val, "D": D_val, "E": E_val,
        "H_inv_num_t": H_inv_num_t, "num_t": num_t,
        "angle_offset": angle_offset,
    }, pos


def _build_render_pipeline(p):
    """Build the render-time pipeline from precomputed values.

    Takes dict p from _build_sort_precomp.  Returns
    (wall_height, wall_top, wall_bottom, tex_col_idx, abs_doc, abs_nuc, doc).
    """
    tan_o = piecewise_linear(
        p["angle_offset"], _TAN_BP,
        lambda x: math.tan(x * 2.0 * math.pi / 256.0),
        name="tan_offset",
    )

    C_tan = piecewise_linear_2d(
        p["C"], tan_o, _DIFF_BP, _TAN_VAL_BP,
        lambda a, b: a * b, name="C_tan_o",
    )
    E_tan = piecewise_linear_2d(
        p["E"], tan_o, _DIFF_BP, _TAN_VAL_BP,
        lambda a, b: a * b, name="E_tan_o",
    )

    doc = subtract(p["sort_den"], C_tan)
    nuc = add(p["D"], E_tan)
    abs_doc = node_abs(doc)
    abs_nuc = node_abs(nuc)

    # Wall height
    wall_height_raw = piecewise_linear_2d(
        p["H_inv_num_t"], abs_doc,
        _HEIGHT_INV_BP, _DOC_BP,
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

    # Texture u-coordinate: u = |nuc| / |doc|
    u_raw = piecewise_linear_2d(
        abs_nuc, abs_doc, _DOC_BP, _DOC_BP,
        lambda n, d: n / d if d > 0.01 else 0.0,
        name="u_ratio",
    )

    return wall_height, wall_top, wall_bottom, u_raw, abs_doc, abs_nuc, doc


def _precomp_inputs(wall_ax, wall_ay, wall_bx, wall_by,
                    player_x, player_y, player_angle_deg, angle_offset):
    """Build input tensor for the precomputed-value test graph."""
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
# Test: Intermediates — wall height + u-coordinate from precomputed values
# ---------------------------------------------------------------------------


class TestRenderIntermediate:
    """Compile precomputed-value wall height + u, check intermediates."""

    @pytest.fixture(scope="class")
    def module(self):
        pos = create_pos_encoding()
        p, pos = _build_sort_precomp(pos)
        wh, wt, wb, u_raw, abs_doc, abs_nuc, doc = _build_render_pipeline(p)

        output = Concatenate([
            p["sort_den"], p["C"], p["num_t"],  # 3: precomputed values
            doc, abs_doc,                        # 2: per-column intermediates
            p["H_inv_num_t"],                    # 1: height scale
            wt, wb, wh,                          # 3: wall geometry
            u_raw,                               # 1: texture u-coordinate
        ])
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
            "doc": out[3].item(),
            "abs_doc": out[4].item(),
            "H_inv_num_t": out[5].item(),
            "wall_top": out[6].item(),
            "wall_bottom": out[7].item(),
            "wall_height": out[8].item(),
            "u_raw": out[9].item(),
        }

    def test_center_dist5(self, module):
        """Player at (0,0) facing east, east wall at x=5: height~4.8."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=0, player_y=0, player_angle_deg=0)
        print(f"  center dist5: {r}")
        assert builtins.abs(r["wall_height"] - 4.8) < 1.5, \
            f"wall_height should be ~4.8, got {r['wall_height']:.3f}"
        # u should be 0.5 (center of wall) for center ray
        assert builtins.abs(r["u_raw"] - 0.5) < 0.15, \
            f"u_raw should be ~0.5 at center, got {r['u_raw']:.3f}"

    def test_near_dist2(self, module):
        """Player at (3,0) facing east, east wall: dist=2, height=12."""
        r = self._run(module,
                      wall_ax=5, wall_ay=-5, wall_bx=5, wall_by=5,
                      player_x=3, player_y=0, player_angle_deg=0)
        print(f"  near dist2: {r}")
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
        """Player at (0,0) facing east, oblique wall (3,-1)→(5,1)."""
        r = self._run(module,
                      wall_ax=3, wall_ay=-1, wall_bx=5, wall_by=1,
                      player_x=0, player_y=0, player_angle_deg=0)
        print(f"  oblique wall: {r}")
        assert r["wall_height"] > 2.0, \
            f"oblique wall should be visible, got {r['wall_height']:.1f}"
        assert r["wall_height"] < H, \
            f"oblique wall shouldn't fill screen, got {r['wall_height']:.1f}"
        # u_raw is the unclamped ratio |nuc|/|doc|; may exceed 1 for
        # columns that see the wall's extended line beyond its endpoints
        assert r["u_raw"] >= 0.0, \
            f"u_raw should be non-negative, got {r['u_raw']:.3f}"


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
        p, pos = _build_sort_precomp(pos)
        wall_tex = create_input("wall_tex", 1)

        wh, wt, wb, u_raw, _, _, _ = _build_render_pipeline(p)

        # Texture column from u_raw
        tex_col_float = multiply_const(u_raw, float(tex_w))
        tex_col_clamped = clamp(tex_col_float, 0.0, float(tex_w) - 0.5)
        tex_col_idx = floor_int(tex_col_clamped, 0, tex_w - 1)

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
            wt, wb, wh,
            tex_column_colors, tex_h, config, max_coord=MAX_COORD,
        )

        output = Concatenate([wh, wt, wb, pixels])
        module = compile_headless(output, pos, d=2048, d_head=32,
                                  max_layers=200, verbose=False)
        return module, texs

    def _run(self, module, wall_ax, wall_ay, wall_bx, wall_by, wall_tex,
             player_x, player_y, player_angle_deg, angle_offset=0):
        pa_rad = player_angle_deg * 2.0 * math.pi / 256.0
        p_cos = math.cos(pa_rad)
        p_sin = math.sin(pa_rad)
        # Alphabetical: angle_offset, player_cos, player_sin, player_x, player_y,
        #               wall_ax, wall_ay, wall_bx, wall_by, wall_tex
        inputs = torch.tensor([[
            angle_offset, p_cos, p_sin, player_x, player_y,
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
