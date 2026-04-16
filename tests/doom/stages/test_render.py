"""Unit tests for the RENDER stage (torchwright.doom.stages.render).

Focus: the two per-column math computations that are most likely to
carry bugs and that don't go through attention:

* ``_compute_wall_height``: height from (H_inv, |den/cos|).
* ``_compute_texture_column``: u-coordinate from (D, E, tan_offset, den/cos).

The attention paths (wall attention over SORTED, texture attention over
TEX_COL) are already exercised end-to-end by ``test_game_graph.py`` and
``test_wall_selection.py``.
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.types import RenderConfig
from torchwright.reference_renderer.trig import generate_trig_table

from torchwright.doom.stages.render import (
    _compute_angle_offset_tan,
    _compute_den_over_cos,
    _compute_texture_column,
    _compute_wall_height,
)


_MAX_COORD = 20.0


@pytest.fixture(scope="module")
def wall_height_module():
    """Compile a subgraph for the wall-height pipeline.

    Exposes wall_height directly so we can probe it at synthetic
    ``r_H_inv, r_sort_den, r_C, angle_offset`` values without going
    through the RENDER attention.
    """
    H = 48
    fov = 32

    pos = create_pos_encoding()
    r_H_inv = create_input("r_H_inv", 1)
    r_sort_den = create_input("r_sort_den", 1)
    r_C = create_input("r_C", 1)
    angle_offset = create_input("angle_offset", 1)

    tan_o, tan_val_bp = _compute_angle_offset_tan(angle_offset, fov=fov)
    _, abs_den_over_cos = _compute_den_over_cos(r_sort_den, r_C, tan_o, tan_val_bp)
    _, _, wall_height = _compute_wall_height(
        r_H_inv, abs_den_over_cos, H=H, max_coord=_MAX_COORD,
    )
    return compile_headless(
        Concatenate([wall_height]), pos,
        d=1024, d_head=32, max_layers=50, verbose=False,
    )


@pytest.fixture(scope="module")
def tex_col_module():
    """Compile a subgraph for the texture u-coordinate computation.

    Exposes intermediates so tests can verify the division isn't degenerate.
    """
    from torchwright.ops.arithmetic_ops import (
        abs, add, clamp, floor_int, multiply_const, piecewise_linear_2d,
    )
    from torchwright.doom.graph_constants import DIFF_BP

    fov = 32
    tex_w = 8

    pos = create_pos_encoding()
    r_D = create_input("r_D", 1)
    r_E = create_input("r_E", 1)
    r_sort_den = create_input("r_sort_den", 1)
    r_C = create_input("r_C", 1)
    angle_offset = create_input("angle_offset", 1)

    tan_o, tan_val_bp = _compute_angle_offset_tan(angle_offset, fov=fov)
    _, abs_den_over_cos = _compute_den_over_cos(r_sort_den, r_C, tan_o, tan_val_bp)

    # Inline _compute_texture_column so we can expose abs_nuc + u_raw.
    E_tan = piecewise_linear_2d(
        r_E, tan_o, DIFF_BP, tan_val_bp,
        lambda a, b: a * b, name="E_tan_o",
    )
    num_u_over_cos = add(r_D, E_tan)
    abs_nuc = abs(num_u_over_cos)
    doc_max = 2.5 * _MAX_COORD
    doc_bp = [doc_max * i / 15 for i in range(16)]
    u_raw = piecewise_linear_2d(
        abs_nuc, abs_den_over_cos,
        doc_bp, doc_bp,
        lambda n, d: n / d if d > 0.01 else 0.0,
        name="u_ratio",
    )
    tex_col_float = multiply_const(u_raw, float(tex_w))
    tex_col_clamped = clamp(tex_col_float, 0.0, float(tex_w) - 0.5)
    tex_col_idx = floor_int(tex_col_clamped, 0, tex_w - 1)

    return compile_headless(
        Concatenate([tex_col_idx, abs_den_over_cos, abs_nuc, u_raw]), pos,
        d=1024, d_head=32, max_layers=50, verbose=False,
    )


def _pack(module, values: dict) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    row = torch.zeros(1, d_input, dtype=torch.float32)
    for name, start, width in module._input_specs:
        row[0, start:start + width] = torch.tensor(
            values[name], dtype=torch.float32,
        ).reshape(width)
    return row


# ---------------------------------------------------------------------------
# Wall height
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H_inv_num_t,den_over_cos,expected_h", [
    # Wall far (large |num_t|), height small: H_inv = 48/100 = 0.48, den/cos = 1 → h = 0.48.
    (0.48, 1.0, 0.48),
    # Middle distance: H_inv = 48/20 = 2.4, den/cos = 1 → h = 2.4.
    (2.4, 1.0, 2.4),
    # Closer: H_inv = 48/4 = 12, den/cos = 1 → h = 12.
    (12.0, 1.0, 12.0),
    # Fish-eye correction: same H_inv, larger den/cos → taller.
    (12.0, 2.0, 24.0),
])
def test_wall_height_scales_as_H_inv_times_den_over_cos(
    wall_height_module, H_inv_num_t, den_over_cos, expected_h,
):
    """wall_height ≈ H_inv * |den/cos|, clamped to [0, H]."""
    # Set up so that r_sort_den - C*tan(angle_offset=0) = den_over_cos.
    # angle_offset=0 → tan_o=0, so abs_den_over_cos = |r_sort_den|.
    inputs = _pack(wall_height_module, {
        "r_H_inv": H_inv_num_t,
        "r_sort_den": den_over_cos,
        "r_C": 0.0,
        "angle_offset": 0.0,
    })
    with torch.no_grad():
        out = wall_height_module(inputs)[0]
    wall_height = out[0].item()
    # Compiled approximation has breakpoint-grid error; 3 pixels is plenty of slack.
    assert abs(wall_height - expected_h) < 3.0, (
        f"H_inv={H_inv_num_t}, den/cos={den_over_cos}: "
        f"wall_height={wall_height:+.2f}, expected {expected_h:+.2f}"
    )


# ---------------------------------------------------------------------------
# Texture column
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "KNOWN ISSUE: the texture u-coordinate is computed as "
        "piecewise_linear_2d(abs_nuc, abs_den, doc_bp, doc_bp, n/d lambda), "
        "which bilinearly interpolates the division n/d over a 16x16 grid. "
        "Division isn't bilinear, so the approximation drifts significantly "
        "across the domain: at (abs_nuc=5, abs_den=10) u_raw=0.31 (expected 0.50); "
        "at (abs_nuc=2.5, abs_den=10) u_raw=0.07 (expected 0.25). Needs a "
        "different decomposition (e.g. reciprocal + signed_multiply) to fix."
    ),
    strict=False,
)
@pytest.mark.parametrize("D,sort_den,expected_u_frac", [
    (5.0, 10.0, 0.5),
    (2.5, 10.0, 0.25),
    (7.5, 10.0, 0.75),
    (0.0, 1.0, 0.0),    # edge: abs_nuc=0
    (0.9, 1.0, 0.9),    # edge: abs_nuc ≈ abs_den
])
def test_texture_column_division_known_bug(
    tex_col_module, D, sort_den, expected_u_frac,
):
    """tex_col_idx SHOULD track D/sort_den but the PL2D approximation drifts."""
    inputs = _pack(tex_col_module, {
        "r_D": D, "r_E": 0.0,
        "r_sort_den": sort_den, "r_C": 0.0,
        "angle_offset": 0.0,
    })
    with torch.no_grad():
        out = tex_col_module(inputs)[0]
    tex_col_idx = out[0].item()
    tex_w = 8
    expected_col = min(int(expected_u_frac * tex_w), tex_w - 1)
    assert abs(tex_col_idx - expected_col) <= 1.0, (
        f"D={D}, sort_den={sort_den}: tex_col_idx={tex_col_idx:+.2f}, "
        f"expected col≈{expected_col}"
    )
