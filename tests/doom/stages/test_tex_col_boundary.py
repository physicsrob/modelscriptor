"""Regression tests for the tex_col thermometer boundary at u=0.5.

Verifies that _compute_texture_column correctly classifies pixels near the
u=k/tex_w boundaries — the exact cases that were wrong under the old
piecewise-linear division approximation.

Reference cases (all have C=0, sort_den=±10):
  angle=192, col=8  → D=+5, E=5, ao=0,  tan=0.000  → u=0.500 (boundary, tex_col=4)
  angle=192, col=9  → D=+5, E=5, ao=+1, tan=0.025  → u=0.512 (above,    tex_col=4)
  angle=192, col=7  → D=+5, E=5, ao=-1, tan=-0.025 → u=0.488 (below,    tex_col=3)
  angle=64,  col=9  → D=-5, E=5, ao=+1, tan=0.025  → u=0.488 (below,    tex_col=3)
"""

import builtins
import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.arithmetic_ops import (
    abs, add, bool_to_01, compare, multiply_const, piecewise_linear_2d, subtract,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.graph_constants import DIFF_BP
from torchwright.doom.stages.render import (
    _compute_angle_offset_tan,
    _compute_den_over_cos,
)


_MAX_COORD = 20.0
_TEX_W = 8
_FOV = 16        # matches test_game_graph.py (W=16, fov_columns=16)
_THRESH = -0.5   # current threshold in render.py


def _build_tex_col_graph(fov=_FOV, tex_w=_TEX_W, thresh=_THRESH):
    """Build a minimal graph that exposes tex_col and its intermediate values.

    Output columns (in order):
      0       tex_col         (comparison thermometer sum)
      1       abs_nuc         |D + E·tan|
      2       abs_den         |sort_den - C·tan|
      3       diff_k4         8·abs_nuc − 4·abs_den
      4..10   diff_k1..k7     8·abs_nuc − k·abs_den
    """
    pos = create_pos_encoding()
    r_D   = create_input("r_D",   1, value_range=(-100.0, 100.0))
    r_E   = create_input("r_E",   1, value_range=(-100.0, 100.0))
    r_sd  = create_input("r_sd",  1, value_range=(-100.0, 100.0))   # sort_den
    r_C   = create_input("r_C",   1, value_range=(-100.0, 100.0))
    r_ao  = create_input("r_ao",  1, value_range=(0.0, 255.0))   # angle_offset

    tan_o, tan_val_bp = _compute_angle_offset_tan(r_ao, fov=fov)
    _, abs_den = _compute_den_over_cos(r_sd, r_C, tan_o, tan_val_bp)

    E_tan = piecewise_linear_2d(
        r_E, tan_o, DIFF_BP, tan_val_bp,
        lambda a, b: a * b, name="E_tan_o",
    )
    abs_nuc = abs(add(r_D, E_tan))

    nuc_scaled = multiply_const(abs_nuc, float(tex_w))

    bits = []
    diffs = []
    for k in range(1, tex_w):
        k_den = multiply_const(abs_den, float(k))
        diff = subtract(nuc_scaled, k_den)
        diffs.append(diff)
        bits.append(bool_to_01(compare(diff, thresh)))

    tex_col = bits[0]
    for b in bits[1:]:
        tex_col = add(tex_col, b)

    diff_k4 = diffs[3]   # k=4 is index 3 in diffs (0-indexed)

    outputs = Concatenate([tex_col, abs_nuc, abs_den, diff_k4] + diffs)
    return outputs, pos


@pytest.fixture(scope="module", params=[1024, 2048])
def diag_module(request):
    d = request.param
    outputs, pos = _build_tex_col_graph()
    return d, compile_headless(
        outputs, pos,
        d=d, d_head=32, max_layers=50, verbose=False,
    )


def _pack(module, r_D, r_E, r_sd, r_C, r_ao):
    specs = {name: (start, width) for name, start, width in module._input_specs}
    d_input = max(s + w for _, s, w in module._input_specs)
    row = torch.zeros(1, d_input, dtype=torch.float32)
    for name, val in [("r_D", r_D), ("r_E", r_E), ("r_sd", r_sd),
                      ("r_C", r_C), ("r_ao", r_ao)]:
        s, w = specs[name]
        row[0, s:s + w] = val
    return row


def _run(module, r_D, r_E, r_sd, r_C, r_ao):
    inp = _pack(module, r_D, r_E, r_sd, r_C, r_ao)
    with torch.no_grad():
        out = module(inp)[0]
    return out.tolist()


@pytest.mark.parametrize("label,r_D,r_E,r_sd,r_C,r_ao,expected_col", [
    # angle=192 south wall: D=+5, E=5, sort_den=10, C=0
    ("192/col8  (boundary u=0.5)", 5.0, 5.0, 10.0, 0.0, 0.0, 4),
    ("192/col9  (above    u≈0.51)", 5.0, 5.0, 10.0, 0.0, 1.0, 4),
    ("192/col7  (below    u≈0.49)", 5.0, 5.0, 10.0, 0.0, -1.0, 3),
    # angle=64 north wall: D=-5, E=5, sort_den=-10, C=0
    ("64/col9   (below    u≈0.49)", -5.0, 5.0, -10.0, 0.0, 1.0, 3),
    # angle=0  east wall: D=0, E≈0, sort_den=5  — trivially tex_col=0
    ("0/col8    (u=0.0)", 0.0, 0.0, 5.0, 0.0, 0.0, 0),
])
def test_tex_col_boundary(diag_module, label, r_D, r_E, r_sd, r_C, r_ao, expected_col):
    d, module = diag_module

    tan_exact = math.tan(r_ao * 2 * math.pi / 256.0)
    exact_abs_nuc = math.fabs(r_D + r_E * tan_exact)
    exact_abs_den = math.fabs(r_sd)
    exact_diff_k4 = 8 * exact_abs_nuc - 4 * exact_abs_den

    out = _run(module, r_D, r_E, r_sd, r_C, r_ao)
    tex_col   = out[0]
    abs_nuc_v = out[1]
    abs_den_v = out[2]
    diff_k4_v = out[3]
    diffs     = out[4:]    # diffs[0]=k1, diffs[3]=k4, ...

    print(f"\n[d={d}] {label}")
    print(f"  exact:    abs_nuc={exact_abs_nuc:.4f}  abs_den={exact_abs_den:.4f}"
          f"  diff_k4={exact_diff_k4:+.4f}  u={exact_abs_nuc/exact_abs_den:.4f}"
          f"  tex_col={expected_col}")
    print(f"  compiled: abs_nuc={abs_nuc_v:.4f}  abs_den={abs_den_v:.4f}"
          f"  diff_k4={diff_k4_v:+.4f}  tex_col={tex_col:.2f}")
    print(f"  diffs k1..k7: {[f'{x:+.3f}' for x in diffs]}")

    assert builtins.abs(tex_col - expected_col) <= 0.5, (
        f"d={d} {label}: compiled tex_col={tex_col:.2f}, expected {expected_col}"
    )
