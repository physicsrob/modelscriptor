"""Unit tests for individual pieces of the visibility mask pipeline.

Tests each component in isolation to find where the bug is:
1. atan2 approximation via piecewise_linear_2d
2. Relative angle computation (subtraction + mod 256)
3. Column index from relative angle
4. in_range mask generation
5. Render attention score computation
"""

import math

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate, Linear, Node
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    compare,
    mod_const,
    multiply_const,
    piecewise_linear_2d,
    subtract,
)
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_pos_encoding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import in_range, select

# ---------------------------------------------------------------------------
# 1. atan2 approximation accuracy
# ---------------------------------------------------------------------------


class TestCrossDotColumnIndex:
    """Test the cross/dot decomposition used to compute column indices
    for wall endpoint projection (replaces the broken atan2 approach)."""

    @pytest.fixture(scope="class")
    def col_module(self):
        """Compile: (dx, dy, cos_pa, sin_pa) → column index."""
        from torchwright.ops.arithmetic_ops import (
            abs,
            clamp,
            negate,
            piecewise_linear,
            reciprocal,
            signed_multiply,
        )

        _TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
        _DIFF_BP = [
            -40,
            -30,
            -20,
            -15,
            -10,
            -7,
            -5,
            -3,
            -2,
            -1,
            -0.5,
            0,
            0.5,
            1,
            2,
            3,
            5,
            7,
            10,
            15,
            20,
            30,
            40,
        ]
        _ATAN_BP = [
            -20,
            -10,
            -5,
            -3,
            -2,
            -1.5,
            -1,
            -0.75,
            -0.5,
            -0.25,
            0,
            0.25,
            0.5,
            0.75,
            1,
            1.5,
            2,
            3,
            5,
            10,
            20,
        ]
        W, fov = 32, 8
        fov_rad = fov * math.pi / 128.0

        pos = create_pos_encoding()
        dx = create_input("dx", 1, value_range=(-20.0, 20.0))
        dy = create_input("dy", 1, value_range=(-20.0, 20.0))
        cos_pa = create_input("cos_pa", 1, value_range=(-1.0, 1.0))
        sin_pa = create_input("sin_pa", 1, value_range=(-1.0, 1.0))

        cross = subtract(
            piecewise_linear_2d(
                cos_pa, dy, _TRIG_BP, _DIFF_BP, lambda a, b: a * b, name="cd"
            ),
            piecewise_linear_2d(
                sin_pa, dx, _TRIG_BP, _DIFF_BP, lambda a, b: a * b, name="sd"
            ),
        )
        dot = add(
            piecewise_linear_2d(
                cos_pa, dx, _TRIG_BP, _DIFF_BP, lambda a, b: a * b, name="cx"
            ),
            piecewise_linear_2d(
                sin_pa, dy, _TRIG_BP, _DIFF_BP, lambda a, b: a * b, name="sy"
            ),
        )

        dot_sign = compare(dot, 0.0)
        dot_abs = abs(dot)
        dot_clamped = select(
            compare(dot_abs, 0.1),
            dot_abs,
            create_literal_value(torch.tensor([0.1]), name="dmin"),
        )
        inv_dot = reciprocal(dot_clamped, min_value=0.1, max_value=40.0)
        signed_inv = select(dot_sign, inv_dot, negate(inv_dot))
        tan_rel = signed_multiply(
            cross,
            signed_inv,
            max_abs1=20.0,
            max_abs2=10.0,
            step=0.5,
            max_abs_output=20.0,
        )
        col = piecewise_linear(
            tan_rel,
            _ATAN_BP,
            lambda t: math.atan(t) * W / fov_rad + W / 2.0,
            name="col",
        )
        output = Concatenate([cross, dot, col])
        return compile_headless(
            output,
            pos,
            d=1024,
            d_head=16,
            max_layers=50,
            verbose=False,
        )

    def _expected(self, dx, dy, cos_pa, sin_pa, W=32, fov=8):
        cross = cos_pa * dy - sin_pa * dx
        dot = cos_pa * dx + sin_pa * dy
        tan_rel = cross / dot if abs(dot) > 0.01 else 999.0
        fov_rad = fov * math.pi / 128.0
        col = math.atan(tan_rel) * W / fov_rad + W / 2.0
        return cross, dot, col

    @pytest.mark.parametrize(
        "dx,dy,pa_deg,desc",
        [
            (5, -5, 0, "east-A from center facing east"),
            (5, 5, 0, "east-B from center facing east"),
            (2, -5, 0, "east-A from (3,0) facing east"),
            (2, 5, 0, "east-B from (3,0) facing east"),
            (-5, 5, 90, "north-A from center facing north"),
            (5, 5, 90, "north-B from center facing north"),
            (-8, 5, 0, "north-A from (3,0) facing east"),
        ],
    )
    def test_column_index(self, col_module, dx, dy, pa_deg, desc):
        pa_rad = pa_deg * math.pi / 180.0
        cos_pa = math.cos(pa_rad)
        sin_pa = math.sin(pa_rad)
        exp_cross, exp_dot, exp_col = self._expected(dx, dy, cos_pa, sin_pa)

        # compile_headless sorts inputs alphabetically: [cos_pa, dx, dy, sin_pa]
        inputs = torch.tensor([[cos_pa, float(dx), float(dy), sin_pa]])
        with torch.no_grad():
            out = col_module(inputs)[0]
        got_cross = out[0].item()
        got_dot = out[1].item()
        got_col = out[2].item()

        assert (
            abs(got_cross - exp_cross) < 1.0
        ), f"cross err: {got_cross:.2f} vs {exp_cross:.2f} ({desc})"
        assert (
            abs(got_dot - exp_dot) < 1.0
        ), f"dot err: {got_dot:.2f} vs {exp_dot:.2f} ({desc})"
        if abs(exp_dot) > 0.5:  # Only check col when endpoint is in front
            assert (
                abs(got_col - exp_col) < 3.0
            ), f"col err: {got_col:.1f} vs {exp_col:.1f} ({desc})"


# ---------------------------------------------------------------------------
# 2. Relative angle + column index
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. in_range mask from column bounds
# ---------------------------------------------------------------------------


class TestInRangeMask:
    """Test that in_range(lo, hi, W) produces correct masks for
    the column ranges we expect from wall endpoint projections."""

    @pytest.fixture(scope="class")
    def mask_module(self):
        pos = create_pos_encoding()
        lo = create_input("lo", 1, value_range=(-40.0, 40.0))
        hi = create_input("hi", 1, value_range=(-40.0, 40.0))
        mask = in_range(lo, hi, 32)
        output = Concatenate([mask])
        return compile_headless(
            output,
            pos,
            d=2048,
            d_head=16,
            max_layers=20,
            verbose=False,
        )

    def test_full_coverage(self, mask_module):
        """lo=-1, hi=33: all 32 columns should be +1."""
        # compile_headless sorts inputs alphabetically: [hi, lo]
        inputs = torch.tensor([[33.0, -1.0]])
        with torch.no_grad():
            mask = mask_module(inputs)[0].numpy()
        assert (mask > 0.5).all(), f"Expected all visible, got {mask}"

    def test_no_coverage_above(self, mask_module):
        """lo=33, hi=40: all columns outside [0,32), should be -1."""
        inputs = torch.tensor([[40.0, 33.0]])
        with torch.no_grad():
            mask = mask_module(inputs)[0].numpy()
        assert (mask < -0.5).all(), f"Expected all hidden, got {mask}"

    def test_no_coverage_below(self, mask_module):
        """lo=-5, hi=-1: all columns outside [0,32), should be -1."""
        inputs = torch.tensor([[-1.0, -5.0]])
        with torch.no_grad():
            mask = mask_module(inputs)[0].numpy()
        assert (mask < -0.5).all(), f"Expected all hidden, got {mask}"

    def test_partial_coverage(self, mask_module):
        """lo=10, hi=20: columns 10-19 visible, rest hidden."""
        inputs = torch.tensor([[20.0, 10.0]])
        with torch.no_grad():
            mask = mask_module(inputs)[0].numpy()
        for c in range(32):
            expected = 1.0 if 10 <= c + 0.5 < 20 else -1.0
            assert (
                abs(mask[c] - expected) < 0.5
            ), f"col {c}: expected {expected}, got {mask[c]:.2f}"

    def test_behind_player(self, mask_module):
        """lo=-2, hi=-1: all behind, should be -1."""
        inputs = torch.tensor([[-1.0, -2.0]])
        with torch.no_grad():
            mask = mask_module(inputs)[0].numpy()
        assert (mask < -0.5).all(), f"Expected all hidden, got {mask}"


# ---------------------------------------------------------------------------
# 4. Column one-hot via in_range
# ---------------------------------------------------------------------------


class TestColumnOneHot:
    """Test that in_range(col, col+1, W) produces a correct one-hot."""

    @pytest.fixture(scope="class")
    def onehot_module(self):
        pos = create_pos_encoding()
        col = create_input("col", 1, value_range=(0.0, 40.0))
        col_p1 = add_const(col, 1.0)
        oh = in_range(col, col_p1, 32)
        oh_01 = multiply_const(add_const(oh, 1.0), 0.5)
        output = Concatenate([oh_01])
        return compile_headless(
            output,
            pos,
            d=1024,
            d_head=16,
            max_layers=20,
            verbose=False,
        )

    @pytest.mark.parametrize("col", [0, 1, 15, 16, 30, 31])
    def test_onehot_at_column(self, onehot_module, col):
        inputs = torch.tensor([[float(col)]])
        with torch.no_grad():
            oh = onehot_module(inputs)[0].numpy()
        for c in range(32):
            expected = 1.0 if c == col else 0.0
            assert (
                abs(oh[c] - expected) < 0.5
            ), f"col_idx={col}, position {c}: expected {expected}, got {oh[c]:.2f}"


# ---------------------------------------------------------------------------
# 5. Dot product of one-hot with mask (what attention computes)
# ---------------------------------------------------------------------------


class TestVisibilityDotProduct:
    """Test that dot(col_onehot_01, vis_mask) extracts the correct
    visibility value at the given column."""

    def test_dot_product_math(self):
        """Pure tensor test: col_onehot_01 · vis_mask should extract vis[col]."""
        W = 32
        for col in [0, 15, 16, 31]:
            col_oh = torch.zeros(W)
            col_oh[col] = 1.0

            # Mask where columns 10-20 are visible
            vis = torch.ones(W) * -1.0
            vis[10:20] = 1.0

            dot = (col_oh * vis).sum().item()
            expected = vis[col].item()
            assert dot == expected, f"col={col}: dot={dot}, expected={expected}"

    def test_dot_with_scaling(self):
        """With VIS_GAIN=200, the score difference between visible and
        hidden should be 2*200=400."""
        VIS_GAIN = 200.0
        W = 32
        col = 15  # visible (in range 10-20)

        col_oh = torch.zeros(W)
        col_oh[col] = 1.0

        vis_visible = torch.ones(W) * -1.0
        vis_visible[10:20] = 1.0
        score_vis = (VIS_GAIN * col_oh * vis_visible).sum().item()

        vis_hidden = torch.ones(W) * -1.0
        vis_hidden[25:30] = 1.0  # visible at different columns
        score_hid = (VIS_GAIN * col_oh * vis_hidden).sum().item()

        diff = score_vis - score_hid
        assert diff == 400.0, f"Score difference should be 400, got {diff}"
