"""Unit tests for ``_endpoint_to_column`` in the SORTED stage.

Tests the fused-2D column projection in isolation (no prefill, no sort,
no render) by compiling a minimal graph that takes two scalar inputs
(``cross``, ``dot``) and emits the projected screen column.  The
compiled output is compared to an exact oracle for a battery of
(cross, dot) cases — in-front, behind, near-axis, oblique.

This file exists because the projection's precision can't be debugged
efficiently via the end-to-end renderer tests: those compile the full
game graph (~35s per run on Modal) and surface a precision regression
as "wrong pixel color," which hides where the column error actually
originates.  Here each case runs locally in ~seconds.
"""

import math

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

from torchwright.doom.stages.wall import _endpoint_to_column

_MAX_COORD = 10.0
_W = 16
_FOV = 16


@pytest.fixture(scope="module")
def endpoint_module():
    """Compile a graph with a single ``_endpoint_to_column`` call.

    Two dummy tokens — position 0 holds the (cross, dot) under test,
    position 1 exists only so the sequence isn't degenerate.
    """
    pos = create_pos_encoding()
    cross = create_input("cross", 1)
    dot = create_input("dot", 1)
    col = _endpoint_to_column(
        cross,
        dot,
        W=_W,
        fov=_FOV,
        max_coord=_MAX_COORD,
        suffix="probe",
    )
    return compile_headless(
        Concatenate([col]),
        pos,
        d=512,
        d_head=32,
        max_layers=20,
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


def _oracle_column(cross: float, dot: float) -> float:
    """Exact reference for the fused projection.

    Matches the behaviour of the pre-refactor ``_endpoint_to_column``:
    ``dot > 0`` projects via the standard tan mapping; ``dot < 0``
    reflects across screen centre (``W − col_front``).  The output is
    clamped to ``[-2, W+2]`` the same way the piecewise does.
    """
    fov_rad = float(_FOV) * math.pi / 128.0
    col_scale = _W / fov_rad
    half_W = _W / 2.0
    if dot > 0:
        col = math.atan(cross / dot) * col_scale + half_W
    elif dot < 0:
        col_front = math.atan(cross / (-dot)) * col_scale + half_W
        col = float(_W) - col_front
    else:
        col = half_W
    return max(-2.0, min(float(_W + 2), col))


# Core cases the fused projection must get right: in-front, behind,
# grid vertices, edge cases.  Tolerance 0.2 column.
_CASES = [
    # id, cross, dot
    ("straight_ahead", 0.0, 5.0),
    ("slight_right_near", 0.5, 5.0),
    ("slight_left_near", -0.5, 5.0),
    ("moderate_right", 2.0, 5.0),
    ("slight_right_far", 0.5, 10.0),
    ("failing_angle_100_endpoint_a", -0.693, 7.037),
    # Shared endpoint (5, 5) projected from player (1, -3) angle 50 —
    # the actual drifting case behind ``test_renders_off_center_oblique[1.0,-3.0,50]``.
    ("failing_oc50_shared_vertex", -1.07, 8.88),
    ("right_45_deg", 1.0, 1.0),
    ("left_45_deg", -1.0, 1.0),
    ("very_slight_right", 0.01, 5.0),
    ("almost_perpendicular_right", 5.0, 0.5),
    ("behind_slight_right", 0.5, -5.0),
    ("behind_slight_left", -0.5, -5.0),
    ("behind_center", 0.0, -5.0),
]


@pytest.mark.parametrize("name,cross,dot", _CASES)
def test_endpoint_projection_matches_oracle(endpoint_module, name, cross, dot):
    """Compiled column must match oracle within 0.2 column precision."""
    rows = [
        {"cross": cross, "dot": dot},
        # Second row present only so module has a non-degenerate sequence.
        {"cross": 0.0, "dot": 1.0},
    ]
    inputs = _pack(endpoint_module, rows)
    with torch.no_grad():
        out = endpoint_module(inputs)
    col_got = out[0, 0].item()
    col_expected = _oracle_column(cross, dot)
    err = abs(col_got - col_expected)
    assert err < 0.2, (
        f"{name}: cross={cross}, dot={dot}: "
        f"got col={col_got:.3f}, expected col={col_expected:.3f} "
        f"(err={err:.3f})"
    )


# Known precision residuals.  Both live at the ``|cross/dot|`` clamp
# boundary of the FOV where the true column is clamped to ``-2`` or
# ``W+2`` but the piecewise_linear_2d's least-squares fit doesn't
# exactly reproduce the clamped grid plateau — pinv's min-norm
# solution on the sum/diff hyperplane family drifts off the plateau
# by 0.4-1.8 columns.  These are the same residuals the old
# ``reciprocal → multiply_2d → piecewise_linear(atan)`` chain showed
# under ``test_game_graph.py`` (``test_renders_oblique_angle[20]``,
# ``test_renders_off_center_oblique[1.0,-3.0,50]``).  Kept as xfail
# so future precision work has a concrete target.
_KNOWN_RESIDUALS = [
    ("failing_angle_20_endpoint_b", 1.71, 4.7),
    ("failing_off_center_50_a", -2.77, 4.58),
]


@pytest.mark.parametrize("name,cross,dot", _KNOWN_RESIDUALS)
def test_endpoint_projection_known_residuals(endpoint_module, name, cross, dot):
    rows = [
        {"cross": cross, "dot": dot},
        {"cross": 0.0, "dot": 1.0},
    ]
    inputs = _pack(endpoint_module, rows)
    with torch.no_grad():
        out = endpoint_module(inputs)
    col_got = out[0, 0].item()
    col_expected = _oracle_column(cross, dot)
    err = abs(col_got - col_expected)
    assert err < 0.2, (
        f"{name}: got col={col_got:.3f}, expected col={col_expected:.3f} "
        f"(err={err:.3f})"
    )
