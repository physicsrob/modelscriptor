"""Unit tests for ``torchwright.doom.stages._normalize.normalize_coord``.

Builds a minimal graph that runs ``normalize_coord`` against a coupled
``(coord, inv_scale)`` distribution drawn from the operating range
(|coord| in [0.1, 100], inv_scale in [1/100, 1], coupled by
|coord · inv_scale| ≤ 1).  Validates:

1. Worst-case relative error < 0.5 % across ~1000 realistic samples.
2. Mean relative error < 0.1 %.
3. No NaN / Inf at boundary inputs (coord ≈ 0, coord at ±max_abs,
   inv_scale at boundaries).
4. ``coord = 0`` produces output ≈ 0 even though ``log_abs`` clamps to
   ``log(min_abs) ≠ 0`` — the sign-mux subtraction cancels the
   non-zero ``abs_normalized`` between its two cond_gates.
"""

import math

import pytest
import torch

from torchwright.doom.stages._normalize import (
    DEFAULT_MAX_ABS,
    DEFAULT_MIN_ABS,
    normalize_coord,
)
from torchwright.ops.inout_nodes import create_input


# ---------------------------------------------------------------------------
# Distribution and graph builders
# ---------------------------------------------------------------------------


_TEST_COORD_ABS_MIN = 0.1
"""Test-sampling floor on |coord|.

Set strictly above ``DEFAULT_MIN_ABS`` so the precision test exercises
the typical operating regime (|coord| ≥ 0.1, where ``log_abs`` is in
its accurate piecewise zone) rather than the floor where the bias at
coord=0 dominates relative error for tiny expected products.  The
floor's behaviour is tested separately by
``test_normalize_coord_at_zero_bounded_residue``.
"""


def _operating_distribution(n_samples: int = 1024, seed: int = 0):
    """Coupled (coord, inv_scale) samples obeying |coord · inv_scale| ≤ 1.

    Sample inv_scale log-uniform in ``[1/max_abs, 1]`` and coord uniform
    in ``[-1/inv_scale, +1/inv_scale]``, then floor |coord| at
    ``_TEST_COORD_ABS_MIN`` (chosen above ``DEFAULT_MIN_ABS`` so we
    exercise the typical operating regime rather than the floor; the
    floor's residue bound is tested separately).  Returns a dict of
    input tensors plus the precomputed log_inv_scale tensor (which the
    production graph computes once via ``_compute_scale_find``; the
    test treats it as a precomputed input so we don't depend on the
    scale-find pass's own noise).
    """
    gen = torch.Generator().manual_seed(seed)
    log_lo = math.log(1.0 / DEFAULT_MAX_ABS)
    log_hi = math.log(1.0)

    log_inv = torch.rand((n_samples, 1), generator=gen) * (log_hi - log_lo) + log_lo
    inv_scale = torch.exp(log_inv)
    gmac = 1.0 / inv_scale
    coord_unscaled = torch.rand((n_samples, 1), generator=gen) * 2.0 - 1.0
    coord = coord_unscaled * gmac
    coord_sign = torch.where(coord >= 0, torch.ones_like(coord), -torch.ones_like(coord))
    coord_abs = coord.abs().clamp(min=_TEST_COORD_ABS_MIN, max=DEFAULT_MAX_ABS)
    coord = coord_sign * coord_abs

    return {
        "coord": coord,
        "log_inv_scale": torch.log(inv_scale),
        "_inv_scale_for_reference": inv_scale,
    }


def _build_graph():
    coord = create_input("coord", 1)
    log_inv_scale = create_input("log_inv_scale", 1)
    out = normalize_coord(coord, log_inv_scale)
    return out


def _run(out_node, n_pos: int, **inputs):
    return out_node.compute(n_pos=n_pos, input_values=inputs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normalize_coord_precision_on_operating_range():
    """End-to-end accuracy across ~1000 coupled samples on the manifold.

    Acceptance bars come from the Phase 0 measurement (max ~0.07 %,
    p99 ~0.04 % at 256 BPs).  We assert generously above the measured
    floor so the test isn't brittle to float32 cross-platform variation.
    """
    samples = _operating_distribution(n_samples=1024)
    out = _build_graph()

    n_pos = samples["coord"].shape[0]
    result = _run(
        out,
        n_pos=n_pos,
        coord=samples["coord"],
        log_inv_scale=samples["log_inv_scale"],
    )

    expected = samples["coord"] * samples["_inv_scale_for_reference"]
    abs_err = (result - expected).abs()
    # Restrict relative-error stats to samples whose expected value has
    # decent magnitude — relative error is meaningless near zero.
    big_enough = expected.abs() >= 1e-3
    if big_enough.any():
        rel_err = abs_err[big_enough] / expected.abs()[big_enough]
        max_rel = float(rel_err.max().item())
        mean_rel = float(rel_err.mean().item())
    else:
        max_rel = mean_rel = 0.0
    max_abs = float(abs_err.max().item())

    assert max_rel < 5e-3, (
        f"max relative error {max_rel:.4g} exceeds 0.5 % budget "
        f"({mean_rel=:.4g}, {max_abs=:.4g})"
    )
    assert mean_rel < 1e-3, f"mean relative error {mean_rel:.4g} exceeds 0.1 %"


def test_normalize_coord_no_nan_or_inf_at_boundaries():
    """Boundary inputs must not produce NaN/Inf.

    Cases: coord = 0, coord at +max_abs, coord at -max_abs, inv_scale at
    its lo/hi bounds.  Independent of whether the math at coord=0 lands
    exactly on zero — that's covered by the dedicated test below.
    """
    out = _build_graph()
    coord_vals = torch.tensor(
        [
            [0.0],
            [DEFAULT_MAX_ABS],
            [-DEFAULT_MAX_ABS],
            [DEFAULT_MIN_ABS],
            [-DEFAULT_MIN_ABS],
            [0.5],
            [50.0],
        ]
    )
    inv_scale_vals = torch.tensor(
        [
            [1.0],  # smallest scale
            [1.0 / DEFAULT_MAX_ABS],  # largest scale
            [1.0 / DEFAULT_MAX_ABS],
            [1.0 / DEFAULT_MAX_ABS],
            [1.0 / DEFAULT_MAX_ABS],
            [0.1],
            [0.05],
        ]
    )
    log_inv_scale_vals = torch.log(inv_scale_vals)
    n_pos = coord_vals.shape[0]
    result = _run(
        out, n_pos=n_pos, coord=coord_vals, log_inv_scale=log_inv_scale_vals
    )
    assert torch.isfinite(result).all(), f"non-finite values: {result}"


def test_normalize_coord_at_zero_bounded_residue():
    """coord = 0 produces a bounded residue, not exactly zero.

    Mechanism: ``log_abs(0)`` clamps to ``log(min_abs)``, so
    ``abs_normalized = min_abs · inv_scale`` (nonzero).  The sign-mux
    via two ``cond_gate(approximate=False)`` calls is asymmetric at
    the compare threshold — ``compare(0, 0)`` returns ``false_level
    = -1`` (strictly-greater convention), so ``pos_part = 0`` and
    ``neg_part = abs_normalized``, giving ``result = -abs_normalized
    = -min_abs · inv_scale``.

    This is a known design tradeoff: lowering ``min_abs`` further
    pushes ``log_abs`` out of its single-sublayer fast path, and
    adding a magnitude gate costs additional sublayers.  For our
    operating envelope (max_coord = 100, smallest realistic scene
    scale ~5), the residue stays below ~0.2 % of the normalized
    [-1, 1] range — within the per-multiply precision budget.

    This test pins the *bound*, not exact zero: ``|result| ≤
    min_abs · inv_scale + slack``.
    """
    out = _build_graph()
    inv_scale_vals = torch.tensor([[1.0], [0.5], [0.1], [1.0 / DEFAULT_MAX_ABS]])
    coord_vals = torch.zeros_like(inv_scale_vals)
    log_inv_scale_vals = torch.log(inv_scale_vals)
    n_pos = coord_vals.shape[0]
    result = _run(
        out, n_pos=n_pos, coord=coord_vals, log_inv_scale=log_inv_scale_vals
    )

    # Per-row expected residue magnitude bound.  Add a small slack to
    # absorb log_abs/exp's measured noise (≤ 5e-4 typical) plus
    # float32 round-off at the multiplications.
    expected_bound = (DEFAULT_MIN_ABS * inv_scale_vals).abs() * 1.5 + 5e-4
    actual_abs = result.abs()
    assert (actual_abs <= expected_bound).all(), (
        f"|result| at coord=0 exceeds min_abs·inv_scale bound. "
        f"got {actual_abs.flatten().tolist()}, "
        f"bound {expected_bound.flatten().tolist()}"
    )


def test_normalize_coord_sign_correctness():
    """Sign of normalize_coord(coord, log_inv_scale) matches sign of coord."""
    out = _build_graph()
    coord_vals = torch.tensor(
        [
            [50.0],
            [-50.0],
            [10.0],
            [-10.0],
            [0.5],
            [-0.5],
        ]
    )
    inv_scale_vals = torch.full_like(coord_vals, 0.05)  # gives products in ±2.5
    log_inv_scale_vals = torch.log(inv_scale_vals)
    n_pos = coord_vals.shape[0]
    result = _run(
        out, n_pos=n_pos, coord=coord_vals, log_inv_scale=log_inv_scale_vals
    )
    expected_sign = torch.sign(coord_vals)
    actual_sign = torch.sign(result)
    assert torch.equal(expected_sign, actual_sign), (
        f"sign mismatch: coord_sign={expected_sign.flatten().tolist()}, "
        f"result_sign={actual_sign.flatten().tolist()}, result={result.flatten().tolist()}"
    )


@pytest.mark.parametrize(
    "coord, inv_scale, expected",
    [
        (50.0, 0.02, 1.0),  # exactly at +max product
        (-50.0, 0.02, -1.0),  # exactly at -max product
        (1.0, 0.5, 0.5),
        (-1.0, 0.5, -0.5),
        (10.0, 0.05, 0.5),
        (25.0, 0.02, 0.5),
    ],
)
def test_normalize_coord_specific_values(coord, inv_scale, expected):
    """Spot checks on hand-picked (coord, inv_scale, expected_product)."""
    out = _build_graph()
    coord_t = torch.tensor([[float(coord)]])
    inv_scale_t = torch.tensor([[float(inv_scale)]])
    log_inv_scale_t = torch.log(inv_scale_t)
    result = _run(
        out, n_pos=1, coord=coord_t, log_inv_scale=log_inv_scale_t
    )
    actual = float(result[0, 0].item())
    assert abs(actual - expected) < 5e-3, (
        f"coord={coord}, inv_scale={inv_scale}: expected {expected}, "
        f"got {actual} (diff {actual - expected:.4g})"
    )
