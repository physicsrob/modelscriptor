"""Unit tests for ``torchwright.ops.quantization``.

These ops are pure affine transforms — the graph-side portion of the
thinking-token boundary quantization.  The actual LSB rounding
happens in the host's ``uint16`` cast outside the graph, which the
tests below simulate explicitly with ``torch.round``.

Per-op reference-eval tests cover:

- endpoint correctness (``lo`` → ``0``, ``hi`` → ``n_levels-1``);
- the roundtrip identity ``dequantize(quantize(x)) == x`` when no host
  cast sits between (the ops are exact);
- the LSB granularity ``(hi - lo) / (2 · (n_levels - 1))`` when a
  simulated host round *does* sit between;
- a three-boundary accumulation estimate (~0.003 at the aggressive
  end of the resolved-position chain);
- parametric coverage of the established value-type mappings.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchwright.graph import InputNode
from torchwright.ops.quantization import (
    DEFAULT_N_LEVELS,
    dequantize_from_range,
    quantize_to_range,
)

# Value-type table: per-name (lo, hi) range and the expected
# 16-bit resolution (LSB step) those bounds imply.
#                       (lo,   hi)     expected_resolution
DESIGN_TABLE: list[tuple[str, float, float, float]] = [
    ("cross_dot_a_b", -40.0, 40.0, 0.0012),
    ("t_lo_t_hi", 0.0, 1.0, 0.000015),
    ("vis_lo_vis_hi", -2.0, 122.0, 0.0019),
    ("resolved_x_y", -20.0, 20.0, 0.0006),
]


def _compute(node, **inputs):
    n_pos = next(iter(inputs.values())).shape[0]
    return node.compute(n_pos=n_pos, input_values=inputs)


# ---------------------------------------------------------------------------
# Endpoint correctness
# ---------------------------------------------------------------------------


def test_quantize_endpoints_maps_to_zero_and_max():
    value = InputNode("v", 1, value_range=(-40.0, 40.0))
    q = quantize_to_range(value, lo=-40.0, hi=40.0)
    result = _compute(q, v=torch.tensor([[-40.0], [40.0]]))
    assert abs(result[0].item() - 0.0) < 1e-3, f"lo: {result[0]}"
    assert abs(result[1].item() - (DEFAULT_N_LEVELS - 1)) < 1e-2, f"hi: {result[1]}"


def test_quantize_midpoint_is_half_scale():
    value = InputNode("v", 1, value_range=(-40.0, 40.0))
    q = quantize_to_range(value, lo=-40.0, hi=40.0)
    result = _compute(q, v=torch.tensor([[0.0]]))
    # Midpoint of [0, n_levels-1] is (n_levels-1)/2.
    expected = (DEFAULT_N_LEVELS - 1) / 2.0
    assert abs(result[0].item() - expected) < 1e-2, f"got {result[0]}"


def test_dequantize_endpoints_recover_lo_and_hi():
    q = InputNode("q", 1, value_range=(0.0, DEFAULT_N_LEVELS - 1))
    v = dequantize_from_range(q, lo=-40.0, hi=40.0)
    result = _compute(v, q=torch.tensor([[0.0], [float(DEFAULT_N_LEVELS - 1)]]))
    assert abs(result[0].item() - (-40.0)) < 1e-3
    assert abs(result[1].item() - 40.0) < 1e-3


# ---------------------------------------------------------------------------
# Exact roundtrip (no host cast) — the ops are affine inverses
# ---------------------------------------------------------------------------


def test_quantize_dequantize_is_identity_without_host_cast():
    """Without a rounding step between, quantize then dequantize is exact."""
    value = InputNode("v", 1, value_range=(-40.0, 40.0))
    q = quantize_to_range(value, lo=-40.0, hi=40.0)

    q_in = InputNode("q", 1, value_range=(0.0, DEFAULT_N_LEVELS - 1))
    v_out = dequantize_from_range(q_in, lo=-40.0, hi=40.0)

    samples = torch.linspace(-40.0, 40.0, steps=17).unsqueeze(1)
    q_vals = _compute(q, v=samples)
    recovered = _compute(v_out, q=q_vals)
    assert torch.allclose(
        recovered, samples, atol=1e-3
    ), f"max diff {(recovered - samples).abs().max().item()}"


# ---------------------------------------------------------------------------
# LSB granularity with a simulated host cast
# ---------------------------------------------------------------------------


def _simulate_host_cast(
    q_float: torch.Tensor, n_levels: int = DEFAULT_N_LEVELS
) -> torch.Tensor:
    """Model the host's uint16 cast: round to nearest integer, clamp to
    ``[0, n_levels - 1]``.  This is what the host runtime does between
    the producer's quantize and the consumer's dequantize."""
    return torch.round(q_float).clamp(0.0, float(n_levels - 1))


@pytest.mark.parametrize(
    "name,lo,hi,expected_resolution",
    DESIGN_TABLE,
    ids=[row[0] for row in DESIGN_TABLE],
)
def test_roundtrip_with_host_cast_matches_design_table(
    name: str, lo: float, hi: float, expected_resolution: float
) -> None:
    """A full producer→host→consumer round-trip has max error equal to
    the LSB half-step ``(hi - lo) / (2 · (n_levels - 1))``, which
    matches the per-name expected_resolution in DESIGN_TABLE."""
    value = InputNode("v", 1, value_range=(lo, hi))
    q_node = quantize_to_range(value, lo=lo, hi=hi)

    q_in = InputNode("q", 1, value_range=(0.0, DEFAULT_N_LEVELS - 1))
    v_out = dequantize_from_range(q_in, lo=lo, hi=hi)

    # Uniform samples across the full range, well away from exact-LSB
    # boundaries so we measure max *interior* error.
    n_samples = 4096
    samples = torch.rand((n_samples, 1)) * (hi - lo) + lo

    q_vals = _compute(q_node, v=samples)
    q_cast = _simulate_host_cast(q_vals)
    recovered = _compute(v_out, q=q_cast)
    max_err = (recovered - samples).abs().max().item()

    # LSB step = spacing between adjacent quantization levels.
    # LSB half-step = worst-case roundtrip error for a uniform random sample.
    lsb_step = (hi - lo) / (DEFAULT_N_LEVELS - 1)
    lsb_half = lsb_step / 2.0
    assert max_err <= 1.05 * lsb_half, (
        f"{name}: max roundtrip error {max_err:.2e} exceeds LSB half-step "
        f"{lsb_half:.2e}"
    )
    # Cross-check expected_resolution: this column is the LSB step
    # (level spacing), not the worst-case error.  Round-tripped errors
    # fall in ``[0, lsb_step / 2]``.
    assert math.isclose(lsb_step, expected_resolution, rel_tol=0.05), (
        f"{name}: LSB step {lsb_step:.6e} disagrees with expected "
        f"resolution {expected_resolution:.6e}"
    )


# ---------------------------------------------------------------------------
# Stacked boundaries — the aggregate budget the design cites
# ---------------------------------------------------------------------------


def test_three_stacked_boundaries_stay_within_design_budget():
    """The design doc estimates ~0.003 column accumulated error through
    a chain of rotation → clip → projection, which crosses three
    quantization boundaries at ~0.001 each.  This test stacks three
    full roundtrip cycles on the [-40, 40] (cross/dot) range and
    confirms the accumulated max error stays within 3 · LSB_half."""
    lo, hi = -40.0, 40.0
    value = InputNode("v", 1, value_range=(lo, hi))
    q_node = quantize_to_range(value, lo=lo, hi=hi)

    q_in = InputNode("q", 1, value_range=(0.0, DEFAULT_N_LEVELS - 1))
    v_out = dequantize_from_range(q_in, lo=lo, hi=hi)

    n_samples = 2048
    current = torch.rand((n_samples, 1)) * (hi - lo) + lo
    original = current.clone()

    for _ in range(3):
        q_vals = _compute(q_node, v=current)
        q_cast = _simulate_host_cast(q_vals)
        current = _compute(v_out, q=q_cast)

    max_err = (current - original).abs().max().item()
    lsb_half = (hi - lo) / (2.0 * (DEFAULT_N_LEVELS - 1))
    # After three roundtrips, accumulated error is bounded by 3 · LSB_half.
    # In practice it's often much less (errors partially cancel), but the
    # strict upper bound is 3 · LSB_half.
    assert max_err <= 3.0 * lsb_half, (
        f"3-stack accumulated error {max_err:.2e} exceeds "
        f"3 · LSB_half = {3 * lsb_half:.2e}"
    )


# ---------------------------------------------------------------------------
# Integer value types — exact, no quantization artefact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi,n_levels",
    [
        (0, 7, 8),  # bsp_rank
        (0, 1, 2),  # is_renderable / hit_* (exact at 2 levels)
        (0, 255, 256),  # resolved_angle
    ],
)
def test_integer_value_types_are_exact(lo: int, hi: int, n_levels: int):
    """Integer-valued types use n_levels matching the exact integer range,
    so roundtrip is bit-exact after the host cast — no quantization
    artefact anywhere."""
    value = InputNode("v", 1, value_range=(float(lo), float(hi)))
    q_node = quantize_to_range(value, lo=float(lo), hi=float(hi), n_levels=n_levels)

    q_in = InputNode("q", 1, value_range=(0.0, float(n_levels - 1)))
    v_out = dequantize_from_range(q_in, lo=float(lo), hi=float(hi), n_levels=n_levels)

    samples = torch.arange(lo, hi + 1, dtype=torch.float32).unsqueeze(1)
    q_vals = _compute(q_node, v=samples)
    q_cast = _simulate_host_cast(q_vals, n_levels=n_levels)
    recovered = _compute(v_out, q=q_cast)
    assert torch.allclose(recovered, samples, atol=1e-5), (
        f"integer roundtrip drift on [{lo}, {hi}] at n_levels={n_levels}: "
        f"max diff {(recovered - samples).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_quantize_rejects_bad_bounds():
    value = InputNode("v", 1, value_range=(-1.0, 1.0))
    with pytest.raises(AssertionError, match="hi > lo"):
        quantize_to_range(value, lo=1.0, hi=1.0)
    with pytest.raises(AssertionError, match="hi > lo"):
        quantize_to_range(value, lo=2.0, hi=1.0)


def test_quantize_rejects_nonscalar():
    value = InputNode("v", 3, value_range=(-1.0, 1.0))
    with pytest.raises(AssertionError, match="1D scalar"):
        quantize_to_range(value, lo=-1.0, hi=1.0)


def test_quantize_rejects_tiny_n_levels():
    value = InputNode("v", 1, value_range=(-1.0, 1.0))
    with pytest.raises(AssertionError, match="n_levels"):
        quantize_to_range(value, lo=-1.0, hi=1.0, n_levels=1)
