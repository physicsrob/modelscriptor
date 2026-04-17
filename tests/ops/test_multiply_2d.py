"""Unit tests for multiply_2d.

Verifies that the 2D piecewise-linear multiplication primitive produces
correct results across sign combinations, grid points, interpolated
values, custom breakpoints, unsigned ranges, output clamping, and
compiled transformer agreement.
"""

import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.ops.arithmetic_ops import multiply_2d, signed_multiply
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_mul(node, a_val, b_val):
    """Evaluate a multiply_2d node at (a, b)."""
    return node.compute(
        n_pos=1,
        input_values={
            "a": torch.tensor([[a_val]]),
            "b": torch.tensor([[b_val]]),
        },
    ).item()


def _build_multiply_2d(**kwargs):
    """Build a multiply_2d graph from two scalar inputs named 'a' and 'b'."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    return multiply_2d(a, b, **kwargs)


# ---------------------------------------------------------------------------
# Grid-point accuracy
# ---------------------------------------------------------------------------


def test_multiply_2d_grid_points():
    """Exact at integer multiples of step."""
    node = _build_multiply_2d(max_abs1=5.0, max_abs2=5.0, step1=1.0, step2=1.0)

    for a in range(-5, 6):
        for b in range(-5, 6):
            expected = float(a * b)
            result = _eval_mul(node, float(a), float(b))
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


def test_multiply_2d_fine_step():
    """Finer step gives tighter accuracy at grid points."""
    node = _build_multiply_2d(max_abs1=3.0, max_abs2=3.0, step1=0.5, step2=0.5)

    for a in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        for b in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Sign combinations and zeros
# ---------------------------------------------------------------------------


def test_multiply_2d_sign_combinations():
    """All four quadrants plus zero produce correct results."""
    node = _build_multiply_2d(max_abs1=10.0, max_abs2=10.0, step1=1.0, step2=1.0)

    cases = [
        (3.0, 4.0, 12.0),
        (-3.0, 4.0, -12.0),
        (3.0, -4.0, -12.0),
        (-3.0, -4.0, 12.0),
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    for a, b, expected in cases:
        result = _eval_mul(node, a, b)
        assert abs(result - expected) < 0.5, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Interpolation accuracy
# ---------------------------------------------------------------------------


def test_multiply_2d_interpolation():
    """Between grid points, error is bounded by h1*h2/4."""
    step1, step2 = 1.0, 1.0
    node = _build_multiply_2d(max_abs1=10.0, max_abs2=10.0, step1=step1, step2=step2)
    max_error = step1 * step2 / 4 + 0.05  # small margin for float noise

    # Test at half-step offsets (worst case for interpolation)
    for a in [0.5, 1.5, -2.5, 4.5]:
        for b in [0.5, -1.5, 3.5]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < max_error, (
                f"{a}*{b} = {expected}, got {result}, "
                f"error {abs(result - expected):.4f} > {max_error:.4f}"
            )


# ---------------------------------------------------------------------------
# Custom breakpoints
# ---------------------------------------------------------------------------


def test_multiply_2d_custom_breakpoints():
    """Non-uniform breakpoints (DOOM _DIFF_BP style) work correctly."""
    diff_bp = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    trig_bp = [-1.0, -0.5, 0.0, 0.5, 1.0]
    node = _build_multiply_2d(
        max_abs1=10.0,
        max_abs2=1.0,
        breakpoints1=diff_bp,
        breakpoints2=trig_bp,
    )

    # Exact at grid points
    for a in diff_bp:
        for b in trig_bp:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Unsigned ranges (min1/min2)
# ---------------------------------------------------------------------------


def test_multiply_2d_unsigned():
    """min2=0 for positive-only second input (e.g., inv_range)."""
    node = _build_multiply_2d(
        max_abs1=10.0,
        max_abs2=2.0,
        step1=1.0,
        step2=0.25,
        min2=0.0,
    )

    # Positive second input
    for a in [-5.0, 0.0, 3.0, 10.0]:
        for b in [0.0, 0.5, 1.0, 2.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.3, f"{a}*{b} = {expected}, got {result}"


def test_multiply_2d_both_unsigned():
    """Both inputs non-negative."""
    node = _build_multiply_2d(
        max_abs1=5.0,
        max_abs2=5.0,
        step1=1.0,
        step2=1.0,
        min1=0.0,
        min2=0.0,
    )

    for a in [0.0, 1.0, 3.0, 5.0]:
        for b in [0.0, 1.0, 3.0, 5.0]:
            expected = a * b
            result = _eval_mul(node, a, b)
            assert abs(result - expected) < 0.01, f"{a}*{b} = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Output clamping
# ---------------------------------------------------------------------------


def test_multiply_2d_output_clamp():
    """max_abs_output caps the product."""
    node = _build_multiply_2d(
        max_abs1=10.0,
        max_abs2=10.0,
        step1=1.0,
        step2=1.0,
        max_abs_output=20.0,
    )

    # Within clamp range
    assert abs(_eval_mul(node, 3.0, 4.0) - 12.0) < 0.5

    # At clamp boundary — 5*5 = 25, clamped to 20
    result = _eval_mul(node, 5.0, 5.0)
    assert abs(result - 20.0) < 0.5, f"5*5 clamped to 20, got {result}"

    # Negative clamp — -5*5 = -25, clamped to -20
    result = _eval_mul(node, -5.0, 5.0)
    assert abs(result - (-20.0)) < 0.5, f"-5*5 clamped to -20, got {result}"

    # Well within range
    assert abs(_eval_mul(node, 2.0, 3.0) - 6.0) < 0.5


# ---------------------------------------------------------------------------
# Comparison with signed_multiply
# ---------------------------------------------------------------------------


def test_multiply_2d_vs_signed_multiply():
    """multiply_2d and signed_multiply agree on a sweep of inputs."""
    a_inp = create_input("a", 1)
    b_inp = create_input("b", 1)

    node_2d = multiply_2d(
        a_inp, b_inp, max_abs1=10.0, max_abs2=10.0, step1=1.0, step2=1.0
    )
    node_sm = signed_multiply(a_inp, b_inp, max_abs1=10.0, max_abs2=10.0, step=1.0)

    for a in [-7.0, -3.0, 0.0, 4.0, 8.0]:
        for b in [-6.0, -1.0, 0.0, 2.0, 9.0]:
            inputs = {
                "a": torch.tensor([[a]]),
                "b": torch.tensor([[b]]),
            }
            r_2d = node_2d.compute(n_pos=1, input_values=inputs).item()
            r_sm = node_sm.compute(n_pos=1, input_values=inputs).item()
            expected = a * b
            # Both should be close to the true product
            assert (
                abs(r_2d - expected) < 0.5
            ), f"multiply_2d({a}, {b}) = {r_2d}, expected {expected}"
            assert (
                abs(r_sm - expected) < 0.5
            ), f"signed_multiply({a}, {b}) = {r_sm}, expected {expected}"


# ---------------------------------------------------------------------------
# Probe (compilation) test
# ---------------------------------------------------------------------------


def test_multiply_2d_probe():
    """Compiled transformer matches the oracle."""
    node = _build_multiply_2d(max_abs1=5.0, max_abs2=5.0, step1=1.0, step2=1.0)

    # Sweep of integer grid points
    n_pos = 11
    a_vals = torch.tensor([[float(i - 5)] for i in range(n_pos)])
    b_vals = torch.tensor([[float(5 - i)] for i in range(n_pos)])
    inputs = {"a": a_vals, "b": b_vals}

    # Oracle check
    cache = reference_eval(node, inputs, n_pos)
    oracle = cache[node].flatten()
    expected = torch.tensor(
        [a_vals[i, 0].item() * b_vals[i, 0].item() for i in range(n_pos)]
    )
    assert torch.allclose(
        oracle, expected, atol=0.01
    ), f"oracle: {oracle.tolist()}\nexpected: {expected.tolist()}"

    # Compiled check
    report = probe_graph(
        node,
        pos_encoding=None,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=16,
        verbose=False,
        atol=0.5,
    )
    assert report.first_divergent is None, report.format_short()
