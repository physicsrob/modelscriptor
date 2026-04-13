"""Tests for graph optimization passes."""

import torch
import pytest

from torchwright.graph import Linear, InputNode, Concatenate
from torchwright.graph.optimize import fuse_consecutive_linears


def test_fuse_simple_chain():
    """Fuse L1 -> L2 into a single Linear."""
    inp = InputNode("x", d_output=4)
    l1 = Linear(inp, torch.randn(4, 3), torch.randn(3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), torch.randn(2), name="l2")

    # Compute before fusion
    n_pos = 5
    x = torch.randn(n_pos, 4)
    out_before = l2.compute(n_pos, {"x": x})

    # Fuse
    fused = fuse_consecutive_linears({l2})
    assert fused == 1
    assert l2.output_matrix.shape == (4, 2)
    assert l2.inputs[0] is inp

    # Output should match
    out_after = l2.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_fuse_chain_of_three():
    """Fuse L1 -> L2 -> L3 in two passes."""
    inp = InputNode("x", d_output=4)
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l3 = Linear(l2, torch.randn(2, 1), name="l3")

    n_pos = 5
    x = torch.randn(n_pos, 4)
    out_before = l3.compute(n_pos, {"x": x})

    # First pass fuses l1+l2, second pass fuses (l1+l2)+l3
    total = 0
    while True:
        fused = fuse_consecutive_linears({l3})
        if fused == 0:
            break
        total += fused

    assert total == 2
    assert l3.output_matrix.shape == (4, 1)
    assert l3.inputs[0] is inp

    out_after = l3.compute(n_pos, {"x": x})
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_no_fuse_multiple_consumers():
    """Don't fuse when L1 has multiple consumers."""
    inp = InputNode("x", d_output=4)
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l3 = Linear(l1, torch.randn(3, 2), name="l3")  # Another consumer of l1

    fused = fuse_consecutive_linears({l2, l3})
    assert fused == 0  # Can't fuse because l1 has two consumers


def test_no_fuse_concatenate_input():
    """Don't fuse when L2's input is a Concatenate (even if it wraps a Linear)."""
    inp = InputNode("x", d_output=4)
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    concat = Concatenate([l1])  # Wrap l1 in a Concatenate
    l2 = Linear(concat, torch.randn(3, 2), name="l2")

    fused = fuse_consecutive_linears({l2})
    assert fused == 0  # Skip Concatenate inputs


def test_fuse_preserves_annotation():
    """Fused node keeps L2's annotation."""
    inp = InputNode("x", d_output=4)
    l1 = Linear(inp, torch.randn(4, 3), name="l1")
    l1.annotation = "first"
    l2 = Linear(l1, torch.randn(3, 2), name="l2")
    l2.annotation = "second"

    fuse_consecutive_linears({l2})
    assert l2.annotation == "second"
