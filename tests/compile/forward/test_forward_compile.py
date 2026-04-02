"""Tests for the forward compiler end-to-end.

Each test builds a graph, compiles it via forward_compile, and verifies
the output matches node.compute() with torch.allclose.
"""

import torch

from modelscriptor.compiler.forward.compile import forward_compile
from modelscriptor.graph import Linear, ReLU, Add, Concatenate
from modelscriptor.graph.misc import InputNode, Constant
from modelscriptor.graph.pos_encoding import PosEncoding
from modelscriptor.modelscript.inout_nodes import (
    create_input,
    create_constant,
    create_pos_encoding,
)
from modelscriptor.modelscript.arithmetic_ops import add, relu, concat, sum_nodes
from modelscriptor.modelscript.logic_ops import cond_add_vector
from modelscriptor.modelscript.map_select import select, map_to_table


D = 256
D_HEAD = 16


def _verify(output_node, n_pos, input_values, pos_encoding=None):
    """Compile and verify output matches node.compute()."""
    net = forward_compile(
        d=D, d_head=D_HEAD, output_node=output_node,
        pos_encoding=pos_encoding, verbose=False,
    )
    assert net.feature_assignment is not None

    result = net.compute(n_pos, input_values)
    assert output_node in result
    actual = result[output_node]

    expected = output_node.compute(n_pos, input_values)
    assert torch.allclose(actual, expected, atol=1e-4), (
        f"Max diff: {(actual - expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Basic graphs
# ---------------------------------------------------------------------------


def test_compile_constant():
    """Single Constant node — simplest possible graph."""
    const = create_constant(torch.tensor([1.0, -2.0, 3.5]))
    _verify(const, n_pos=2, input_values={})


def test_compile_linear():
    """Input -> Linear (with bias)."""
    x = create_input("x", 4)
    W = torch.randn(4, 3)
    b = torch.randn(3)
    out = Linear(x, W, b, name="lin")
    _verify(out, n_pos=3, input_values={"x": torch.randn(3, 4)})


def test_compile_relu_chain():
    """Input -> Linear -> ReLU -> Linear (the FFN pattern)."""
    x = create_input("x", 4)
    l1 = Linear(x, torch.randn(4, 8), torch.randn(8), name="l1")
    r = ReLU(l1)
    l2 = Linear(r, torch.randn(8, 3), torch.randn(3), name="l2")
    _verify(l2, n_pos=3, input_values={"x": torch.randn(3, 4)})


def test_compile_add():
    """Add(input1, input2)."""
    a = create_input("a", 4)
    b = create_input("b", 4)
    out = add(a, b)
    _verify(out, n_pos=3, input_values={
        "a": torch.randn(3, 4),
        "b": torch.randn(3, 4),
    })


# ---------------------------------------------------------------------------
# Patterns from the adder
# ---------------------------------------------------------------------------


def test_compile_select():
    """select(cond, true, false) — uses cond_add_vector + ffn_layer."""
    cond = create_input("cond", 1)
    true_val = create_input("true_val", 4)
    false_val = create_input("false_val", 4)
    out = select(cond, true_val, false_val)

    n_pos = 3
    _verify(out, n_pos=n_pos, input_values={
        "cond": torch.tensor([[1.0], [-1.0], [1.0]]),
        "true_val": torch.randn(n_pos, 4),
        "false_val": torch.randn(n_pos, 4),
    })


def test_compile_get_last_value():
    """pos_encoding.get_last_value() — attention-based position lookup."""
    pos = create_pos_encoding()
    v = create_input("v", 4)
    out = pos.get_last_value(v, delta_pos=-1)

    n_pos = 5
    _verify(out, n_pos=n_pos, input_values={
        "v": torch.randn(n_pos, 4),
    }, pos_encoding=pos)


def test_compile_map_to_table():
    """map_to_table — table lookup via FFN (large d_intermediate)."""
    x = create_input("x", 2)
    table = {
        torch.tensor([1.0, 0.0]): torch.tensor([10.0]),
        torch.tensor([0.0, 1.0]): torch.tensor([20.0]),
        torch.tensor([1.0, 1.0]): torch.tensor([30.0]),
    }
    out = map_to_table(x, table, default=torch.tensor([0.0]))

    n_pos = 3
    _verify(out, n_pos=n_pos, input_values={
        "x": torch.tensor([
            [1.0, 0.0],  # expect 10.0
            [0.0, 1.0],  # expect 20.0
            [1.0, 1.0],  # expect 30.0
        ]),
    })
