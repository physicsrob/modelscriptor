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
from modelscriptor.modelscript.arithmetic_ops import (
    add,
    add_scalar,
    add_scaled_nodes,
    relu,
    relu_add,
    concat,
    sum_nodes,
)
from modelscriptor.modelscript.logic_ops import cond_gate, cond_add_vector
from modelscriptor.modelscript.map_select import select, map_to_table

D = 256
D_HEAD = 16


def _verify(output_node, n_pos, input_values, pos_encoding=None, max_layers=100):
    """Compile and verify output matches node.compute()."""
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
        max_layers=max_layers,
    )
    assert net.feature_assignment is not None

    result = net.compute(n_pos, input_values)
    assert output_node in result
    actual = result[output_node]

    expected = output_node.compute(n_pos, input_values)
    assert torch.allclose(
        actual.cpu(), expected, atol=1e-4
    ), f"Max diff: {(actual.cpu() - expected).abs().max().item():.6f}"


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
    _verify(
        out,
        n_pos=3,
        input_values={
            "a": torch.randn(3, 4),
            "b": torch.randn(3, 4),
        },
    )


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
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "cond": torch.tensor([[1.0], [-1.0], [1.0]]),
            "true_val": torch.randn(n_pos, 4),
            "false_val": torch.randn(n_pos, 4),
        },
    )


def test_compile_get_last_value():
    """pos_encoding.get_last_value() — attention-based position lookup."""
    pos = create_pos_encoding()
    v = create_input("v", 4)
    out = pos.get_last_value(v, delta_pos=-1)

    n_pos = 5
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "v": torch.randn(n_pos, 4),
        },
        pos_encoding=pos,
    )


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
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "x": torch.tensor(
                [
                    [1.0, 0.0],  # expect 10.0
                    [0.0, 1.0],  # expect 20.0
                    [1.0, 1.0],  # expect 30.0
                ]
            ),
        },
    )


def test_compile_sum_nodes():
    """sum_nodes with 4 x 8-dim inputs -- Concatenate(32) -> Linear(32 -> 8).

    The Linear's input dim (32) exceeds d_head (16), requiring multi-head
    attention. This is the exact pattern from the adder's output_sequence.
    """
    inputs = [create_input(f"v{i}", 8) for i in range(4)]
    out = sum_nodes(inputs)

    n_pos = 3
    input_values = {f"v{i}": torch.randn(n_pos, 8) for i in range(4)}
    _verify(out, n_pos=n_pos, input_values=input_values)


# ---------------------------------------------------------------------------
# Patterns that exercise untested code paths (pre-Phase 5 validation)
# ---------------------------------------------------------------------------


def test_compile_cond_gate():
    """cond_gate — exercises standalone ReLU (Add -> ReLU -> Add pattern).

    This is the key pattern from the adder that uses standalone ReLU.
    cond_gate(cond, inp) = cond_add_vector(cond, relu(cond_add_vector(cond, inp, ...)), ...)
    """
    pos = create_pos_encoding()
    cond = create_input("cond", 1)
    inp = create_input("inp", 4)
    out = cond_gate(cond, inp)

    n_pos = 4
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "cond": torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]),
            "inp": torch.randn(n_pos, 4),
        },
        pos_encoding=pos,
    )


def test_compile_get_prev_value():
    """pos_encoding.get_prev_value() — cross-position attention with Concatenate key_in.

    get_prev_value returns the most recent value where cond was true.
    Exercises Attn node with Concatenate([pos, cond]) as key_in.
    """
    pos = create_pos_encoding()
    v = create_input("v", 4)
    cond = create_input("cond", 1)
    out = pos.get_prev_value(v, cond)

    n_pos = 5
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "v": torch.tensor(
                [
                    [10.0, 20.0, 30.0, 40.0],
                    [11.0, 21.0, 31.0, 41.0],
                    [12.0, 22.0, 32.0, 42.0],
                    [13.0, 23.0, 33.0, 43.0],
                    [14.0, 24.0, 34.0, 44.0],
                ]
            ),
            "cond": torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]]),
        },
        pos_encoding=pos,
    )


# ---------------------------------------------------------------------------
# Retargeted from old compiler tests (unique patterns)
# ---------------------------------------------------------------------------


def test_compile_repeated_adds():
    """Chain of adds on constants — exercises Add scheduling."""
    c1 = create_constant(torch.tensor([1.0]))
    c2 = create_constant(torch.tensor([1.0]))
    c3 = create_constant(torch.tensor([1.0]))
    c4 = create_constant(torch.tensor([1.0]))
    a1 = add(c1, c2)
    a2 = add(c3, c4)
    out = add(a1, a2)
    _verify(out, n_pos=2, input_values={})


def test_compile_add_relu():
    """Add -> ReLU -> ReLU — chained standalone ReLUs."""
    v1 = create_input("v1", 4)
    v2 = create_input("v2", 4)
    n_pos = 2
    a = add(v1, v2)
    r1 = relu(a)
    out = relu(r1)
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "v1": torch.randn(n_pos, 4) - 0.5,
            "v2": torch.randn(n_pos, 4) - 0.5,
        },
    )


def test_compile_add_scalar():
    """add_scalar — FFN bias-only addition via Add."""
    v = create_input("v", 1)
    out = add_scalar(v, 100.0)
    _verify(out, n_pos=1, input_values={"v": torch.tensor([[1.0]])})


def test_compile_cond_add_vector():
    """cond_add_vector — FFN multiplexer + Add."""
    cond = create_input("cond", 1)
    x = create_constant(torch.tensor([15.0, 25.0]))
    out = cond_add_vector(
        cond,
        x,
        true_vector=torch.tensor([100.0, 0.0]),
        false_vector=torch.tensor([0.0, 100.0]),
    )
    for cond_val in [-1.0, 1.0]:
        _verify(
            out,
            n_pos=1,
            input_values={
                "cond": torch.tensor([[cond_val]]),
            },
        )


def test_compile_relu_add():
    """relu_add — fused ReLU(a+b) via concatenate + FFN."""
    v1 = create_input("v1", 3)
    v2 = create_input("v2", 3)
    out = relu_add(v1, v2)
    for _ in range(3):
        _verify(
            out,
            n_pos=1,
            input_values={
                "v1": (100.0 * (torch.rand(1, 3) - 0.5)),
                "v2": (100.0 * (torch.rand(1, 3) - 0.5)),
            },
        )


def test_compile_multiple_concats():
    """Shared constants across multiple concat -> add paths."""
    c1 = create_constant(torch.tensor([1.0]))
    c2 = create_constant(torch.tensor([1.0]))
    c3 = create_constant(torch.tensor([1.0]))
    add1 = add(concat([c1, c2]), create_constant(torch.tensor([2.0, 2.0])))
    add2 = add(concat([c1, c3]), create_constant(torch.tensor([2.0, 2.0])))
    out = add(add1, add2)
    _verify(out, n_pos=1, input_values={})


def test_compile_switch():
    """switch(conditions, values) — N-way select via cond_gate + sum_nodes.

    This is the pattern used by the calculator for operator dispatch.
    Exercises the case where cond_gate creates Add(inp, chain_output) nodes
    whose live addends must survive cancellation during compilation.
    """
    from modelscriptor.modelscript.map_select import switch

    cond1 = create_input("c1", 1)
    cond2 = create_input("c2", 1)
    cond3 = create_input("c3", 1)
    v1 = create_constant(torch.tensor([10.0, 20.0]))
    v2 = create_constant(torch.tensor([30.0, 40.0]))
    v3 = create_constant(torch.tensor([50.0, 60.0]))
    out = switch([cond1, cond2, cond3], [v1, v2, v3])

    # Condition 1 true
    _verify(
        out,
        n_pos=1,
        input_values={
            "c1": torch.tensor([[1.0]]),
            "c2": torch.tensor([[-1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )

    # Condition 2 true
    _verify(
        out,
        n_pos=1,
        input_values={
            "c1": torch.tensor([[-1.0]]),
            "c2": torch.tensor([[1.0]]),
            "c3": torch.tensor([[-1.0]]),
        },
    )


def test_compile_multi_switch_shared_constants():
    """Multiple switch calls sharing the same constant placeholders.

    This is the exact pattern from the calculator: switch is called once per
    result digit position, with the same placeholder constants for unimplemented
    operations. The shared constants + many cond_gate chains create a graph where
    add_into live addends can be incorrectly freed.
    """
    from modelscriptor.modelscript.map_select import switch
    from modelscriptor.modelscript.logic_ops import cond_gate

    pos = create_pos_encoding()
    flag = create_input("flag", 1)

    # Three conditions via attention (like which_plus, which_minus, which_times)
    c1 = pos.get_prev_value(flag, flag)
    c2 = pos.get_prev_value(flag, flag)
    c3 = pos.get_prev_value(flag, flag)

    # Real values for c1, placeholder zeros for c2/c3
    zero = create_constant(torch.zeros(4))
    real_values = [create_constant(torch.randn(4)) for _ in range(3)]

    # Multiple switches sharing the same zero placeholder — like the calculator
    results = []
    for i in range(3):
        results.append(switch([c1, c2, c3], [real_values[i], zero, zero]))

    out = sum_nodes(results)

    n_pos = 3
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "flag": torch.tensor([[1.0], [-1.0], [1.0]]),
        },
        pos_encoding=pos,
    )


def test_compile_switch_with_attention_conditions():
    """switch where conditions come from attention (get_prev_value).

    This mirrors the calculator's operator dispatch: the conditions are
    latched via get_prev_value and feed into cond_gate chains. The deeper
    graph triggers multi-layer scheduling where add_into live addends
    can be incorrectly cancelled.
    """
    from modelscriptor.modelscript.map_select import switch
    from modelscriptor.modelscript.logic_ops import compare_to_vector

    pos = create_pos_encoding()
    embedding_dim = 8
    v1 = create_constant(torch.randn(embedding_dim))
    v2 = create_constant(torch.randn(embedding_dim))
    v3 = create_constant(torch.randn(embedding_dim))

    # Conditions via attention — like compare_to_vector + get_prev_value
    flag = create_input("flag", 1)
    c1 = pos.get_prev_value(flag, flag)
    c2 = pos.get_prev_value(flag, flag)
    c3 = pos.get_prev_value(flag, flag)

    out = switch([c1, c2, c3], [v1, v2, v3])

    n_pos = 3
    _verify(
        out,
        n_pos=n_pos,
        input_values={
            "flag": torch.tensor([[1.0], [-1.0], [1.0]]),
        },
        pos_encoding=pos,
    )


# ---------------------------------------------------------------------------
# Scheduler deadlock: Add nodes with shared inputs
# ---------------------------------------------------------------------------


def test_compile_add_shared_inputs():
    """Multiple Add nodes sharing the same inputs must not deadlock the scheduler.

    The scheduler only schedules Add via add_into when one input is "dead"
    (all its other consumers already computed). When two Add nodes share both
    inputs, neither input can become dead because each has the other Add as
    an unconsumed consumer — a circular dependency.

    This is the minimal reproduction: x and y feed two separate Add nodes,
    whose results are combined via a Linear (add_scaled_nodes).
    """
    x = create_input("x", 1)
    y = create_input("y", 1)

    # Two Add nodes sharing both inputs — deadlocks if not handled
    sum1 = add(x, y)
    sum2 = add(x, y)

    # Combine via Linear (not Add) so the output itself isn't blocked
    output = add_scaled_nodes(1.0, sum1, 1.0, sum2)

    _verify(
        output,
        n_pos=2,
        input_values={
            "x": torch.tensor([[3.0], [7.0]]),
            "y": torch.tensor([[4.0], [2.0]]),
        },
        max_layers=10,  # Deadlock manifests immediately; don't spin for 100 layers
    )


def test_compile_add_shared_inputs_wide():
    """Shared-input Add deadlock with wider vectors (multi-head compute_add)."""
    x = create_input("x", 8)
    y = create_input("y", 8)

    sum1 = add(x, y)
    sum2 = add(x, y)
    output = add_scaled_nodes(1.0, sum1, 1.0, sum2)

    _verify(
        output,
        n_pos=2,
        input_values={
            "x": torch.randn(2, 8),
            "y": torch.randn(2, 8),
        },
        max_layers=10,
    )


def test_compile_three_adds_shared_inputs():
    """Three Add nodes sharing the same inputs — the calculator pattern.

    This mirrors calculator_v2: number_a and number_b feed add, subtract,
    and multiply paths, each creating Add nodes on the shared operands.
    """
    x = create_input("x", 1)
    y = create_input("y", 1)

    s1 = add(x, y)  # addition path
    s2 = add(x, y)  # subtraction path (subtract creates add+negate)
    s3 = add(x, y)  # multiplication path

    output = sum_nodes([s1, s2, s3])

    _verify(
        output,
        n_pos=2,
        input_values={
            "x": torch.tensor([[3.0], [7.0]]),
            "y": torch.tensor([[4.0], [2.0]]),
        },
        max_layers=10,
    )
