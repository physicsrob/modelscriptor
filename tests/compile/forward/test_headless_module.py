"""Tests for HeadlessTransformerModule (raw float I/O).

Verifies numerical equivalence between HeadlessTransformer.compute()
and the converted HeadlessTransformerModule.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import to_headless_module
from torchwright.compiler.export import compile_headless
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.arithmetic_ops import compare, signed_multiply, add
from torchwright.ops.map_select import select

D = 256
D_HEAD = 16


def _compile_and_convert(output_node, pos_encoding=None, input_nodes=None):
    """Compile a graph and convert to HeadlessTransformerModule."""
    if pos_encoding is None:
        pos_encoding = create_pos_encoding()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    module = to_headless_module(net, output_node, device="cpu")
    module.eval()
    return net, module


# ---------------------------------------------------------------------------
# Test 1: Forward output matches HeadlessTransformer.compute()
# ---------------------------------------------------------------------------


def test_headless_forward_matches_compiled():
    """HeadlessTransformerModule output matches HeadlessTransformer.compute()."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    cond = compare(a, 0.0)
    out = select(cond, b, a)

    net, module = _compile_and_convert(out)

    n_pos = 3
    a_vals = torch.tensor([[5.0], [-3.0], [0.5]])
    b_vals = torch.tensor([[10.0], [20.0], [30.0]])

    # HeadlessTransformer path
    result = net.compute(n_pos, {"a": a_vals, "b": b_vals})
    expected = result[out].cpu()

    # Module path — inputs ordered alphabetically: a then b
    inputs = torch.cat([a_vals, b_vals], dim=1)  # (3, 2)
    with torch.no_grad():
        actual = module(inputs)

    assert torch.allclose(actual, expected, atol=1e-4), (
        f"Max diff: {(actual - expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: signed_multiply
# ---------------------------------------------------------------------------


def test_headless_multiply():
    """signed_multiply compiled to headless module produces correct results."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = signed_multiply(a, b, max_abs1=10, max_abs2=10)

    _, module = _compile_and_convert(out)

    test_cases = [
        ([3.0, 4.0], 12.0),
        ([5.0, 2.0], 10.0),
        ([-3.0, 4.0], -12.0),
        ([0.0, 7.0], 0.0),
    ]
    for (a_val, b_val), expected in test_cases:
        inputs = torch.tensor([[a_val, b_val]])
        with torch.no_grad():
            result = module(inputs)
        assert abs(result.item() - expected) < 0.5, (
            f"{a_val} * {b_val}: expected {expected}, got {result.item():.2f}"
        )


# ---------------------------------------------------------------------------
# Test 3: compare
# ---------------------------------------------------------------------------


def test_headless_compare():
    """compare compiled to headless module produces correct boolean output."""
    x = create_input("x", 1)
    out = compare(x, 0.0)

    _, module = _compile_and_convert(out)

    test_cases = [
        (5.0, 1.0),
        (-3.0, -1.0),
        (100.0, 1.0),
        (-0.5, -1.0),
    ]
    for x_val, expected in test_cases:
        inputs = torch.tensor([[x_val]])
        with torch.no_grad():
            result = module(inputs)
        assert abs(result.item() - expected) < 0.1, (
            f"compare({x_val}, 0): expected {expected}, got {result.item():.2f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Multiple positions with different values
# ---------------------------------------------------------------------------


def test_headless_multi_position():
    """Multiple positions produce independent correct results."""
    x = create_input("x", 1)
    out = compare(x, 0.0)

    _, module = _compile_and_convert(out)

    # 5 positions with different values
    inputs = torch.tensor([[3.0], [-2.0], [0.5], [-0.5], [10.0]])
    expected = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]])

    with torch.no_grad():
        actual = module(inputs)

    assert torch.allclose(actual, expected, atol=0.1), (
        f"Multi-position mismatch: {actual.squeeze().tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 5: state_dict roundtrip
# ---------------------------------------------------------------------------


def test_headless_state_dict_roundtrip():
    """Save and load state_dict, verify identical output."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = add(a, b)

    net, module1 = _compile_and_convert(out)

    inputs = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    with torch.no_grad():
        result1 = module1(inputs)

    # Create second module and load state_dict
    module2 = to_headless_module(net, out, device="cpu")
    module2.load_state_dict(module1.state_dict())
    module2.eval()

    with torch.no_grad():
        result2 = module2(inputs)

    assert torch.equal(result1, result2), "state_dict roundtrip changed output"


# ---------------------------------------------------------------------------
# Test 6: input_names ordering
# ---------------------------------------------------------------------------


def test_headless_input_names():
    """input_names lists InputNode names in alphabetical order."""
    z = create_input("zebra", 1)
    a = create_input("alpha", 1)
    out = add(z, a)

    _, module = _compile_and_convert(out)

    assert module.input_names == ["alpha", "zebra"]


# ---------------------------------------------------------------------------
# Test 7: compile_headless convenience function
# ---------------------------------------------------------------------------


def test_compile_headless():
    """compile_headless produces a working HeadlessTransformerModule."""
    x = create_input("x", 1)
    out = compare(x, 5.0)
    pos = create_pos_encoding()

    module = compile_headless(
        out, pos, d=D, d_head=D_HEAD, verbose=False, device="cpu"
    )
    module.eval()

    with torch.no_grad():
        result = module(torch.tensor([[10.0]]))
    assert result.item() > 0.5, f"compare(10, 5) should be true, got {result.item()}"

    with torch.no_grad():
        result = module(torch.tensor([[1.0]]))
    assert result.item() < -0.5, f"compare(1, 5) should be false, got {result.item()}"
