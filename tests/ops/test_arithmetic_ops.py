from torchwright.ops.arithmetic_ops import (
    add_scalar,
    relu_add,
    compare,
    negate,
    subtract,
    multiply_scalar,
    square,
    multiply_integers,
)
from torchwright.ops.inout_nodes import create_input
import torch


def test_add_scalar():
    offset = 100.0
    value_input = create_input("value", 1)
    n = add_scalar(value_input, offset)
    output = n.compute(n_pos=1, input_values={"value": torch.tensor([[1.0]])})
    assert output.tolist() == [[101.0]]


def test_relu_add():
    input_val1 = create_input("val1", 3)
    input_val2 = create_input("val2", 3)
    x = relu_add(input_val1, input_val2)
    for i in range(10):
        val1 = 100.0 * (torch.rand(3) - 0.5)
        val2 = 100.0 * (torch.rand(3) - 0.5)
        output = x.compute(
            n_pos=1, input_values={"val1": val1.unsqueeze(0), "val2": val2.unsqueeze(0)}
        )
        expected_value = torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
        assert (output == expected_value).all()


def test_negate():
    x = create_input("x", 3)
    out = negate(x)
    vals = torch.tensor([[1.0, -2.0, 3.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, -vals)


def test_subtract():
    a = create_input("a", 3)
    b = create_input("b", 3)
    out = subtract(a, b)
    va = torch.tensor([[5.0, 10.0, 15.0]])
    vb = torch.tensor([[1.0, 3.0, 5.0]])
    result = out.compute(n_pos=1, input_values={"a": va, "b": vb})
    assert torch.allclose(result, va - vb)


def test_multiply_scalar():
    x = create_input("x", 2)
    out = multiply_scalar(x, 3.0)
    vals = torch.tensor([[2.0, -4.0]])
    result = out.compute(n_pos=1, input_values={"x": vals})
    assert torch.allclose(result, 3.0 * vals)


def test_compare():
    thresh = 100.0
    for delta in [-10.0, -5.0, 5.0, 10.0]:
        for true_level in [-10.0, 0.0, 1.0, 10.0]:
            for false_level in [-5.0, -1.0]:
                value_input = create_input("value", 1)
                n = compare(value_input, thresh, true_level, false_level)
                test_val = thresh + delta
                output = n.compute(
                    n_pos=1, input_values={"value": torch.tensor([[test_val]])}
                )
                assert output.item() == true_level if delta > 0 else false_level


def test_square():
    x = create_input("x", 1)
    sq = square(x, max_value=9)
    for val in range(10):
        result = sq.compute(n_pos=1, input_values={"x": torch.tensor([[float(val)]])})
        expected = val * val
        assert abs(result.item() - expected) < 0.01, (
            f"{val}² = {expected}, got {result.item()}"
        )


def test_square_large():
    """Test with max_value=18 (needed for digit a+b range in multiply_integers)."""
    x = create_input("x", 1)
    sq = square(x, max_value=18)
    for val in [0, 1, 9, 10, 17, 18]:
        result = sq.compute(n_pos=1, input_values={"x": torch.tensor([[float(val)]])})
        expected = val * val
        assert abs(result.item() - expected) < 0.01, (
            f"{val}² = {expected}, got {result.item()}"
        )


def test_multiply_integers_all_digit_pairs():
    """Verify a*b for all 100 single-digit pairs."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = multiply_integers(a, b, max_value=9)
    for i in range(10):
        for j in range(10):
            result = prod.compute(
                n_pos=1,
                input_values={
                    "a": torch.tensor([[float(i)]]),
                    "b": torch.tensor([[float(j)]]),
                },
            )
            expected = i * j
            assert abs(result.item() - expected) < 0.5, (
                f"{i}*{j} = {expected}, got {result.item()}"
            )


def test_multiply_integers_zero():
    """0 * anything = 0."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    prod = multiply_integers(a, b, max_value=9)
    for val in [0, 5, 9]:
        result = prod.compute(
            n_pos=1,
            input_values={
                "a": torch.tensor([[0.0]]),
                "b": torch.tensor([[float(val)]]),
            },
        )
        assert abs(result.item()) < 0.1, f"0*{val} should be 0, got {result.item()}"
