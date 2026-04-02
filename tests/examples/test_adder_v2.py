import pytest
import torch

from modelscriptor.modelscript.inout_nodes import create_embedding, create_input
from examples.adder_v2 import (
    digit_to_scaled_scalar,
    digits_to_number,
    thermometer_floor_div,
    number_to_digit_scalars,
    scalar_to_embedding,
    create_network,
)


@pytest.fixture
def embedding():
    vocab = list("0123456789+-=") + ["<eos>", "default"]
    return create_embedding(vocab=vocab)


class TestDigitToScaledScalar:
    def test_ones_place(self, embedding):
        for digit in range(10):
            node = digit_to_scaled_scalar(embedding, embedding, place_value=1.0)
            result = node.compute(
                n_pos=1, input_values={"embedding_input": [str(digit)]}
            )
            assert abs(result[0, 0].item() - float(digit)) < 0.1

    def test_hundreds_place(self, embedding):
        for digit in range(10):
            node = digit_to_scaled_scalar(embedding, embedding, place_value=100.0)
            result = node.compute(
                n_pos=1, input_values={"embedding_input": [str(digit)]}
            )
            assert abs(result[0, 0].item() - float(digit) * 100.0) < 1.0


class TestDigitsToNumber:
    def test_single_digit(self, embedding):
        node = digits_to_number(embedding, [embedding])
        for digit in range(10):
            result = node.compute(
                n_pos=1, input_values={"embedding_input": [str(digit)]}
            )
            assert abs(result[0, 0].item() - float(digit)) < 0.1

    def test_three_digits(self, embedding):
        """Test with 3-position sequence to verify place values."""
        node = digits_to_number(embedding, [embedding, embedding, embedding])
        # For a single-position evaluation, all three digits see the same token.
        # With input "5", result should be 5*100 + 5*10 + 5*1 = 555
        result = node.compute(n_pos=1, input_values={"embedding_input": ["5"]})
        assert abs(result[0, 0].item() - 555.0) < 1.0


class TestThermometerFloorDiv:
    def test_div_by_10(self):
        inp = create_input("x", 1)
        node = thermometer_floor_div(inp, divisor=10, max_value=99)
        for x in [0, 5, 9, 10, 15, 19, 50, 90, 95, 99]:
            result = node.compute(
                n_pos=1, input_values={"x": torch.tensor([[float(x)]])}
            )
            expected = x // 10
            assert (
                abs(result[0, 0].item() - expected) < 0.1
            ), f"floor({x}/10): expected {expected}, got {result[0, 0].item()}"

    def test_div_by_100(self):
        inp = create_input("x", 1)
        node = thermometer_floor_div(inp, divisor=100, max_value=999)
        for x in [0, 50, 99, 100, 150, 500, 999]:
            result = node.compute(
                n_pos=1, input_values={"x": torch.tensor([[float(x)]])}
            )
            expected = x // 100
            assert (
                abs(result[0, 0].item() - expected) < 0.1
            ), f"floor({x}/100): expected {expected}, got {result[0, 0].item()}"

    def test_div_by_1000(self):
        inp = create_input("x", 1)
        node = thermometer_floor_div(inp, divisor=1000, max_value=1998)
        for x in [0, 500, 999, 1000, 1500, 1998]:
            result = node.compute(
                n_pos=1, input_values={"x": torch.tensor([[float(x)]])}
            )
            expected = x // 1000
            assert (
                abs(result[0, 0].item() - expected) < 0.1
            ), f"floor({x}/1000): expected {expected}, got {result[0, 0].item()}"


class TestNumberToDigitScalars:
    def test_four_digit_extraction(self):
        inp = create_input("x", 1)
        digits = number_to_digit_scalars(inp, num_digits=4, max_value=1998)
        assert len(digits) == 4
        for x in [0, 7, 42, 123, 999, 1000, 1234, 1998]:
            expected = [
                (x // 1000) % 10,
                (x // 100) % 10,
                (x // 10) % 10,
                x % 10,
            ]
            for i, digit_node in enumerate(digits):
                result = digit_node.compute(
                    n_pos=1, input_values={"x": torch.tensor([[float(x)]])}
                )
                assert (
                    abs(result[0, 0].item() - expected[i]) < 0.2
                ), f"x={x}, digit[{i}]: expected {expected[i]}, got {result[0, 0].item()}"

    def test_two_digit_extraction(self):
        inp = create_input("x", 1)
        digits = number_to_digit_scalars(inp, num_digits=2, max_value=18)
        for x in [0, 5, 9, 10, 15, 18]:
            expected = [x // 10, x % 10]
            for i, digit_node in enumerate(digits):
                result = digit_node.compute(
                    n_pos=1, input_values={"x": torch.tensor([[float(x)]])}
                )
                assert (
                    abs(result[0, 0].item() - expected[i]) < 0.2
                ), f"x={x}, digit[{i}]: expected {expected[i]}, got {result[0, 0].item()}"


class TestScalarToEmbedding:
    def test_all_digits(self, embedding):
        inp = create_input("x", 1)
        node = scalar_to_embedding(inp, embedding)
        for digit in range(10):
            result = node.compute(
                n_pos=1, input_values={"x": torch.tensor([[float(digit)]])}
            )
            expected = embedding.get_embedding(str(digit))
            assert torch.allclose(
                result[0], expected, atol=0.1
            ), f"digit={digit}: max diff={torch.max(torch.abs(result[0] - expected)).item()}"


class TestEndToEnd:
    """End-to-end tests using raw graph evaluation (no compilation).

    Graph evaluation is slow for large networks, so we use max_digits=1
    for full autoregressive tests and max_digits=2 for single-shot checks.
    """

    @pytest.fixture(scope="class")
    def network_1digit(self):
        return create_network(max_digits=1)

    @pytest.fixture(scope="class")
    def network_2digit(self):
        return create_network(max_digits=2)

    @staticmethod
    def _eval(network, input_text: str) -> str:
        in_tokens = ["<bos"] + list(input_text)
        generated = []
        for _ in range(10):
            out_tokens = network.compute(
                n_pos=len(in_tokens),
                input_values={"embedding_input": in_tokens},
            )
            if out_tokens[-1] == "<eos>":
                break
            generated.append(out_tokens[-1])
            in_tokens.append(out_tokens[-1])
        return "".join(generated)

    @pytest.mark.parametrize(
        "a,b",
        [(0, 0), (1, 1), (5, 3), (9, 9), (0, 9), (7, 8)],
    )
    def test_1digit_addition(self, network_1digit, a, b):
        result = self._eval(network_1digit, f"{a}+{b}=")
        expected = str(a + b)
        assert result == expected, f"{a}+{b}: expected '{expected}', got '{result}'"

    def test_2digit_first_token(self, network_2digit):
        """Verify the first output token for 2-digit addition (no autoregressive loop)."""
        for a, b, expected_first in [(12, 34, "4"), (50, 50, "1"), (99, 1, "1")]:
            in_tokens = ["<bos"] + list(f"{a}+{b}=")
            out = network_2digit.compute(
                n_pos=len(in_tokens),
                input_values={"embedding_input": in_tokens},
            )
            assert (
                out[-1] == expected_first
            ), f"{a}+{b}=: first token expected '{expected_first}', got '{out[-1]}'"
