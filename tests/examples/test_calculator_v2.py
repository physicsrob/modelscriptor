import pytest

from examples.calculator_v2 import create_network


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
        for _ in range(15):
            out_tokens = network.compute(
                n_pos=len(in_tokens),
                input_values={"embedding_input": in_tokens},
            )
            if out_tokens[-1] == "<eos>":
                break
            generated.append(out_tokens[-1])
            in_tokens.append(out_tokens[-1])
        return "".join(generated)

    # --- Addition ---

    @pytest.mark.parametrize(
        "a,b",
        [(0, 0), (1, 1), (5, 3), (9, 9), (0, 9), (7, 8)],
    )
    def test_1digit_addition(self, network_1digit, a, b):
        result = self._eval(network_1digit, f"{a}+{b}=")
        expected = str(a + b)
        assert result == expected, f"{a}+{b}: expected '{expected}', got '{result}'"

    # --- Subtraction ---

    @pytest.mark.parametrize(
        "a,b",
        [(5, 3), (9, 0), (0, 0), (5, 5)],
    )
    def test_1digit_subtraction(self, network_1digit, a, b):
        result = self._eval(network_1digit, f"{a}-{b}=")
        expected = str(a - b)
        assert result == expected, f"{a}-{b}: expected '{expected}', got '{result}'"

    @pytest.mark.parametrize(
        "a,b",
        [(3, 5), (0, 9), (1, 8)],
    )
    def test_1digit_subtraction_negative(self, network_1digit, a, b):
        result = self._eval(network_1digit, f"{a}-{b}=")
        expected = str(a - b)
        assert result == expected, f"{a}-{b}: expected '{expected}', got '{result}'"

    # --- Multiplication ---

    @pytest.mark.parametrize(
        "a,b",
        [(2, 3), (9, 9), (0, 5), (1, 0), (7, 8)],
    )
    def test_1digit_multiplication(self, network_1digit, a, b):
        result = self._eval(network_1digit, f"{a}*{b}=")
        expected = str(a * b)
        assert result == expected, f"{a}*{b}: expected '{expected}', got '{result}'"

    # --- 2-digit first-token spot checks ---

    def test_2digit_addition_first_token(self, network_2digit):
        for a, b, expected_first in [(12, 34, "4"), (50, 50, "1"), (99, 1, "1")]:
            in_tokens = ["<bos"] + list(f"{a}+{b}=")
            out = network_2digit.compute(
                n_pos=len(in_tokens),
                input_values={"embedding_input": in_tokens},
            )
            assert out[-1] == expected_first, (
                f"{a}+{b}=: first token expected '{expected_first}', got '{out[-1]}'"
            )

    def test_2digit_multiplication_first_token(self, network_2digit):
        for a, b, expected_first in [(12, 34, "4"), (10, 10, "1")]:
            in_tokens = ["<bos"] + list(f"{a}*{b}=")
            out = network_2digit.compute(
                n_pos=len(in_tokens),
                input_values={"embedding_input": in_tokens},
            )
            assert out[-1] == expected_first, (
                f"{a}*{b}=: first token expected '{expected_first}', got '{out[-1]}'"
            )
