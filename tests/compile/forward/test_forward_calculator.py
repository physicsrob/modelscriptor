"""Tests for the V1 embedding-space calculator."""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Embedding

D = 1024
D_HEAD = 16


def decode_token(embedding: Embedding, vector: torch.Tensor) -> str:
    dists = torch.cdist(vector.unsqueeze(0).cpu(), embedding.table)
    return embedding.tokenizer.decode_id(int(dists.argmin().item()))


def run_autoregressive(
    net: HeadlessTransformer,
    output_node: Node,
    embedding: Embedding,
    input_tokens: list,
    max_new_tokens: int = 15,
) -> str:
    tokens = list(input_tokens)
    for _ in range(max_new_tokens):
        result = net.compute(
            n_pos=len(tokens), input_values={"embedding_input": tokens}
        )
        next_token = decode_token(embedding, result[output_node][-1])
        if next_token == "<eos>":
            break
        tokens.append(next_token)
    return "".join(tokens[len(input_tokens) :])


def _build(digits):
    import examples.calculator as calc_module

    original = calc_module.max_digits
    try:
        calc_module.max_digits = digits
        output_node, pos_encoding, embedding = calc_module.create_network_parts()
    finally:
        calc_module.max_digits = original
    return output_node, pos_encoding, embedding


def _compile(digits, d=D):
    output_node, pos_encoding, embedding = _build(digits)
    net = forward_compile(
        d=d,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=True,
    )
    return net, output_node, embedding


def _check(net, output_node, embedding, input_str, expected):
    tokens = ["<bos"] + list(input_str)
    result = run_autoregressive(net, output_node, embedding, tokens)
    assert (
        result == expected
    ), f"For {input_str}: expected '{expected}' but got '{result}'"


# ---------------------------------------------------------------------------
# Fixtures — compile once per digit count, share across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calc_1digit():
    net, output_node, embedding = _compile(1)
    return net, output_node, embedding


@pytest.fixture(scope="module")
def calc_3digit():
    net, output_node, embedding = _compile(3)
    return net, output_node, embedding


# ---------------------------------------------------------------------------
# Phase 1: Addition
# ---------------------------------------------------------------------------


def test_calc_addition_1digit(calc_1digit):
    net, output_node, embedding = calc_1digit
    _check(net, output_node, embedding, "1+1\n", "2")
    _check(net, output_node, embedding, "4+5\n", "9")
    _check(net, output_node, embedding, "0+0\n", "0")


def test_calc_addition_3digit(calc_3digit):
    net, output_node, embedding = calc_3digit
    _check(net, output_node, embedding, "1+1\n", "2")
    _check(net, output_node, embedding, "123+456\n", "579")
    _check(net, output_node, embedding, "99+1\n", "100")
    _check(net, output_node, embedding, "0+0\n", "0")


# ---------------------------------------------------------------------------
# Phase 2: Subtraction
# ---------------------------------------------------------------------------


def test_calc_subtraction_1digit(calc_1digit):
    net, output_node, embedding = calc_1digit
    _check(net, output_node, embedding, "5-3\n", "2")
    _check(net, output_node, embedding, "9-0\n", "9")
    _check(net, output_node, embedding, "0-0\n", "0")


def test_calc_subtraction_3digit(calc_3digit):
    net, output_node, embedding = calc_3digit
    _check(net, output_node, embedding, "456-123\n", "333")
    _check(net, output_node, embedding, "100-100\n", "0")


def test_calc_subtraction_negative(calc_3digit):
    net, output_node, embedding = calc_3digit
    _check(net, output_node, embedding, "1-5\n", "-4")
    _check(net, output_node, embedding, "100-999\n", "-899")


# ---------------------------------------------------------------------------
# Phase 3: Multiplication
# ---------------------------------------------------------------------------


def test_calc_multiplication_1digit(calc_1digit):
    net, output_node, embedding = calc_1digit
    _check(net, output_node, embedding, "2*3\n", "6")
    _check(net, output_node, embedding, "9*9\n", "81")
    _check(net, output_node, embedding, "0*5\n", "0")


def test_calc_multiplication_3digit(calc_3digit):
    net, output_node, embedding = calc_3digit
    _check(net, output_node, embedding, "12*34\n", "408")
    _check(net, output_node, embedding, "123*456\n", "56088")
    _check(net, output_node, embedding, "100*100\n", "10000")
