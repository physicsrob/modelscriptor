"""Compile Fibonacci generator and verify correctness.

Tests that the autoregressive recurrence produces correct Fibonacci numbers.
The input prompt must have enough tokens before \\n to avoid out-of-bounds
attention (at least n_terms * digit_width tokens).
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Embedding

from examples.fibonacci import create_network_parts

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
    max_new_tokens: int = 20,
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


def _build():
    output_node, pos_encoding, embedding = create_network_parts()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=True,
    )
    return net, output_node, embedding


def test_fibonacci():
    """Compile and verify Fibonacci sequence generation."""
    net, output_node, embedding = _build()

    n_layers = len(net.layers)
    print(f"fibonacci: {n_layers} layers, d={D}, d_head={D_HEAD}")
    assert n_layers <= 50, f"Too many layers: {n_layers}"

    # Use "fibonacci" as the prompt — 9 letters provides enough tokens
    # before \n to avoid OOB attention for 17 output entries.
    prompt = list("fibonacci")
    tokens = ["<bos>"] + prompt + ["\n"]

    result = run_autoregressive(net, output_node, embedding, tokens)

    # Expected: 8 Fibonacci terms, each zero-padded to 2 digits
    # F = 1, 1, 2, 3, 5, 8, 13, 21
    expected_fibs = [1, 1, 2, 3, 5, 8, 13, 21]
    expected = "".join(f"{f:02d}" for f in expected_fibs)

    assert result == expected, f"Expected '{expected}' but got '{result}'"
