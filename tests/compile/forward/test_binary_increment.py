"""Compile binary increment and verify correctness.

Tests carry propagation, overflow, and variable-length inputs up to 4 bits.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Embedding

from examples.binary_increment import create_network_parts

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
    max_new_tokens: int = 8,
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


def test_binary_increment():
    """Compile and verify binary increment on a range of inputs."""
    net, output_node, embedding = _build()

    n_layers = len(net.layers)
    print(f"binary_increment: {n_layers} layers, d={D}, d_head={D_HEAD}")
    assert n_layers <= 40, f"Too many layers: {n_layers}"

    test_cases = [
        # Simple increments
        ("0", "1"),
        ("1", "10"),
        ("10", "11"),
        ("11", "100"),
        ("100", "101"),
        ("101", "110"),
        ("110", "111"),
        ("111", "1000"),
        # Multi-bit carry propagation
        ("1011", "1100"),
        ("1111", "10000"),
    ]
    for binary_in, expected in test_cases:
        tokens = ["<bos>"] + list(binary_in) + ["\n"]
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert (
            result == expected
        ), f"For '{binary_in}': expected '{expected}' but got '{result}'"
