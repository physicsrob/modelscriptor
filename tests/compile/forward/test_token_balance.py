"""Compile token-balance checker and verify correctness.

This validates the parallel prefix sum + position-gating strategy
that the balanced-parentheses checker builds on.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Embedding

from examples.token_balance import create_network_parts

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
    max_new_tokens: int = 5,
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


def test_token_balance():
    """Compile and verify equal-count detection for 'a' and 'b'."""
    net, output_node, embedding = _build()

    test_cases = [
        # Balanced
        ("ab", "Y"),
        ("ba", "Y"),
        ("aabb", "Y"),
        ("abab", "Y"),
        ("abba", "Y"),
        ("aaabbb", "Y"),
        ("ababab", "Y"),
        ("", "Y"),
        # Unbalanced
        ("a", "N"),
        ("b", "N"),
        ("aab", "N"),
        ("abb", "N"),
        ("aaab", "N"),
        ("abbb", "N"),
    ]
    for input_str, expected in test_cases:
        tokens = ["<bos>"] + list(input_str) + ["?"]
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert (
            result == expected
        ), f"For '{input_str}': expected '{expected}' but got '{result}'"


def test_token_balance_layer_count():
    """Verify compilation stays within a reasonable layer budget."""
    net, _, _ = _build()
    n_layers = len(net.layers)
    print(f"token_balance: {n_layers} layers, d={D}, d_head={D_HEAD}")
    assert n_layers <= 40, f"Too many layers: {n_layers}"
