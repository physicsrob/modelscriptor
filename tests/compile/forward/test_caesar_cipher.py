"""Compile Caesar cipher and verify correctness.

Tests various shift amounts and letter combinations.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Embedding

from examples.caesar_cipher import create_network_parts

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


def _caesar(text: str, shift: int) -> str:
    """Reference Caesar cipher implementation."""
    return "".join(chr((ord(c) - ord("a") + shift) % 26 + ord("a")) for c in text)


def test_caesar_cipher():
    """Compile and verify Caesar cipher on various inputs."""
    net, output_node, embedding = _build()

    test_cases = [
        # shift=0: identity
        ("0", "hello", "hello"),
        ("0", "abcde", "abcde"),
        # shift=1: each letter +1
        ("1", "abcde", "bcdef"),
        ("1", "hello", "ifmmp"),
        # shift=3: classic Caesar
        ("3", "hello", "khoor"),
        ("3", "abcde", "defgh"),
        # Wraparound
        ("1", "xyzab", "yzabc"),
        ("3", "xyzab", "abcde"),
        # Larger shifts
        ("9", "abcde", "jklmn"),
        ("5", "vwxyz", "abcde"),
    ]
    for shift, plaintext, expected in test_cases:
        # Verify reference matches expected
        assert (
            _caesar(plaintext, int(shift)) == expected
        ), f"Reference mismatch for shift={shift}, text={plaintext}"
        tokens = ["<bos>", shift] + list(plaintext) + ["\n"]
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert result == expected, (
            f"For shift={shift}, '{plaintext}': "
            f"expected '{expected}' but got '{result}'"
        )


def test_caesar_cipher_layer_count():
    """Verify compilation stays within a reasonable layer budget."""
    net, _, _ = _build()
    n_layers = len(net.layers)
    print(f"caesar_cipher: {n_layers} layers, d={D}, d_head={D_HEAD}")
    assert n_layers <= 40, f"Too many layers: {n_layers}"
