"""Phase 5: Compile 1-digit and 3-digit adders via forward_compile,
verify arithmetic correctness.
"""

import torch

from modelscriptor.compiler.forward.compile import forward_compile
from modelscriptor.compiler.transformer import HeadlessTransformer
from modelscriptor.graph import Node, Embedding

from examples.adder import create_network_parts

D = 1024
D_HEAD = 16


def decode_token(embedding: Embedding, vector: torch.Tensor) -> str:
    """Decode a single embedding vector to its nearest token."""
    dists = torch.cdist(vector.unsqueeze(0).cpu(), embedding.table)
    return embedding.tokenizer.decode_id(int(dists.argmin().item()))


def run_autoregressive(
    net: HeadlessTransformer,
    output_node: Node,
    embedding: Embedding,
    input_tokens: list,
    max_new_tokens: int = 10,
) -> str:
    """Run a compiled network autoregressively, appending output tokens until <eos>."""
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


# ---------------------------------------------------------------------------
# 1-digit adder
# ---------------------------------------------------------------------------


def _build_1digit():
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        output_node, pos_encoding, embedding = create_network_parts()
    finally:
        adder_module.max_digits = original
    return output_node, pos_encoding, embedding


def test_1digit_adder():
    """Compile 1-digit adder and verify arithmetic at the '=' position."""
    output_node, pos_encoding, embedding = _build_1digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=True,
    )


    test_cases = [
        ("1+1=", "2"),
        ("2+3=", "5"),
        ("0+0=", "0"),
        ("4+5=", "9"),
        ("7+2=", "9"),
        ("6+3=", "9"),
    ]
    for input_str, expected in test_cases:
        tokens = ["<bos"] + list(input_str)
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert (
            result == expected
        ), f"For {input_str}: expected '{expected}' but got '{result}'"


def test_1digit_layer_count():
    """Verify 1-digit adder compiles in a reasonable number of layers."""
    output_node, pos_encoding, embedding = _build_1digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )

    n_layers = len(net.layers)
    print(f"1-digit adder: {n_layers} layers")
    assert n_layers <= 20, f"Too many layers: {n_layers}"


# ---------------------------------------------------------------------------
# 3-digit adder
# ---------------------------------------------------------------------------


def _build_3digit():
    output_node, pos_encoding, embedding = create_network_parts()
    return output_node, pos_encoding, embedding


def test_3digit_adder():
    """Compile 3-digit adder, verify arithmetic at the position after '='."""
    output_node, pos_encoding, embedding = _build_3digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=True,
    )


    test_cases = [
        ("1+1=", "2"),
        ("12+34=", "46"),
        ("123+456=", "579"),
        ("100+200=", "300"),
        ("0+0=", "0"),
        ("99+1=", "100"),
    ]
    for input_str, expected in test_cases:
        tokens = ["<bos"] + list(input_str)
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert (
            result == expected
        ), f"For {input_str}: expected '{expected}' but got '{result}'"


def test_3digit_autoregressive():
    """Run 3-digit adder autoregressively and verify complete output sequences."""
    output_node, pos_encoding, embedding = _build_3digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )


    test_cases = [
        ("1+2=", "3"),
        ("99+1=", "100"),
        ("100+200=", "300"),
        ("111+222=", "333"),
        ("456+123=", "579"),
    ]
    for input_str, expected in test_cases:
        tokens = ["<bos"] + list(input_str)
        result = run_autoregressive(net, output_node, embedding, tokens)
        assert (
            result == expected
        ), f"For {input_str}: expected '{expected}' but got '{result}'"


def test_3digit_resource_usage():
    """Log layers and peak column utilization for the 3-digit adder."""
    output_node, pos_encoding, embedding = _build_3digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=True,
    )

    n_layers = len(net.layers)
    print(f"3-digit adder: {n_layers} layers, d={D}, d_head={D_HEAD}")
    # Sanity bound — should compile in well under 100 layers
    assert n_layers <= 50, f"Too many layers: {n_layers}"
