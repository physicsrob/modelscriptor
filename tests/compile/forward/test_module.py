"""Tests for CompiledTransformerModule (nn.Module conversion).

Verifies numerical equivalence between the original HeadlessTransformer
and the converted nn.Module at every level: attention, FFN, input scatter,
output gather, and full model.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import (
    CompiledTransformerModule,
    _AttentionLayer,
    _FFNLayer,
    to_module,
)
from torchwright.graph import Embedding

from examples.adder import create_network_parts

D = 1024
D_HEAD = 16


def _build_1digit():
    import examples.adder as adder_module

    original = adder_module.max_digits
    try:
        adder_module.max_digits = 1
        output_node, pos_encoding, embedding = create_network_parts()
    finally:
        adder_module.max_digits = original
    return output_node, pos_encoding, embedding


def _compile_1digit():
    output_node, pos_encoding, embedding = _build_1digit()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    return net, output_node, pos_encoding, embedding


# ---------------------------------------------------------------------------
# Test 1: Full model forward matches compiled HeadlessTransformer
# ---------------------------------------------------------------------------


def test_module_forward_matches_compiled():
    """Module output embedding vectors match HeadlessTransformer.compute() output."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)
    module.eval()
    device = next(module.parameters()).device

    test_inputs = [
        ["<bos", "1", "+", "2", "\n"],
        ["<bos", "3", "+", "4", "\n"],
        ["<bos", "0", "+", "0", "\n"],
    ]

    for tokens in test_inputs:
        # HeadlessTransformer path
        result = net.compute(
            n_pos=len(tokens),
            input_values={"embedding_input": tokens},
        )
        expected_emb = result[output_node]  # (seq_len, d_embed)

        # Module path
        token_ids = torch.tensor(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=torch.long,
            device=device,
        )
        logits = module(token_ids)  # (seq_len, vocab_size)

        # Extract the module's output embedding for comparison
        with torch.no_grad():
            res = module.token_embedding(token_ids) @ module.embedding_proj
            pos = module.pos_encoding[: len(tokens)] @ module.pos_proj
            res = res + pos + module.constant_values
            for layer_pair in module.layers:
                res = layer_pair[0](res)
                res = layer_pair[1](res)
            actual_emb = res[:, module.output_gather_indices]

        assert torch.allclose(actual_emb.cpu(), expected_emb.cpu(), atol=1e-4), (
            f"Max diff: {(actual_emb.cpu() - expected_emb.cpu()).abs().max().item():.6f} "
            f"for tokens {tokens}"
        )


# ---------------------------------------------------------------------------
# Test 2: Attention layer matches AttnLayerComponent
# ---------------------------------------------------------------------------


def test_attention_layer_matches_component():
    """_AttentionLayer produces same output as AttnLayerComponent + skip."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    device = net.device

    layer = net.layers[0]
    attn_comp = layer.attn.attn
    n_heads = attn_comp.n_heads
    d_head = attn_comp.d_head

    # Fuse weights same way as to_module
    W_Q = attn_comp.query_matrix.permute(1, 0, 2).reshape(D, D)
    W_K = attn_comp.key_matrix.permute(1, 0, 2).reshape(D, D)
    W_V = attn_comp.value_matrix.permute(1, 0, 2).reshape(D, D)
    W_O = attn_comp.output_matrix.clone()
    causal_mask = torch.triu(torch.ones(512, 512, dtype=torch.bool), diagonal=1)

    attn_mod = _AttentionLayer(W_Q, W_K, W_V, W_O, n_heads, d_head, causal_mask)
    attn_mod.to(device)
    attn_mod.eval()

    inp = torch.randn(8, D, device=device)

    # Original: attn component forward + skip
    expected = attn_comp.forward(inp) + inp

    # Module version
    actual = attn_mod(inp)

    assert torch.allclose(
        actual.cpu(), expected.cpu(), atol=1e-5
    ), f"Attention max diff: {(actual.cpu() - expected.cpu()).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 3: FFN layer matches FFNSubLayer
# ---------------------------------------------------------------------------


def test_ffn_layer_matches_component():
    """_FFNLayer produces same output as FFNSubLayer.forward()."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    device = net.device

    layer = net.layers[0]
    ffn_comp = layer.ffn

    W1 = ffn_comp.linear1.output_matrix.clone()
    b1 = ffn_comp.linear1.output_bias.clone()
    W2 = ffn_comp.linear2.output_matrix.clone()
    b2 = ffn_comp.linear2.output_bias.clone()

    ffn_mod = _FFNLayer(W1, b1, W2, b2)
    ffn_mod.to(device)
    ffn_mod.eval()

    inp = torch.randn(8, D, device=device)

    expected = ffn_comp.forward(inp)
    actual = ffn_mod(inp)

    assert torch.allclose(
        actual.cpu(), expected.cpu(), atol=1e-5
    ), f"FFN max diff: {(actual.cpu() - expected.cpu()).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 4: Input scatter matches get_input_res_stream
# ---------------------------------------------------------------------------


def test_input_scatter_matches_get_input_res_stream():
    """Module's input pipeline produces same residual stream as HeadlessTransformer."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)
    module.eval()
    device = next(module.parameters()).device

    tokens = ["<bos", "1", "+", "2", "\n"]

    # HeadlessTransformer path
    expected = net.get_input_res_stream(
        n_pos=len(tokens),
        input_values={"embedding_input": tokens},
    )

    # Module path
    token_ids = torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in tokens],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        embedded = module.token_embedding(token_ids)
        pos = module.pos_encoding[: len(tokens)]
        actual = (
            embedded @ module.embedding_proj
            + pos @ module.pos_proj
            + module.constant_values
        )

    assert torch.allclose(
        actual.cpu(), expected.cpu(), atol=1e-5
    ), f"Input scatter max diff: {(actual.cpu() - expected.cpu()).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 5: Output gather + logits match decode_token
# ---------------------------------------------------------------------------


def test_output_gather_matches_decode():
    """Module logits argmax produces same tokens as HeadlessTransformer decode."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)
    module.eval()
    device = next(module.parameters()).device

    tokens = ["<bos", "1", "+", "2", "\n"]

    # HeadlessTransformer path: get output embedding, decode each position
    result = net.compute(
        n_pos=len(tokens),
        input_values={"embedding_input": tokens},
    )
    ht_output = result[output_node]  # (seq_len, d_embed)
    ht_token_ids = []
    for pos in range(len(tokens)):
        dists = torch.cdist(ht_output[pos].unsqueeze(0).cpu(), embedding.table)
        ht_token_ids.append(dists.argmin().item())

    # Module path: logits argmax
    token_ids = torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in tokens],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        logits = module(token_ids)
    module_token_ids = logits.argmax(dim=-1).tolist()

    assert (
        ht_token_ids == module_token_ids
    ), f"Token mismatch: HT={ht_token_ids}, Module={module_token_ids}"


# ---------------------------------------------------------------------------
# Test 6: Autoregressive generation (1-digit adder)
# ---------------------------------------------------------------------------


def _module_generate(module, tokenizer, input_text, max_new_tokens=10):
    """Run autoregressive generation using the nn.Module."""
    device = next(module.parameters()).device
    tokens = ["<bos"] + list(input_text)
    token_ids = torch.tensor(
        [tokenizer.get_token_id(t) for t in tokens], dtype=torch.long, device=device
    )
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = module(token_ids)
        next_id = logits[-1].argmax().item()
        next_token = tokenizer.decode_id(next_id)
        if next_token == "<eos>":
            break
        token_ids = torch.cat([token_ids, torch.tensor([next_id], device=device)])
    return "".join(tokenizer.decode_id(tid.item()) for tid in token_ids[len(tokens) :])


def test_module_autoregressive_1digit():
    """Compile 1-digit adder to module, verify arithmetic via autoregressive generation."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)
    module.eval()

    test_cases = [
        ("1+1\n", "2"),
        ("2+3\n", "5"),
        ("0+0\n", "0"),
        ("4+5\n", "9"),
        ("7+2\n", "9"),
    ]
    for input_str, expected in test_cases:
        result = _module_generate(module, embedding.tokenizer, input_str)
        assert (
            result == expected
        ), f"For {input_str}: expected '{expected}' but got '{result}'"


# ---------------------------------------------------------------------------
# Test 7: Autoregressive generation (3-digit adder)
# ---------------------------------------------------------------------------


def test_module_autoregressive_3digit():
    """Compile 3-digit adder to module, verify arithmetic via autoregressive generation."""
    output_node, pos_encoding, embedding = create_network_parts()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    module = to_module(net, embedding, output_node)
    module.eval()

    test_cases = [
        ("1+2\n", "3"),
        ("12+34\n", "46"),
        ("99+1\n", "100"),
        ("100+200\n", "300"),
        ("456+123\n", "579"),
    ]
    for input_str, expected in test_cases:
        result = _module_generate(module, embedding.tokenizer, input_str)
        assert (
            result == expected
        ), f"For {input_str}: expected '{expected}' but got '{result}'"


# ---------------------------------------------------------------------------
# Test 8: state_dict roundtrip
# ---------------------------------------------------------------------------


def test_module_state_dict_roundtrip():
    """Save and load state_dict, verify identical output."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module1 = to_module(net, embedding, output_node)
    module1.eval()
    device = next(module1.parameters()).device

    tokens = ["<bos", "3", "+", "6", "\n"]
    token_ids = torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in tokens],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits1 = module1(token_ids)

    # Create a second module and load state_dict
    module2 = to_module(net, embedding, output_node)
    module2.load_state_dict(module1.state_dict())
    module2.eval()

    with torch.no_grad():
        logits2 = module2(token_ids)

    assert torch.equal(logits1, logits2), "state_dict roundtrip changed output"


# ---------------------------------------------------------------------------
# Test 9: Variable sequence lengths
# ---------------------------------------------------------------------------


def test_variable_sequence_lengths():
    """Module handles different sequence lengths without error."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)
    module.eval()
    device = next(module.parameters()).device

    for length in [1, 5, 10, 20]:
        token_ids = torch.zeros(length, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = module(token_ids)
        assert logits.shape == (
            length,
            embedding.table.shape[0],
        ), f"Wrong shape for length {length}: {logits.shape}"


# ---------------------------------------------------------------------------
# Test 10: Parameters are properly registered
# ---------------------------------------------------------------------------


def test_parameters_are_registered():
    """All layer weights are registered as nn.Parameters."""
    net, output_node, pos_encoding, embedding = _compile_1digit()
    module = to_module(net, embedding, output_node)

    n_layers = len(net.layers)

    # Count parameters: per layer = 4 attn (Q,K,V,O) + 4 ffn (W1,b1,W2,b2) = 8
    # Plus 1 for token_embedding.weight
    param_names = [name for name, _ in module.named_parameters()]

    # Check embedding
    assert "token_embedding.weight" in param_names

    # Check each layer has attn and ffn params
    for i in range(n_layers):
        prefix = f"layers.{i}"
        assert f"{prefix}.0.W_Q" in param_names, f"Missing W_Q in layer {i}"
        assert f"{prefix}.0.W_K" in param_names, f"Missing W_K in layer {i}"
        assert f"{prefix}.0.W_V" in param_names, f"Missing W_V in layer {i}"
        assert f"{prefix}.0.W_O" in param_names, f"Missing W_O in layer {i}"
        assert f"{prefix}.1.W1" in param_names, f"Missing W1 in layer {i}"
        assert f"{prefix}.1.b1" in param_names, f"Missing b1 in layer {i}"
        assert f"{prefix}.1.W2" in param_names, f"Missing W2 in layer {i}"
        assert f"{prefix}.1.b2" in param_names, f"Missing b2 in layer {i}"

    # Check buffers
    buffer_names = [name for name, _ in module.named_buffers()]
    assert "embedding_proj" in buffer_names
    assert "pos_proj" in buffer_names
    assert "constant_values" in buffer_names
    assert "pos_encoding" in buffer_names
    assert "output_gather_indices" in buffer_names
    assert "unembed_table" in buffer_names


# ---------------------------------------------------------------------------
# Test 11: ONNX export and inference roundtrip
# ---------------------------------------------------------------------------

onnxruntime = pytest.importorskip("onnxruntime")


def test_onnx_export_and_inference():
    """Export to ONNX via compile_to_onnx, load with onnxruntime, verify output matches PyTorch."""
    from torchwright.compiler.export import compile_to_onnx

    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        # Verify vocab sidecar was written
        vocab_path = os.path.join(tmpdir, "adder.vocab.json")
        assert os.path.exists(vocab_path)
        import json

        with open(vocab_path) as f:
            vocab_data = json.load(f)
        assert vocab_data["vocab"] == embedding.tokenizer.vocab

        # Compare PyTorch module vs ONNX
        net = forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=output_node,
            pos_encoding=pos_encoding,
            verbose=False,
        )
        module = to_module(net, embedding, output_node)
        module.eval()
        device = next(module.parameters()).device

        tokens = ["<bos", "1", "+", "2", "\n"]
        token_ids = torch.tensor(
            [embedding.tokenizer.get_token_id(t) for t in tokens],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            pt_logits = module(token_ids).cpu().numpy()

        session = onnxruntime.InferenceSession(onnx_path)
        onnx_logits = session.run(None, {"token_ids": token_ids.cpu().numpy()})[0]

        assert (
            pt_logits.argmax(axis=-1).tolist() == onnx_logits.argmax(axis=-1).tolist()
        ), f"Token mismatch: PT={pt_logits.argmax(axis=-1)}, ONNX={onnx_logits.argmax(axis=-1)}"
        import numpy as np

        assert np.allclose(
            pt_logits, onnx_logits, atol=1e-4
        ), f"ONNX max diff: {np.abs(pt_logits - onnx_logits).max():.6f}"


def test_onnx_repl_generate():
    """Test the standalone REPL generate function against an ONNX model."""
    from torchwright.compiler.export import compile_to_onnx
    from torchwright.compiler.repl import generate, _Vocab

    output_node, pos_encoding, embedding = _build_1digit()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "adder.onnx")
        compile_to_onnx(
            output_node,
            pos_encoding,
            embedding,
            onnx_path,
            d=D,
            d_head=D_HEAD,
            verbose=False,
        )

        import json

        with open(os.path.join(tmpdir, "adder.vocab.json")) as f:
            vocab = _Vocab(json.load(f)["vocab"])

        session = onnxruntime.InferenceSession(onnx_path)

        test_cases = [
            ("1+1\n", "2"),
            ("2+3\n", "5"),
            ("4+5\n", "9"),
        ]
        for input_str, expected in test_cases:
            result = "".join(generate(session, vocab, input_str))
            assert (
                result == expected
            ), f"For {input_str}: expected '{expected}' but got '{result}'"
