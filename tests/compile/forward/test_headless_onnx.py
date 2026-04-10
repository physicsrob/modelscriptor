"""ONNX export + KV cache tests for HeadlessTransformerModule.

Separate from test_headless_module.py so that onnxruntime-skip does not
propagate to the file's full collection.
"""

import json
import os
import tempfile

import pytest
import torch

from torchwright.compiler.export import (
    _meta_path_for,
    compile_headless_to_onnx,
)
from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import _CachedHeadlessWrapper, to_headless_module
from torchwright.ops.arithmetic_ops import add
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

onnxruntime = pytest.importorskip("onnxruntime")

D = 256
D_HEAD = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_past_feeds(n_layers: int, n_heads: int, d_head: int) -> dict:
    import numpy as np

    feeds = {"past_len": np.array(0, dtype=np.int64)}
    for i in range(n_layers):
        feeds[f"past_K_{i}"] = np.zeros((n_heads, 0, d_head), dtype=np.float32)
        feeds[f"past_V_{i}"] = np.zeros((n_heads, 0, d_head), dtype=np.float32)
    return feeds


def _discover_meta(session):
    inputs = {inp.name: inp for inp in session.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    shape0 = inputs["past_K_0"].shape
    return n_layers, int(shape0[0]), int(shape0[2])


def _build_sample_graph():
    """Small headless graph with two float inputs and an add."""
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = add(a, b)
    return out, create_pos_encoding()


def _compile_sample_module():
    out, pos = _build_sample_graph()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        pos_encoding=pos,
        verbose=False,
        device=None,
    )
    module = to_headless_module(net, out, device=None)
    module.eval()
    return module, out, pos


# ---------------------------------------------------------------------------
# Test 1: PT-level wrapper — empty past matches non-cached module
# ---------------------------------------------------------------------------


def test_cached_headless_wrapper_empty_past_matches_module():
    module, _, _ = _compile_sample_module()
    wrapper = _CachedHeadlessWrapper(module)
    wrapper.eval()

    first_attn = module.layers[0][0]
    n_heads = first_attn.n_heads
    d_head = first_attn.d_head
    n_layers = len(module.layers)

    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )

    past_len = torch.tensor(0, dtype=torch.long)
    past_kvs = []
    for _ in range(n_layers):
        past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))
        past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))

    with torch.no_grad():
        expected = module(inputs)
        out = wrapper(inputs, past_len, *past_kvs)
    actual = out[0]

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Empty-past wrapper diff: {(actual - expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: PT-level wrapper — prefill then decode equals full forward
# ---------------------------------------------------------------------------


def test_cached_headless_wrapper_decode_step_matches_full():
    module, _, _ = _compile_sample_module()
    wrapper = _CachedHeadlessWrapper(module)
    wrapper.eval()

    first_attn = module.layers[0][0]
    n_heads = first_attn.n_heads
    d_head = first_attn.d_head
    n_layers = len(module.layers)

    inputs = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )

    empty_past_kvs = []
    for _ in range(n_layers):
        empty_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))
        empty_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))

    with torch.no_grad():
        expected = module(inputs)

        # Prefill first 4 rows
        prefill_out = wrapper(
            inputs[:4], torch.tensor(0, dtype=torch.long), *empty_past_kvs
        )
        captured_kvs = list(prefill_out[1:])

        # Decode row 4
        decode_out = wrapper(
            inputs[4:5], torch.tensor(4, dtype=torch.long), *captured_kvs
        )
        decoded = decode_out[0]

    assert torch.allclose(decoded[0], expected[-1], atol=1e-5), (
        f"Decode step diff: {(decoded[0] - expected[-1]).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: ONNX export — prefill matches PT
# ---------------------------------------------------------------------------


def test_headless_onnx_export_prefill_matches_pt():
    import numpy as np

    out_node, pos = _build_sample_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "headless.onnx")
        compile_headless_to_onnx(
            out_node, pos, onnx_path, d=D, d_head=D_HEAD, verbose=False
        )

        assert os.path.exists(_meta_path_for(onnx_path))

        net = forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out_node,
            pos_encoding=pos,
            verbose=False,
            device=None,
        )
        module = to_headless_module(net, out_node, device=None)
        module.eval()

        inputs = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            pt_output = module(inputs).cpu().numpy()

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, n_heads, d_head = _discover_meta(session)
        feeds = {"inputs": inputs.numpy()}
        feeds.update(_empty_past_feeds(n_layers, n_heads, d_head))
        onnx_output = session.run(["output"], feeds)[0]

        assert np.allclose(pt_output, onnx_output, atol=1e-4), (
            f"Headless ONNX prefill diff: {np.abs(pt_output - onnx_output).max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4: ONNX export — decode step matches PT full forward
# ---------------------------------------------------------------------------


def test_headless_onnx_decode_step_matches_pt():
    import numpy as np

    out_node, pos = _build_sample_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "headless.onnx")
        compile_headless_to_onnx(
            out_node, pos, onnx_path, d=D, d_head=D_HEAD, verbose=False
        )

        net = forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out_node,
            pos_encoding=pos,
            verbose=False,
            device=None,
        )
        module = to_headless_module(net, out_node, device=None)
        module.eval()

        inputs = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            pt_output = module(inputs).cpu().numpy()

        session = onnxruntime.InferenceSession(onnx_path)
        n_layers, n_heads, d_head = _discover_meta(session)

        inputs_np = inputs.numpy()

        # Prefill rows 0..3
        feeds = {"inputs": inputs_np[:4]}
        feeds.update(_empty_past_feeds(n_layers, n_heads, d_head))
        out_names = ["output"]
        for i in range(n_layers):
            out_names += [f"new_K_{i}", f"new_V_{i}"]
        outputs = session.run(out_names, feeds)

        past_K = [outputs[1 + 2 * i] for i in range(n_layers)]
        past_V = [outputs[1 + 2 * i + 1] for i in range(n_layers)]

        feeds = {
            "inputs": inputs_np[4:5],
            "past_len": np.array(4, dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"past_K_{i}"] = past_K[i]
            feeds[f"past_V_{i}"] = past_V[i]
        outputs = session.run(out_names, feeds)
        decode_output = outputs[0]

        assert np.allclose(pt_output[-1], decode_output[0], atol=1e-4), (
            f"Decode diff: {np.abs(pt_output[-1] - decode_output[0]).max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: meta sidecar records input names in alphabetical order
# ---------------------------------------------------------------------------


def test_headless_onnx_sidecar_input_names_order():
    zebra = create_input("zebra", 1)
    alpha = create_input("alpha", 1)
    middle = create_input("middle", 1)
    out_node = add(add(zebra, alpha), middle)
    pos = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "headless.onnx")
        compile_headless_to_onnx(
            out_node, pos, onnx_path, d=D, d_head=D_HEAD, verbose=False
        )

        meta_path = _meta_path_for(onnx_path)
        with open(meta_path) as f:
            data = json.load(f)

        assert data["format"] == "torchwright.headless.v1"
        assert data["input_names"] == ["alpha", "middle", "zebra"]

        session = onnxruntime.InferenceSession(onnx_path)
        inputs_info = {inp.name: inp for inp in session.get_inputs()}
        assert int(inputs_info["inputs"].shape[1]) == len(data["input_names"])
