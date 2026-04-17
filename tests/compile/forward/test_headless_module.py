"""Tests for the headless streaming cached ONNX exporter.

Parity oracle: HeadlessTransformer.compute() against
compile_headless_to_onnx + OnnxHeadlessModule.  Covers basic graph
operations, input ordering, and multi-position evaluation.  KV cache
prefill/decode specifics live in test_headless_onnx.py.
"""

import os
import tempfile

import pytest
import torch

from torchwright.compiler.export import compile_headless_to_onnx
from torchwright.compiler.forward.compile import forward_compile
from torchwright.ops.arithmetic_ops import add, compare, signed_multiply
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.map_select import select

onnxruntime = pytest.importorskip("onnxruntime")

D = 256
D_HEAD = 16


def _export_and_load(output_node, pos_encoding, tmpdir):
    """Compile to ONNX in tmpdir and return an OnnxHeadlessModule."""
    from torchwright.compiler.onnx_load import OnnxHeadlessModule

    onnx_path = os.path.join(tmpdir, "model.onnx")
    compile_headless_to_onnx(
        output_node,
        pos_encoding,
        onnx_path,
        d=D,
        d_head=D_HEAD,
        max_seq_len=32,
        verbose=False,
    )
    return OnnxHeadlessModule(onnx_path)


def _compute_reference(output_node, pos_encoding, input_values, n_pos):
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=False,
    )
    result = net.compute(n_pos=n_pos, input_values=input_values)
    return result[output_node].cpu()


# ---------------------------------------------------------------------------
# Test 1: select(compare(a, 0), b, a) — mixed boolean and data flow
# ---------------------------------------------------------------------------


def test_headless_onnx_select_matches_compute():
    a = create_input("a", 1)
    b = create_input("b", 1)
    cond = compare(a, 0.0)
    out = select(cond, b, a)
    pos_encoding = create_pos_encoding()

    a_vals = torch.tensor([[5.0], [-3.0], [0.5]])
    b_vals = torch.tensor([[10.0], [20.0], [30.0]])

    expected = _compute_reference(
        out, pos_encoding, {"a": a_vals, "b": b_vals}, n_pos=3
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        module = _export_and_load(out, pos_encoding, tmpdir)
        # Inputs alphabetically ordered: a, b
        inputs = torch.cat([a_vals, b_vals], dim=1)
        actual = module(inputs)

    assert torch.allclose(
        actual, expected, atol=1e-4
    ), f"max diff: {(actual - expected).abs().max().item():.6f}"


# ---------------------------------------------------------------------------
# Test 2: signed_multiply
# ---------------------------------------------------------------------------


def test_headless_onnx_signed_multiply():
    a = create_input("a", 1)
    b = create_input("b", 1)
    out = signed_multiply(a, b, max_abs1=10, max_abs2=10)
    pos_encoding = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        module = _export_and_load(out, pos_encoding, tmpdir)
        test_cases = [
            ([3.0, 4.0], 12.0),
            ([5.0, 2.0], 10.0),
            ([-3.0, 4.0], -12.0),
            ([0.0, 7.0], 0.0),
        ]
        for (a_val, b_val), expected in test_cases:
            inputs = torch.tensor([[a_val, b_val]])
            result = module(inputs)
            assert (
                abs(result.item() - expected) < 0.5
            ), f"{a_val} * {b_val}: expected {expected}, got {result.item():.2f}"


# ---------------------------------------------------------------------------
# Test 3: compare (scalar condition)
# ---------------------------------------------------------------------------


def test_headless_onnx_compare():
    x = create_input("x", 1)
    out = compare(x, 0.0)
    pos_encoding = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        module = _export_and_load(out, pos_encoding, tmpdir)
        test_cases = [
            (5.0, 1.0),
            (-3.0, -1.0),
            (100.0, 1.0),
            (-0.5, -1.0),
        ]
        for x_val, expected in test_cases:
            inputs = torch.tensor([[x_val]])
            result = module(inputs)
            assert (
                abs(result.item() - expected) < 0.1
            ), f"compare({x_val}, 0): expected {expected}, got {result.item():.2f}"


# ---------------------------------------------------------------------------
# Test 4: Multi-position independent evaluation
# ---------------------------------------------------------------------------


def test_headless_onnx_multi_position():
    x = create_input("x", 1)
    out = compare(x, 0.0)
    pos_encoding = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        module = _export_and_load(out, pos_encoding, tmpdir)
        inputs = torch.tensor([[3.0], [-2.0], [0.5], [-0.5], [10.0]])
        expected = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [1.0]])
        actual = module(inputs)

    assert torch.allclose(
        actual, expected, atol=0.1
    ), f"multi-position mismatch: {actual.squeeze().tolist()}"


# ---------------------------------------------------------------------------
# Test 5: input_names are alphabetical
# ---------------------------------------------------------------------------


def test_headless_onnx_input_names_alphabetical():
    z = create_input("zebra", 1)
    a = create_input("alpha", 1)
    out = add(z, a)
    pos_encoding = create_pos_encoding()

    with tempfile.TemporaryDirectory() as tmpdir:
        module = _export_and_load(out, pos_encoding, tmpdir)
        assert module.input_names == ["alpha", "zebra"]
