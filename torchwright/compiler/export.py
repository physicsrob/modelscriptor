"""Compile a graph to ONNX + vocab files."""

import json
import os

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import (
    CompiledTransformerModule,
    HeadlessTransformerModule,
    _CachedHeadlessWrapper,
    _CachedTransformerWrapper,
    to_module,
    to_headless_module,
)
from torchwright.graph import Node
from torchwright.graph.embedding import Embedding
from torchwright.graph.pos_encoding import PosEncoding


def _export_onnx_cached(
    module: CompiledTransformerModule,
    output_path: str,
    opset_version: int = 14,
    example_seq_len: int = 4,
) -> None:
    """Export with KV cache plumbing.

    Graph I/O:
        inputs:  token_ids, past_len, past_K_i, past_V_i  (i in 0..n_layers-1)
        outputs: logits, new_K_i, new_V_i                  (i in 0..n_layers-1)

    Prefill uses empty past tensors of shape (n_heads, 0, d_head) and
    past_len=0. Decode uses (n_heads, past_len, d_head).
    """
    module.eval()
    wrapper = _CachedTransformerWrapper(module)
    wrapper.eval()

    n_layers = len(module.layers)
    first_layer = module.layers[0]
    assert isinstance(first_layer, torch.nn.ModuleList)
    first_attn = first_layer[0]
    n_heads = first_attn.n_heads
    d_head = first_attn.d_head

    dummy_tokens = torch.zeros(example_seq_len, dtype=torch.long)
    dummy_past_len = torch.tensor(0, dtype=torch.long)
    dummy_past_kvs = []
    for _ in range(n_layers):
        dummy_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))
        dummy_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))

    input_names = ["token_ids", "past_len"]
    output_names = ["logits"]
    for i in range(n_layers):
        input_names += [f"past_K_{i}", f"past_V_{i}"]
        output_names += [f"new_K_{i}", f"new_V_{i}"]

    dynamic_axes: dict = {
        "token_ids": {0: "n_new"},
        "logits": {0: "n_new"},
    }
    for i in range(n_layers):
        dynamic_axes[f"past_K_{i}"] = {1: "n_past"}
        dynamic_axes[f"past_V_{i}"] = {1: "n_past"}
        dynamic_axes[f"new_K_{i}"] = {1: "n_total"}
        dynamic_axes[f"new_V_{i}"] = {1: "n_total"}

    torch.onnx.export(
        wrapper,
        (dummy_tokens, dummy_past_len, *dummy_past_kvs),
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def _export_headless_onnx_cached(
    module: HeadlessTransformerModule,
    output_path: str,
    opset_version: int = 14,
    example_seq_len: int = 4,
) -> None:
    """Export headless module with KV cache plumbing.

    Graph I/O:
        inputs:  inputs (n_new, d_input), past_len, past_K_i, past_V_i
        outputs: output (n_new, d_output), new_K_i, new_V_i

    Prefill uses empty past tensors of shape (n_heads, 0, d_head) and
    past_len=0. Decode uses (n_heads, past_len, d_head).
    """
    module.eval()
    wrapper = _CachedHeadlessWrapper(module)
    wrapper.eval()

    n_layers = len(module.layers)
    first_layer = module.layers[0]
    assert isinstance(first_layer, torch.nn.ModuleList)
    first_attn = first_layer[0]
    n_heads = first_attn.n_heads
    d_head = first_attn.d_head

    d_input = module.input_proj.shape[0]

    dummy_inputs = torch.zeros(example_seq_len, d_input, dtype=torch.float32)
    dummy_past_len = torch.tensor(0, dtype=torch.long)
    dummy_past_kvs = []
    for _ in range(n_layers):
        dummy_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))
        dummy_past_kvs.append(torch.zeros(n_heads, 0, d_head, dtype=torch.float32))

    input_names = ["inputs", "past_len"]
    output_names = ["output"]
    for i in range(n_layers):
        input_names += [f"past_K_{i}", f"past_V_{i}"]
        output_names += [f"new_K_{i}", f"new_V_{i}"]

    dynamic_axes: dict = {
        "inputs": {0: "n_new"},
        "output": {0: "n_new"},
    }
    for i in range(n_layers):
        dynamic_axes[f"past_K_{i}"] = {1: "n_past"}
        dynamic_axes[f"past_V_{i}"] = {1: "n_past"}
        dynamic_axes[f"new_K_{i}"] = {1: "n_total"}
        dynamic_axes[f"new_V_{i}"] = {1: "n_total"}

    torch.onnx.export(
        wrapper,
        (dummy_inputs, dummy_past_len, *dummy_past_kvs),
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def _vocab_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".vocab.json"


def _meta_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".meta.json"


def compile_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    embedding: Embedding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    verbose: bool = True,
) -> None:
    """Compile a computation graph to ONNX + vocab files.

    Produces two files:
        <output_path>          -- the ONNX model
        <output_path>.vocab.json -- the vocabulary (needed by the REPL)

    Args:
        output_node: The graph node whose value is the model output.
        pos_encoding: Positional encoding node.
        embedding: Embedding node (provides tokenizer + embedding table).
        output_path: Path for the .onnx output file.
        d: Residual stream dimension.
        d_head: Attention head dimension.
        max_seq_len: Maximum sequence length.
        verbose: Print progress.
    """
    if verbose:
        print("Compiling graph...")
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        device=None,
    )

    if verbose:
        print("Converting to nn.Module...")
    module = to_module(net, embedding, output_node, max_seq_len=max_seq_len, device=None)
    module.eval()

    n_params = sum(p.numel() for p in module.parameters())
    n_layers = len(module.layers)

    if verbose:
        print(f"Exporting ONNX: {n_layers} layers, {n_params:,} parameters")
    _export_onnx_cached(module, output_path)

    vocab_path = _vocab_path_for(output_path)
    with open(vocab_path, "w") as f:
        json.dump({"vocab": embedding.tokenizer.vocab}, f)

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {vocab_path}")


def compile_headless(
    output_node: Node,
    pos_encoding: PosEncoding,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 100,
    verbose: bool = True,
    device: str = "auto",
) -> HeadlessTransformerModule:
    """Compile a graph with raw float I/O to a headless nn.Module.

    Unlike ``compile_to_onnx``, this does not require an Embedding and
    returns a ``HeadlessTransformerModule`` directly (no ONNX export).

    Args:
        output_node: The graph node whose value is the model output.
        pos_encoding: Positional encoding node.
        d: Residual stream dimension.
        d_head: Attention head dimension.
        max_seq_len: Maximum sequence length.
        max_layers: Safety limit on number of compiled layers.
        verbose: Print progress.
        device: Target device.

    Returns:
        A HeadlessTransformerModule ready for inference.
    """
    if verbose:
        print("Compiling graph...")
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
    )

    if verbose:
        n_layers = len(net.layers)
        print(f"Converting to headless module ({n_layers} layers)...")

    return to_headless_module(
        net, output_node, max_seq_len=max_seq_len, device=device
    )


def compile_headless_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 100,
    verbose: bool = True,
) -> None:
    """Compile a graph with raw float I/O to ONNX + meta sidecar.

    Produces two files:
        <output_path>          -- the ONNX model
        <output_path>.meta.json -- input column names (alphabetical order)

    The module being exported must be CPU-resident — pass ``device=None``
    paths through ``to_headless_module``. If you have a CUDA-resident
    module, call ``.cpu()`` on it first; this exporter does not insert any
    device-cast nodes into the traced graph.

    Args:
        output_node: The graph node whose value is the model output.
        pos_encoding: Positional encoding node.
        output_path: Path for the .onnx output file.
        d: Residual stream dimension.
        d_head: Attention head dimension.
        max_seq_len: Maximum sequence length.
        max_layers: Safety limit on number of compiled layers.
        verbose: Print progress.
    """
    if verbose:
        print("Compiling graph...")
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
    )

    if verbose:
        print("Converting to headless module...")
    module = to_headless_module(
        net, output_node, max_seq_len=max_seq_len, device=None
    )
    module.eval()

    n_params = sum(p.numel() for p in module.parameters())
    n_layers = len(module.layers)

    if verbose:
        print(f"Exporting ONNX: {n_layers} layers, {n_params:,} parameters")
    _export_headless_onnx_cached(module, output_path)

    meta_path = _meta_path_for(output_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "format": "torchwright.headless.v1",
                "input_names": list(module.input_names),
            },
            f,
        )

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {meta_path}")
