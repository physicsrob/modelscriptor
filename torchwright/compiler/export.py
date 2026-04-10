"""Compile a graph to ONNX + vocab files."""

import json
import os
import time

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import (
    CompiledTransformerModule,
    HeadlessTransformerModule,
    to_module,
    to_headless_module,
)
from torchwright.graph import Node
from torchwright.graph.embedding import Embedding
from torchwright.graph.pos_encoding import PosEncoding


def _export_onnx(
    module: CompiledTransformerModule,
    output_path: str,
    opset_version: int = 14,
    example_seq_len: int = 10,
) -> None:
    module.eval()
    dummy_input = torch.zeros(example_seq_len, dtype=torch.long)
    torch.onnx.export(
        module,
        (dummy_input,),
        output_path,
        opset_version=opset_version,
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "seq_len"},
            "logits": {0: "seq_len"},
        },
        dynamo=False,
    )


def _vocab_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".vocab.json"


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
    _export_onnx(module, output_path)

    vocab_path = _vocab_path_for(output_path)
    with open(vocab_path, "w") as f:
        json.dump({"vocab": embedding.tokenizer.vocab}, f)

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {vocab_path}")


def export_headless_to_onnx(
    module: HeadlessTransformerModule,
    output_path: str,
    example_seq_len: int = 10,
    opset_version: int = 14,
    verbose: bool = True,
) -> None:
    """Export a HeadlessTransformerModule to ONNX (no embedding required).

    Writes two files:
        <output_path>                  -- the ONNX model
        <output_path>.input_names.json -- ordered list of input column names

    The ONNX model has a single float input tensor named ``inputs`` with
    shape ``(seq_len, d_input)`` and a single float output tensor named
    ``outputs`` with shape ``(seq_len, d_output)``.  The input column
    ordering matches ``module.input_names`` (alphabetical by graph
    InputNode name); the sidecar JSON records this so consumers know how
    to populate each column.
    """
    module.eval()
    d_input = len(module.input_names)
    dummy_input = torch.zeros(example_seq_len, d_input, dtype=torch.float32)

    n_params = sum(p.numel() for p in module.parameters())
    n_layers = len(module.layers)
    if verbose:
        print(f"Exporting ONNX: {n_layers} layers, {n_params:,} parameters")

    # Trace with dynamo=True so initializers stay in memory (no multi-file
    # disk spill).  Then sparsify mostly-zero initializers in-place and save
    # the consolidated model.
    seq_dim = torch.export.Dim("seq_len", min=1, max=8192)
    t0 = time.perf_counter()
    prog = torch.onnx.export(
        module,
        (dummy_input,),
        None,  # in-memory ONNXProgram
        dynamo=True,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_shapes={"inputs": {0: seq_dim}},
        verbose=False,
    )
    t_trace = time.perf_counter() - t0

    model = prog.model_proto

    t0 = time.perf_counter()
    n_sparse, n_dense_kept, bytes_saved = _sparsify_initializers(model, verbose=verbose)
    t_sparsify = time.perf_counter() - t0

    t0 = time.perf_counter()
    onnx.save_model(model, output_path)
    t_save = time.perf_counter() - t0

    if verbose:
        print(
            f"Phases: trace {t_trace:.2f}s, "
            f"sparsify {t_sparsify:.2f}s, save {t_save:.2f}s"
        )
        print(
            f"Sparsified {n_sparse} initializers, kept {n_dense_kept} dense, "
            f"saved ~{bytes_saved/1e6:.0f} MB"
        )

    sidecar_path = output_path + ".input_names.json"
    with open(sidecar_path, "w") as f:
        json.dump({"input_names": list(module.input_names)}, f)

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {sidecar_path}")


def _sparsify_initializers(model, verbose: bool = True):
    """Replace mostly-zero float32 initializers with SparseTensorProto in place.

    COO sparse storage uses 12 bytes per nonzero (8B int64 index + 4B fp32
    value) vs 4 bytes per element dense, so sparse is smaller when the
    zero fraction exceeds ~67%. We use 75% so conversion is meaningfully
    smaller, not just a wash.

    Returns:
        (n_sparsified, n_kept_dense, bytes_saved_estimate)
    """
    SPARSITY_THRESHOLD = 0.75
    MIN_ELEMENTS = 1024  # skip tiny tensors — per-init overhead dominates

    n_sparse = 0
    n_dense_kept = 0
    bytes_saved = 0

    dense_keep = []
    sparse_new = []
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            dense_keep.append(init)
            continue

        n_elem = 1
        for d in init.dims:
            n_elem *= d
        if n_elem < MIN_ELEMENTS:
            dense_keep.append(init)
            continue

        # Zero-copy view into raw_data when present (the common case for
        # initializers exported by torch.onnx.export); otherwise fall back
        # to the helper which handles the legacy float_data/double_data
        # representations.
        if init.raw_data:
            flat = np.frombuffer(init.raw_data, dtype=np.float32)
        else:
            flat = numpy_helper.to_array(init).reshape(-1)

        nnz = int(np.count_nonzero(flat))
        zero_frac = 1.0 - nnz / n_elem
        if zero_frac < SPARSITY_THRESHOLD:
            n_dense_kept += 1
            dense_keep.append(init)
            continue

        nz_idx = np.flatnonzero(flat)  # int64 on 64-bit systems
        nz_val = flat[nz_idx]

        values_tp = helper.make_tensor(
            name=init.name,  # SparseTensorProto identified by values.name
            data_type=TensorProto.FLOAT,
            dims=[nnz],
            vals=nz_val.astype(np.float32, copy=False).tobytes(),
            raw=True,
        )
        indices_tp = helper.make_tensor(
            name=init.name + "__indices",
            data_type=TensorProto.INT64,
            dims=[nnz],
            vals=nz_idx.astype(np.int64, copy=False).tobytes(),
            raw=True,
        )
        sparse_tp = helper.make_sparse_tensor(
            values=values_tp,
            indices=indices_tp,
            dims=list(init.dims),
        )
        sparse_new.append(sparse_tp)

        n_sparse += 1
        bytes_saved += max(0, 4 * n_elem - 12 * nnz)

    model.graph.ClearField("initializer")
    model.graph.initializer.extend(dense_keep)
    model.graph.ClearField("sparse_initializer")
    model.graph.sparse_initializer.extend(sparse_new)

    return n_sparse, n_dense_kept, bytes_saved


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
    t0 = time.perf_counter()
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
    )
    t_forward = time.perf_counter() - t0

    if verbose:
        n_layers = len(net.layers)
        print(f"Converting to headless module ({n_layers} layers)...")

    t0 = time.perf_counter()
    module = to_headless_module(
        net, output_node, max_seq_len=max_seq_len, device=device
    )
    t_convert = time.perf_counter() - t0

    if verbose:
        print(
            f"Phases: forward_compile {t_forward:.2f}s, "
            f"to_headless_module {t_convert:.2f}s"
        )

    return module
