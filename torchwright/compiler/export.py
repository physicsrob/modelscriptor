"""Compile a graph to ONNX + vocab files."""

import json
import os
import time
from typing import List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.module import (
    CompiledTransformerModule,
    HeadlessTransformerModule,
    _CachedHeadlessWrapper,
    _CachedTransformerWrapper,
    _compute_pos_encoding,
    to_headless_module,
    to_module,
)
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Concatenate, Embedding, LiteralValue, Node
from torchwright.graph.misc import InputNode
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


HEADLESS_META_FORMAT = "torchwright.headless.v1"


def _write_headless_meta(
    onnx_path: str, input_names: List[str], cached: bool
) -> str:
    meta_path = _meta_path_for(onnx_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "format": HEADLESS_META_FORMAT,
                "input_names": list(input_names),
                "cached": cached,
            },
            f,
        )
    return meta_path


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


def emit_headless_onnx(
    compiled: HeadlessTransformer,
    output_node: Node,
    output_path: str,
    max_seq_len: int = 512,
    verbose: bool = True,
    free_layers: bool = True,
) -> None:
    """Emit an ONNX model directly from a HeadlessTransformer.

    Mirrors ``HeadlessTransformerModule.forward`` by walking
    ``compiled.layers`` and streaming each layer's weights into
    ``TensorProto`` initializers while emitting the corresponding ONNX
    nodes.  This skips the ``nn.Module`` intermediate (and therefore
    ``torch.onnx.export``) so the convert-phase memory peak is no longer
    doubled by a second copy of every weight tensor in ``nn.Parameter``
    form.

    When ``free_layers=True`` the compiled layer's weight attributes are
    nulled out as soon as their initializers are built, so Python can
    reclaim each layer's torch storage before the next layer runs.

    Writes two files:
        ``<output_path>``              — the ONNX model
        ``<stem>.meta.json``           — ordered input column names + format tag

    ``cached`` in the sidecar is ``false``: this path emits a plain
    feed-forward graph (no KV cache inputs/outputs).
    """
    assert compiled.residual_assignment is not None
    d = compiled.d
    d_head = compiled.d_head
    n_heads = d // d_head

    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].mlp.out_state

    # --- Extract I/O metadata (mirrors to_headless_module header) ------
    t0 = time.perf_counter()

    input_nodes_list: List[tuple] = []  # (name, indices)
    pos_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, InputNode):
            input_nodes_list.append((node.name, indices))
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for i, idx in enumerate(indices):
                constant_values[idx] = float(node.value[i])
        elif isinstance(node, (Concatenate, Embedding)):
            pass

    assert len(input_nodes_list) > 0, "No InputNode found in residual assignment"
    assert pos_indices is not None, "No PosEncoding node found in residual assignment"

    input_nodes_list.sort(key=lambda x: x[0])
    input_names = [name for name, _ in input_nodes_list]

    all_input_indices: List[int] = []
    for _, idx in input_nodes_list:
        all_input_indices.extend(idx)
    d_input = len(all_input_indices)

    input_proj = np.zeros((d_input, d), dtype=np.float32)
    for i, idx in enumerate(all_input_indices):
        input_proj[i, idx] = 1.0

    d_pos = len(pos_indices)
    pos_proj = np.zeros((d_pos, d), dtype=np.float32)
    for i, idx in enumerate(pos_indices):
        pos_proj[i, idx] = 1.0

    pos_encoding_buf = (
        _compute_pos_encoding(d_pos, max_seq_len)
        .numpy()
        .astype(np.float32, copy=False)
    )

    # -1000 above the diagonal, 0 on/below — mirrors _AttentionLayer.forward
    # where a bool mask is filled with -1000.  Using a float adder avoids the
    # Where/bool-mask path entirely.
    causal_mask = np.triu(
        np.full((max_seq_len, max_seq_len), -1000.0, dtype=np.float32), k=1
    )

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)
    d_output = len(output_gather_indices)

    t_extract = time.perf_counter() - t0

    # --- Build ONNX nodes and initializers -----------------------------
    t0 = time.perf_counter()

    initializers: List[TensorProto] = []
    nodes: List = []

    def add_init_float(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.FLOAT,
                dims=list(arr.shape),
                vals=arr.tobytes(),
                raw=True,
            )
        )

    def add_init_int64(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
        initializers.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.INT64,
                dims=list(arr.shape),
                vals=arr.tobytes(),
                raw=True,
            )
        )

    def add_node(op: str, ins: List[str], outs: List[str], **attrs) -> None:
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Static initializers for the preamble
    add_init_float("input_proj", input_proj)
    add_init_float("pos_proj", pos_proj)
    add_init_float("constant_values", constant_values)
    add_init_float("pos_encoding_full", pos_encoding_buf)
    add_init_float("causal_mask_full", causal_mask)
    add_init_int64("output_gather_indices_init", output_gather_indices)

    # Shape-math constants (reused across ops where shape matches).
    add_init_int64("_zero_1d", np.array([0], dtype=np.int64))
    add_init_int64("_zeros_2d", np.array([0, 0], dtype=np.int64))
    add_init_int64("_axes_01", np.array([0, 1], dtype=np.int64))
    add_init_int64(
        "_qkv_view_shape", np.array([0, n_heads, d_head], dtype=np.int64)
    )
    # For the fused attention output projection: reshape
    # (t, n_heads, d_head) → (t, d).
    add_init_int64("_ctx_flat_shape", np.array([0, d], dtype=np.int64))

    # --- Preamble -------------------------------------------------------
    add_node("Shape", ["inputs"], ["input_shape"])
    add_node("Gather", ["input_shape", "_zero_1d"], ["seq_len_1d"], axis=0)
    add_node(
        "Slice",
        ["pos_encoding_full", "_zero_1d", "seq_len_1d", "_zero_1d"],
        ["pos"],
    )
    add_node("MatMul", ["inputs", "input_proj"], ["inp_res"])
    add_node("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add_node("Add", ["inp_res", "pos_res"], ["res_pi"])
    add_node("Add", ["res_pi", "constant_values"], ["res_0"])
    add_node("Concat", ["seq_len_1d", "seq_len_1d"], ["mask_ends"], axis=0)
    add_node(
        "Slice",
        ["causal_mask_full", "_zeros_2d", "mask_ends", "_axes_01"],
        ["mask_2d"],
    )
    add_node("Unsqueeze", ["mask_2d", "_zero_1d"], ["mask_3d"])

    # --- Per-layer emit -------------------------------------------------
    current_res = "res_0"
    n_layers = len(compiled.layers)
    for i, layer in enumerate(compiled.layers):
        attn = layer.attn.attn
        # permute().reshape() forces a contiguous copy (needed to fuse heads).
        # .cpu() is a no-op if already on CPU; otherwise copies from device.
        W_Q = (
            attn.query_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy()
        )
        W_K = (
            attn.key_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy()
        )
        W_V = (
            attn.value_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy()
        )
        # (n_heads, d_head, d) → (d, d).  Same bytes, different shape,
        # so a single (t, d) @ (d, d) MatMul replaces the per-head
        # MatMul + ReduceSum form.
        W_O = attn.output_matrix.reshape(d, d).contiguous().cpu().numpy()
        W1 = layer.mlp.linear1.output_matrix.cpu().numpy()
        b1 = layer.mlp.linear1.output_bias.cpu().numpy()
        W2 = layer.mlp.linear2.output_matrix.cpu().numpy()
        b2 = layer.mlp.linear2.output_bias.cpu().numpy()

        p = f"l{i}"
        add_init_float(f"{p}_WQ", W_Q)
        add_init_float(f"{p}_WK", W_K)
        add_init_float(f"{p}_WV", W_V)
        add_init_float(f"{p}_WO", W_O)
        add_init_float(f"{p}_W1", W1)
        add_init_float(f"{p}_b1", b1)
        add_init_float(f"{p}_W2", W2)
        add_init_float(f"{p}_b2", b2)

        current_res = _emit_layer_nodes(
            nodes, i, current_res, d, d_head, n_heads
        )

        if free_layers:
            attn.query_matrix = None
            attn.key_matrix = None
            attn.value_matrix = None
            attn.output_matrix = None
            layer.mlp.linear1.output_matrix = None
            layer.mlp.linear1.output_bias = None
            layer.mlp.linear2.output_matrix = None
            layer.mlp.linear2.output_bias = None
        del W_Q, W_K, W_V, W_O, W1, b1, W2, b2

    # --- Postamble ------------------------------------------------------
    add_node(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["outputs"],
        axis=1,
    )

    t_emit = time.perf_counter() - t0

    # --- Build graph + model -------------------------------------------
    t0 = time.perf_counter()
    inputs_vi = helper.make_tensor_value_info(
        "inputs", TensorProto.FLOAT, ["seq_len", d_input]
    )
    outputs_vi = helper.make_tensor_value_info(
        "outputs", TensorProto.FLOAT, ["seq_len", d_output]
    )
    graph = helper.make_graph(
        nodes,
        "headless_transformer",
        inputs=[inputs_vi],
        outputs=[outputs_vi],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
        producer_name="torchwright",
    )
    t_build = time.perf_counter() - t0

    # --- Sparsify + save -----------------------------------------------
    t0 = time.perf_counter()
    n_sparse, n_dense_kept, bytes_saved = _sparsify_initializers(
        model, verbose=verbose
    )
    t_sparsify = time.perf_counter() - t0

    t0 = time.perf_counter()
    onnx.save_model(model, output_path)
    t_save = time.perf_counter() - t0

    sidecar_path = _write_headless_meta(output_path, input_names, cached=False)

    if verbose:
        print(
            f"Phases: extract {t_extract:.2f}s, emit {t_emit:.2f}s, "
            f"build {t_build:.2f}s, sparsify {t_sparsify:.2f}s, "
            f"save {t_save:.2f}s"
        )
        print(
            f"{n_layers} layers, "
            f"sparsified {n_sparse} initializers, kept {n_dense_kept} dense, "
            f"saved ~{bytes_saved/1e6:.0f} MB"
        )
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


# ---------------------------------------------------------------------------
# Streaming compile-and-emit: build ONNX initializers layer-by-layer while
# forward_compile is still running, freeing each layer's dense torch tensors
# as we go.  Lets 320x200 tex-64 game graphs fit in ~5 GB of RAM instead of
# accumulating 6*d^2*N bytes of dense weights.
# ---------------------------------------------------------------------------


_SPARSITY_THRESHOLD = 0.75
_MIN_SPARSE_ELEMENTS = 1024  # skip tiny tensors — per-init overhead dominates


def _tensor_to_proto(name: str, arr: np.ndarray):
    """Convert a float32 numpy array to (dense_tp, sparse_tp).

    Exactly one of the returned values is non-None.  Float tensors with
    zero-fraction >= 75% and at least 1024 elements become
    SparseTensorProto (COO, flat int64 indices).  Everything else is a
    dense TensorProto with ``raw_data``.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n_elem = arr.size
    dims = list(arr.shape)

    if n_elem < _MIN_SPARSE_ELEMENTS:
        dense_tp = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=dims,
            vals=arr.tobytes(),
            raw=True,
        )
        return dense_tp, None

    flat = arr.reshape(-1)
    nnz = int(np.count_nonzero(flat))
    zero_frac = 1.0 - nnz / n_elem
    if zero_frac < _SPARSITY_THRESHOLD:
        dense_tp = helper.make_tensor(
            name=name,
            data_type=TensorProto.FLOAT,
            dims=dims,
            vals=arr.tobytes(),
            raw=True,
        )
        return dense_tp, None

    nz_idx = np.flatnonzero(flat)
    nz_val = flat[nz_idx]
    values_tp = helper.make_tensor(
        name=name,  # SparseTensorProto identified by values.name
        data_type=TensorProto.FLOAT,
        dims=[nnz],
        vals=nz_val.astype(np.float32, copy=False).tobytes(),
        raw=True,
    )
    indices_tp = helper.make_tensor(
        name=name + "__indices",
        data_type=TensorProto.INT64,
        dims=[nnz],
        vals=nz_idx.astype(np.int64, copy=False).tobytes(),
        raw=True,
    )
    sparse_tp = helper.make_sparse_tensor(
        values=values_tp, indices=indices_tp, dims=dims,
    )
    return None, sparse_tp


def _append_proto(
    dense_tp,
    sparse_tp,
    dense_inits: list,
    sparse_inits: list,
) -> None:
    if dense_tp is not None:
        dense_inits.append(dense_tp)
    else:
        sparse_inits.append(sparse_tp)


def _emit_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
) -> str:
    """Emit ONNX nodes for one transformer layer.  Returns the name of
    the updated residual stream tensor to feed into the next layer.

    Attention-output projection is fused: ``ctx`` of shape
    ``(n_heads, t, d_head)`` is transposed + reshaped to ``(t, d)`` and
    projected through a single ``(d, d)`` W_O matmul — no
    ``(n_heads, t, d)`` intermediate.  Mathematically identical to the
    per-head matmul + ReduceSum form, just written the way the original
    multi-head attention paper does it.
    """
    p = f"l{layer_idx}"

    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Attention sublayer
    node("MatMul", [current_res, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", "_qkv_view_shape"], [f"{p}_Q_view"])
    node("Transpose", [f"{p}_Q_view"], [f"{p}_Q"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WK"], [f"{p}_K_flat"])
    node("Reshape", [f"{p}_K_flat", "_qkv_view_shape"], [f"{p}_K_view"])
    node("Transpose", [f"{p}_K_view"], [f"{p}_K"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WV"], [f"{p}_V_flat"])
    node("Reshape", [f"{p}_V_flat", "_qkv_view_shape"], [f"{p}_V_view"])
    node("Transpose", [f"{p}_V_view"], [f"{p}_V"], perm=[1, 0, 2])

    node("Transpose", [f"{p}_K"], [f"{p}_K_T"], perm=[0, 2, 1])
    node("MatMul", [f"{p}_Q", f"{p}_K_T"], [f"{p}_logits"])
    node("Add", [f"{p}_logits", "mask_3d"], [f"{p}_logits_masked"])
    node("Softmax", [f"{p}_logits_masked"], [f"{p}_weights"], axis=-1)
    node("MatMul", [f"{p}_weights", f"{p}_V"], [f"{p}_ctx"])

    # Fuse heads into feature dim and apply the (d, d) output projection
    # in one shot.  Equivalent to the old MatMul+ReduceSum but avoids
    # the (n_heads, t, d) transient.
    node("Transpose", [f"{p}_ctx"], [f"{p}_ctx_t"], perm=[1, 0, 2])
    node("Reshape", [f"{p}_ctx_t", "_ctx_flat_shape"], [f"{p}_ctx_flat"])
    node("MatMul", [f"{p}_ctx_flat", f"{p}_WO"], [f"{p}_attn_sum"])
    node("Add", [current_res, f"{p}_attn_sum"], [f"{p}_res_attn"])

    # MLP sublayer
    node("MatMul", [f"{p}_res_attn", f"{p}_W1"], [f"{p}_l1_m"])
    node("Add", [f"{p}_l1_m", f"{p}_b1"], [f"{p}_l1_b"])
    node("Relu", [f"{p}_l1_b"], [f"{p}_l1_r"])
    node("MatMul", [f"{p}_l1_r", f"{p}_W2"], [f"{p}_l2_m"])
    node("Add", [f"{p}_l2_m", f"{p}_b2"], [f"{p}_l2_b"])
    node("Add", [f"{p}_res_attn", f"{p}_l2_b"], [f"{p}_res_next"])

    return f"{p}_res_next"


def compile_and_emit_onnx(
    output_node: Node,
    pos_encoding: Optional[PosEncoding],
    output_path: str,
    d: int,
    d_head: int,
    max_seq_len: int = 512,
    max_layers: int = 200,
    verbose: bool = True,
) -> None:
    """Compile a graph and emit ONNX in one streaming pass.

    After each layer is compiled, its dense weight tensors are extracted,
    sparsified on the fly (so ``SparseTensorProto`` is the only copy of
    mostly-zero matrices), and the compile layer's tensor attributes are
    nulled out.  Peak in-memory weight footprint therefore stays around
    one layer's dense size plus the cumulative sparse bytes — 5-10 GB for
    a 320x200 tex-64 game graph instead of the 50-100 GB that holding all
    layers dense would require.

    Writes two files:
        ``<output_path>``      — the ONNX model
        ``<stem>.meta.json``   — ordered input column names + format tag

    ``cached`` in the sidecar is ``false``: this path emits a plain
    feed-forward graph (no KV cache inputs/outputs).
    """
    dense_inits: list = []
    sparse_inits: list = []

    def on_layer_compiled(i: int, layer) -> None:
        attn = layer.attn.attn
        mlp = layer.mlp

        def emit(name: str, arr: np.ndarray) -> None:
            dense_tp, sparse_tp = _tensor_to_proto(name, arr)
            _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)

        # Extract Q/K/V/O one at a time and free each torch tensor as soon
        # as its bytes are inside a TensorProto/SparseTensorProto.
        emit(
            f"l{i}_WQ",
            attn.query_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy(),
        )
        attn.query_matrix = None
        emit(
            f"l{i}_WK",
            attn.key_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy(),
        )
        attn.key_matrix = None
        emit(
            f"l{i}_WV",
            attn.value_matrix.permute(1, 0, 2).reshape(d, d).contiguous().cpu().numpy(),
        )
        attn.value_matrix = None
        # Fuse (n_heads, d_head, d) → (d, d).  Same bytes, different
        # shape metadata: the resulting initializer matches the canonical
        # "Attention Is All You Need" W_O layout and feeds a single
        # (t, d) @ (d, d) MatMul at inference time.
        emit(
            f"l{i}_WO",
            attn.output_matrix.reshape(d, d).contiguous().cpu().numpy(),
        )
        attn.output_matrix = None

        emit(f"l{i}_W1", mlp.linear1.output_matrix.cpu().numpy())
        mlp.linear1.output_matrix = None
        emit(f"l{i}_b1", mlp.linear1.output_bias.cpu().numpy())
        mlp.linear1.output_bias = None
        emit(f"l{i}_W2", mlp.linear2.output_matrix.cpu().numpy())
        mlp.linear2.output_matrix = None
        emit(f"l{i}_b2", mlp.linear2.output_bias.cpu().numpy())
        mlp.linear2.output_bias = None

    # --- Phase 1: streaming compile ------------------------------------
    t0 = time.perf_counter()
    compiled = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=None,
        on_layer_compiled=on_layer_compiled,
    )
    t_compile = time.perf_counter() - t0

    # --- Phase 2: preamble / postamble metadata + graph assembly -------
    assert compiled.residual_assignment is not None
    n_heads = d // d_head
    n_layers = len(compiled.layers)

    t0 = time.perf_counter()
    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].mlp.out_state

    input_nodes_list: List[tuple] = []
    pos_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, InputNode):
            input_nodes_list.append((node.name, indices))
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for k, idx in enumerate(indices):
                constant_values[idx] = float(node.value[k])
        elif isinstance(node, (Concatenate, Embedding)):
            pass

    assert len(input_nodes_list) > 0, "No InputNode found in residual assignment"
    assert pos_indices is not None, "No PosEncoding node found in residual assignment"

    input_nodes_list.sort(key=lambda x: x[0])
    input_names = [name for name, _ in input_nodes_list]

    all_input_indices: List[int] = []
    for _, idx in input_nodes_list:
        all_input_indices.extend(idx)
    d_input = len(all_input_indices)

    input_proj = np.zeros((d_input, d), dtype=np.float32)
    for k, idx in enumerate(all_input_indices):
        input_proj[k, idx] = 1.0

    d_pos = len(pos_indices)
    pos_proj = np.zeros((d_pos, d), dtype=np.float32)
    for k, idx in enumerate(pos_indices):
        pos_proj[k, idx] = 1.0

    pos_encoding_buf = (
        _compute_pos_encoding(d_pos, max_seq_len)
        .numpy()
        .astype(np.float32, copy=False)
    )
    causal_mask = np.triu(
        np.full((max_seq_len, max_seq_len), -1000.0, dtype=np.float32), k=1
    )

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)
    d_output = len(output_gather_indices)

    # Preamble initializers — float ones run through _tensor_to_proto so
    # the very sparse input_proj/pos_proj/constant_values get COO-encoded
    # at build time.
    def add_float(name: str, arr: np.ndarray) -> None:
        dense_tp, sparse_tp = _tensor_to_proto(name, arr)
        _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)

    def add_int64(name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
        dense_inits.append(
            helper.make_tensor(
                name=name,
                data_type=TensorProto.INT64,
                dims=list(arr.shape),
                vals=arr.tobytes(),
                raw=True,
            )
        )

    add_float("input_proj", input_proj)
    add_float("pos_proj", pos_proj)
    add_float("constant_values", constant_values)
    add_float("pos_encoding_full", pos_encoding_buf)
    add_float("causal_mask_full", causal_mask)
    add_int64("output_gather_indices_init", output_gather_indices)
    add_int64("_zero_1d", np.array([0], dtype=np.int64))
    add_int64("_zeros_2d", np.array([0, 0], dtype=np.int64))
    add_int64("_axes_01", np.array([0, 1], dtype=np.int64))
    add_int64("_qkv_view_shape", np.array([0, n_heads, d_head], dtype=np.int64))
    # Shape constant for the fused-attention output projection:
    # Reshape (t, n_heads, d_head) → (t, d).
    add_int64("_ctx_flat_shape", np.array([0, d], dtype=np.int64))

    nodes: list = []

    def add_node(op: str, ins: List[str], outs: List[str], **attrs) -> None:
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Preamble
    add_node("Shape", ["inputs"], ["input_shape"])
    add_node("Gather", ["input_shape", "_zero_1d"], ["seq_len_1d"], axis=0)
    add_node(
        "Slice",
        ["pos_encoding_full", "_zero_1d", "seq_len_1d", "_zero_1d"],
        ["pos"],
    )
    add_node("MatMul", ["inputs", "input_proj"], ["inp_res"])
    add_node("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add_node("Add", ["inp_res", "pos_res"], ["res_pi"])
    add_node("Add", ["res_pi", "constant_values"], ["res_0"])
    add_node("Concat", ["seq_len_1d", "seq_len_1d"], ["mask_ends"], axis=0)
    add_node(
        "Slice",
        ["causal_mask_full", "_zeros_2d", "mask_ends", "_axes_01"],
        ["mask_2d"],
    )
    add_node("Unsqueeze", ["mask_2d", "_zero_1d"], ["mask_3d"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_layer_nodes(nodes, i, current_res, d, d_head, n_heads)

    add_node(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["outputs"],
        axis=1,
    )

    inputs_vi = helper.make_tensor_value_info(
        "inputs", TensorProto.FLOAT, ["seq_len", d_input]
    )
    outputs_vi = helper.make_tensor_value_info(
        "outputs", TensorProto.FLOAT, ["seq_len", d_output]
    )
    graph = helper.make_graph(
        nodes,
        "headless_transformer",
        inputs=[inputs_vi],
        outputs=[outputs_vi],
        initializer=dense_inits,
        sparse_initializer=sparse_inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14)],
        producer_name="torchwright",
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    onnx.save_model(model, output_path)
    t_save = time.perf_counter() - t0

    sidecar_path = _write_headless_meta(output_path, input_names, cached=False)

    if verbose:
        print(
            f"Phases: compile+emit {t_compile:.2f}s, "
            f"build {t_build:.2f}s, save {t_save:.2f}s"
        )
        print(
            f"{n_layers} layers, "
            f"{len(sparse_inits)} sparse inits, {len(dense_inits)} dense inits"
        )
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {sidecar_path}")


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
    """Compile a graph with raw float I/O to a KV-cached ONNX model.

    Produces two files:
        ``<output_path>``      — the ONNX model (KV-cache prefill/decode protocol)
        ``<stem>.meta.json``   — ordered input column names + format tag

    The sidecar records ``cached: true``, distinguishing this output from
    the feed-forward graphs produced by :func:`emit_headless_onnx` and
    :func:`compile_and_emit_onnx`.

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

    meta_path = _write_headless_meta(
        output_path, module.input_names, cached=True
    )

    if verbose:
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {meta_path}")
