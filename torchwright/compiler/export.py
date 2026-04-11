"""Compile a torchwright graph to a KV-cached ONNX model.

Two exporters, symmetric:

    compile_to_onnx(output_node, pos_encoding, embedding, path, ...)
        Token I/O: token_ids -> logits.  Sidecar format
        ``torchwright.token.v1`` carries the vocab.  Consumer:
        :mod:`torchwright.compiler.repl`.

    compile_headless_to_onnx(output_node, pos_encoding, path, ...)
        Float I/O: inputs -> outputs.  Sidecar format
        ``torchwright.headless.v1`` carries the alphabetically-ordered
        input column names.  Consumer:
        :mod:`torchwright.compiler.onnx_load`.

Both speak the KV-cache prefill/decode protocol:

    graph inputs:  <seq-input>, past_len, past_K_i, past_V_i  (i in 0..n_layers-1)
    graph outputs: <seq-output>, new_K_i, new_V_i

Prefill uses empty past tensors (shape ``(n_heads, 0, d_head)``) and
``past_len = 0``.  Decode uses past tensors produced by a prior run.

Both exporters stream each layer's weights into ONNX initializers (with
per-tensor sparsification) as the layer is compiled, then null out the
torch tensor references.  Peak in-memory weight footprint stays around
one dense layer's worth regardless of model depth — the path that lets
big graphs (e.g. the DOOM renderer) fit in realistic RAM.
"""

import json
import os
import time
from typing import Callable, List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import Concatenate, Embedding, LiteralValue, Node, PosEncoding
from torchwright.graph.misc import InputNode


HEADLESS_META_FORMAT = "torchwright.headless.v1"
TOKEN_META_FORMAT = "torchwright.token.v1"


# ---------------------------------------------------------------------------
# Sidecar plumbing
# ---------------------------------------------------------------------------


def meta_path_for(onnx_path: str) -> str:
    base, _ = os.path.splitext(onnx_path)
    return base + ".meta.json"


def _write_meta(onnx_path: str, meta: dict) -> str:
    meta_path = meta_path_for(onnx_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta_path


def _write_headless_meta(
    onnx_path: str,
    input_names: List[str],
    extra: Optional[dict] = None,
) -> str:
    """Write the headless sidecar JSON, optionally with an ``extra`` dict.

    ``extra`` is a free-form dict for per-export metadata (e.g. DOOM's
    ``rows_per_patch`` — surfaced to the host via
    :class:`OnnxHeadlessModule.metadata`). Kept totally general so the
    compiler layer has no project-specific keys.
    """
    meta: dict = {
        "format": HEADLESS_META_FORMAT,
        "input_names": list(input_names),
    }
    if extra:
        meta["extra"] = dict(extra)
    return _write_meta(onnx_path, meta)


# ---------------------------------------------------------------------------
# Positional encoding buffer (numpy; no torch dependency at runtime)
# ---------------------------------------------------------------------------


def _compute_pos_encoding(d_pos: int, max_seq_len: int) -> np.ndarray:
    """Precomputed pos encoding buffer for the ONNX ``pos_encoding_full``
    initializer.  Delegates to :meth:`PosEncoding.get_pos_encoding` so
    the ONNX graph and ``HeadlessTransformer.compute`` share one source
    of truth — any drift would silently break reference parity.
    """
    return (
        PosEncoding(d_pos)
        .get_pos_encoding(max_seq_len)
        .numpy()
        .astype(np.float32, copy=False)
    )


# ---------------------------------------------------------------------------
# Sparse-or-dense initializer conversion
# ---------------------------------------------------------------------------


_SPARSITY_THRESHOLD = 0.75
_MIN_SPARSE_ELEMENTS = 1024


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


def _add_float_init(
    name: str, arr: np.ndarray, dense_inits: list, sparse_inits: list
) -> None:
    dense_tp, sparse_tp = _tensor_to_proto(name, arr)
    _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)


def _add_int64_init(name: str, arr: np.ndarray, dense_inits: list) -> None:
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


def _add_scalar_inits(dense_inits: list) -> None:
    """Register 0-D scalar and tiny 1-D helper initializers used by the
    cached preamble (Range, Where, Unsqueeze/Slice axes).
    """
    dense_inits.append(
        helper.make_tensor(
            name="_i64_zero_s",
            data_type=TensorProto.INT64,
            dims=[],
            vals=np.array(0, dtype=np.int64).tobytes(),
            raw=True,
        )
    )
    dense_inits.append(
        helper.make_tensor(
            name="_i64_one_s",
            data_type=TensorProto.INT64,
            dims=[],
            vals=np.array(1, dtype=np.int64).tobytes(),
            raw=True,
        )
    )
    dense_inits.append(
        helper.make_tensor(
            name="_f32_neg1000_s",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=np.array(-1000.0, dtype=np.float32).tobytes(),
            raw=True,
        )
    )
    _add_int64_init("_axes0_1d", np.array([0], dtype=np.int64), dense_inits)
    _add_int64_init("_axes1_1d", np.array([1], dtype=np.int64), dense_inits)


# ---------------------------------------------------------------------------
# Streaming weight emission callback (shared by both exporters)
# ---------------------------------------------------------------------------


def _make_stream_layer_weights_cb(
    d: int,
    dense_inits: list,
    sparse_inits: list,
) -> Callable[[int, object], None]:
    """Factory for the forward_compile on_layer_compiled callback.

    Emits each freshly-compiled layer's dense weights into the
    dense/sparse init lists (sparsified on the fly when mostly zero) and
    nulls out the layer's tensor attributes so GC can reclaim them.
    """

    def on_layer_compiled(i: int, layer) -> None:
        attn = layer.attn.attn
        mlp = layer.mlp

        def emit(name: str, arr: np.ndarray) -> None:
            dense_tp, sparse_tp = _tensor_to_proto(name, arr)
            _append_proto(dense_tp, sparse_tp, dense_inits, sparse_inits)

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
        # (n_heads, d_head, d) → (d, d): canonical W_O layout that feeds
        # one (t, d) @ (d, d) MatMul at inference time.
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

    return on_layer_compiled


# ---------------------------------------------------------------------------
# Cached preamble and per-layer node emission
# ---------------------------------------------------------------------------


def _emit_cached_preamble(nodes: list, seq_input_name: str) -> None:
    """Emit nodes that produce ``pos`` and ``mask_bool_3d`` tensors.

    Requires:
      - graph input ``past_len``: scalar int64
      - graph input ``seq_input_name``: first dim is ``n_new``
      - initializer ``pos_encoding_full``: (max_seq_len, d_pos)
      - scalar initializers from :func:`_add_scalar_inits`

    Produces:
      - ``pos``: (n_new, d_pos) float, sliced from pos_encoding_full
      - ``mask_bool_3d``: (1, n_new, n_total) bool, True where a query
        position must NOT attend to a key position.  Applied via
        ``Where(mask_bool_3d, -1000, logits)`` — this overwrites masked
        logits (unlike an additive -1000 mask, which is numerically
        unsafe when the original logits can dominate the additive shift).
    """

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # n_new as 0-D scalar: Shape(seq_input)[0]
    add("Shape", [seq_input_name], ["_seq_shape"])
    add("Gather", ["_seq_shape", "_i64_zero_s"], ["_n_new_s"], axis=0)
    # n_total = past_len + n_new  (both 0-D scalars)
    add("Add", ["past_len", "_n_new_s"], ["_n_total_s"])

    # Dynamic causal mask — True where a new row must NOT attend to a column.
    # Row r corresponds to absolute position past_len + r; column c is
    # absolute position c.  mask[r, c] = c > past_len + r.
    add("Range", ["_i64_zero_s", "_n_new_s", "_i64_one_s"], ["_rows"])
    add("Range", ["_i64_zero_s", "_n_total_s", "_i64_one_s"], ["_cols"])
    add("Add", ["_rows", "past_len"], ["_abs_rows"])  # (n_new,)
    add("Unsqueeze", ["_abs_rows", "_axes1_1d"], ["_abs_rows_col"])  # (n_new, 1)
    add("Unsqueeze", ["_cols", "_axes0_1d"], ["_cols_row"])  # (1, n_total)
    add("Greater", ["_cols_row", "_abs_rows_col"], ["_mask_bool"])  # (n_new, n_total)
    add("Unsqueeze", ["_mask_bool", "_axes0_1d"], ["mask_bool_3d"])  # (1, n_new, n_total)

    # Positional encoding slice:
    # pos_encoding_full[past_len : past_len + n_new]
    add("Unsqueeze", ["past_len", "_axes0_1d"], ["_past_len_1d"])
    add("Unsqueeze", ["_n_total_s", "_axes0_1d"], ["_n_total_1d"])
    add(
        "Slice",
        ["pos_encoding_full", "_past_len_1d", "_n_total_1d", "_axes0_1d"],
        ["pos"],
    )


def _emit_cached_layer_nodes(
    nodes: list,
    layer_idx: int,
    current_res: str,
    d: int,
    d_head: int,
    n_heads: int,
) -> str:
    """Emit cached attention + FFN nodes for one layer.

    Reads graph inputs ``past_K_{i}`` / ``past_V_{i}`` and writes graph
    outputs ``new_K_{i}`` / ``new_V_{i}``.  Uses the shared ``mask_3d``
    produced by :func:`_emit_cached_preamble`.

    Returns the name of the next residual stream tensor.
    """
    p = f"l{layer_idx}"

    def node(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Project Q, K_new, V_new from the new rows only, reshape to heads.
    node("MatMul", [current_res, f"{p}_WQ"], [f"{p}_Q_flat"])
    node("Reshape", [f"{p}_Q_flat", "_qkv_view_shape"], [f"{p}_Q_view"])
    node("Transpose", [f"{p}_Q_view"], [f"{p}_Q"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WK"], [f"{p}_K_flat"])
    node("Reshape", [f"{p}_K_flat", "_qkv_view_shape"], [f"{p}_K_view"])
    node("Transpose", [f"{p}_K_view"], [f"{p}_K_new"], perm=[1, 0, 2])

    node("MatMul", [current_res, f"{p}_WV"], [f"{p}_V_flat"])
    node("Reshape", [f"{p}_V_flat", "_qkv_view_shape"], [f"{p}_V_view"])
    node("Transpose", [f"{p}_V_view"], [f"{p}_V_new"], perm=[1, 0, 2])

    # Concatenate the cached past with the new rows along the seq axis.
    # These Concat outputs are also exposed as graph outputs (new_K_i /
    # new_V_i) so callers can feed them back as the next step's past.
    node(
        "Concat",
        [f"past_K_{layer_idx}", f"{p}_K_new"],
        [f"new_K_{layer_idx}"],
        axis=1,
    )
    node(
        "Concat",
        [f"past_V_{layer_idx}", f"{p}_V_new"],
        [f"new_V_{layer_idx}"],
        axis=1,
    )

    # Attention over the full (past + new) K and V.
    node("Transpose", [f"new_K_{layer_idx}"], [f"{p}_K_T"], perm=[0, 2, 1])
    node("MatMul", [f"{p}_Q", f"{p}_K_T"], [f"{p}_logits"])
    # Overwrite-mask with -1000 (equivalent to torch's masked_fill).  An
    # additive -1000 would leave masked positions at "logit - 1000",
    # which is not dominated by real logits when they reach ~800.
    node(
        "Where",
        ["mask_bool_3d", "_f32_neg1000_s", f"{p}_logits"],
        [f"{p}_logits_masked"],
    )
    node("Softmax", [f"{p}_logits_masked"], [f"{p}_weights"], axis=-1)
    node("MatMul", [f"{p}_weights", f"new_V_{layer_idx}"], [f"{p}_ctx"])

    # Fused output projection: (n_heads, t, d_head) → (t, d) → (t, d)
    node("Transpose", [f"{p}_ctx"], [f"{p}_ctx_t"], perm=[1, 0, 2])
    node("Reshape", [f"{p}_ctx_t", "_ctx_flat_shape"], [f"{p}_ctx_flat"])
    node("MatMul", [f"{p}_ctx_flat", f"{p}_WO"], [f"{p}_attn_sum"])
    node("Add", [current_res, f"{p}_attn_sum"], [f"{p}_res_attn"])

    # FFN + skip
    node("MatMul", [f"{p}_res_attn", f"{p}_W1"], [f"{p}_l1_m"])
    node("Add", [f"{p}_l1_m", f"{p}_b1"], [f"{p}_l1_b"])
    node("Relu", [f"{p}_l1_b"], [f"{p}_l1_r"])
    node("MatMul", [f"{p}_l1_r", f"{p}_W2"], [f"{p}_l2_m"])
    node("Add", [f"{p}_l2_m", f"{p}_b2"], [f"{p}_l2_b"])
    node("Add", [f"{p}_res_attn", f"{p}_l2_b"], [f"{p}_res_next"])

    return f"{p}_res_next"


def _kv_io_value_info(
    n_layers: int, n_heads: int, d_head: int
) -> tuple[list, list]:
    """Build ValueInfoProto entries for the KV-cache inputs and outputs.

    Returns:
        (past_vis, new_vis) — each a list of length 2*n_layers
        alternating ``past_K_i`` / ``past_V_i`` (inputs) and
        ``new_K_i`` / ``new_V_i`` (outputs).
    """
    past_vis: list = []
    new_vis: list = []
    for i in range(n_layers):
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_K_{i}", TensorProto.FLOAT, [n_heads, "n_past", d_head]
            )
        )
        past_vis.append(
            helper.make_tensor_value_info(
                f"past_V_{i}", TensorProto.FLOAT, [n_heads, "n_past", d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"new_K_{i}", TensorProto.FLOAT, [n_heads, "n_total", d_head]
            )
        )
        new_vis.append(
            helper.make_tensor_value_info(
                f"new_V_{i}", TensorProto.FLOAT, [n_heads, "n_total", d_head]
            )
        )
    return past_vis, new_vis


# ---------------------------------------------------------------------------
# Public exporters
# ---------------------------------------------------------------------------


def compile_headless_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 200,
    verbose: bool = True,
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
) -> None:
    """Compile a float-I/O graph to a KV-cached ONNX model.

    Writes two files:
        ``<output_path>``    — the ONNX model
        ``<stem>.meta.json`` — ``{"format": "torchwright.headless.v1",
                                  "input_names": [...]}``

    The graph speaks the KV-cache prefill/decode protocol:
        inputs:  inputs (n_new, d_input), past_len (scalar),
                 past_K_i, past_V_i (n_heads, n_past, d_head)
        outputs: outputs (n_new, d_output),
                 new_K_i, new_V_i (n_heads, n_total, d_head)

    Prefill = empty past tensors + past_len=0.  Decode = feed back the
    new_K_i / new_V_i from a previous run and set past_len accordingly.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.
    """
    dense_inits: list = []
    sparse_inits: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(d, dense_inits, sparse_inits)

    # --- Phase 1: streaming compile ---------------------------------------
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
        d_hidden=d_hidden,
    )
    t_compile = time.perf_counter() - t0

    # --- Phase 2: metadata + graph assembly -------------------------------
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

    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)
    d_output = len(output_gather_indices)

    # Initializers
    _add_float_init("input_proj", input_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_int64_init(
        "output_gather_indices_init", output_gather_indices, dense_inits
    )
    _add_int64_init(
        "_qkv_view_shape",
        np.array([0, n_heads, d_head], dtype=np.int64),
        dense_inits,
    )
    _add_int64_init(
        "_ctx_flat_shape", np.array([0, d], dtype=np.int64), dense_inits
    )
    _add_scalar_inits(dense_inits)

    # Nodes: preamble (mask + pos), residual stream, layers, postamble.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, seq_input_name="inputs")
    add("MatMul", ["inputs", "input_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes, i, current_res, d, d_head, n_heads
        )

    add(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["outputs"],
        axis=1,
    )

    # Graph I/O value infos
    inputs_vi = helper.make_tensor_value_info(
        "inputs", TensorProto.FLOAT, ["n_new", d_input]
    )
    past_len_vi = helper.make_tensor_value_info("past_len", TensorProto.INT64, [])
    past_vis, new_vis = _kv_io_value_info(n_layers, n_heads, d_head)
    outputs_vi = helper.make_tensor_value_info(
        "outputs", TensorProto.FLOAT, ["n_new", d_output]
    )

    graph = helper.make_graph(
        nodes,
        "headless_transformer_cached",
        inputs=[inputs_vi, past_len_vi, *past_vis],
        outputs=[outputs_vi, *new_vis],
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

    meta_path = _write_headless_meta(
        output_path,
        list(input_names),
        extra=extra_metadata,
    )

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
        print(f"Wrote {meta_path}")


def compile_to_onnx(
    output_node: Node,
    pos_encoding: PosEncoding,
    embedding: Embedding,
    output_path: str,
    d: int = 1024,
    d_head: int = 16,
    max_seq_len: int = 512,
    max_layers: int = 200,
    verbose: bool = True,
) -> None:
    """Compile a token-I/O graph to a KV-cached ONNX model.

    Writes two files:
        ``<output_path>``    — the ONNX model
        ``<stem>.meta.json`` — ``{"format": "torchwright.token.v1",
                                  "vocab": [...]}``

    The graph speaks the KV-cache prefill/decode protocol:
        inputs:  token_ids (n_new,) int64, past_len (scalar),
                 past_K_i, past_V_i (n_heads, n_past, d_head)
        outputs: logits (n_new, vocab_size),
                 new_K_i, new_V_i (n_heads, n_total, d_head)

    Prefill = empty past + past_len=0; decode = feed back the new_K_i /
    new_V_i from a prior run with the accumulated past_len.
    """
    dense_inits: list = []
    sparse_inits: list = []

    on_layer_compiled = _make_stream_layer_weights_cb(d, dense_inits, sparse_inits)

    # --- Phase 1: streaming compile ---------------------------------------
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

    # --- Phase 2: metadata + graph assembly -------------------------------
    assert compiled.residual_assignment is not None
    n_heads = d // d_head
    n_layers = len(compiled.layers)

    t0 = time.perf_counter()
    in_state = compiled.layers[0].attn.in_state
    out_state = compiled.layers[-1].mlp.out_state

    embedding_indices: Optional[List[int]] = None
    pos_indices: Optional[List[int]] = None
    constant_values = np.zeros(d, dtype=np.float32)

    for node in compiled.residual_assignment.get_nodes(in_state):
        indices = compiled.residual_assignment.get_node_indices(in_state, node)
        if isinstance(node, Embedding):
            embedding_indices = indices
        elif isinstance(node, PosEncoding):
            pos_indices = indices
        elif isinstance(node, LiteralValue):
            for k, idx in enumerate(indices):
                constant_values[idx] = float(node.value[k])
        elif isinstance(node, Concatenate):
            pass

    assert embedding_indices is not None, "No Embedding node in residual assignment"
    assert pos_indices is not None, "No PosEncoding node in residual assignment"

    d_embed = len(embedding_indices)
    d_pos = len(pos_indices)

    embedding_proj = np.zeros((d_embed, d), dtype=np.float32)
    for k, idx in enumerate(embedding_indices):
        embedding_proj[k, idx] = 1.0

    pos_proj = np.zeros((d_pos, d), dtype=np.float32)
    for k, idx in enumerate(pos_indices):
        pos_proj[k, idx] = 1.0

    pos_encoding_buf = _compute_pos_encoding(d_pos, max_seq_len)

    embed_table_np = (
        embedding.table.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    vocab_size, d_embed_check = embed_table_np.shape
    assert d_embed_check == d_embed, (
        f"Embedding table last dim {d_embed_check} disagrees with "
        f"d_embed {d_embed} derived from feature assignment"
    )

    output_indices = compiled.residual_assignment.get_node_indices(
        out_state, output_node
    )
    output_gather_indices = np.asarray(output_indices, dtype=np.int64)

    # Initializers
    _add_float_init("embedding_proj", embedding_proj, dense_inits, sparse_inits)
    _add_float_init("pos_proj", pos_proj, dense_inits, sparse_inits)
    _add_float_init("constant_values", constant_values, dense_inits, sparse_inits)
    _add_float_init("pos_encoding_full", pos_encoding_buf, dense_inits, sparse_inits)
    _add_float_init("embed_table", embed_table_np, dense_inits, sparse_inits)
    _add_int64_init(
        "output_gather_indices_init", output_gather_indices, dense_inits
    )
    _add_int64_init(
        "_qkv_view_shape",
        np.array([0, n_heads, d_head], dtype=np.int64),
        dense_inits,
    )
    _add_int64_init(
        "_ctx_flat_shape", np.array([0, d], dtype=np.int64), dense_inits
    )
    _add_scalar_inits(dense_inits)

    # Nodes: preamble (mask + pos), token embed, residual stream, layers,
    # output gather, unembed.
    nodes: list = []

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    _emit_cached_preamble(nodes, seq_input_name="token_ids")
    # Token embedding lookup: (vocab, d_embed) gather rows by token_ids.
    add("Gather", ["embed_table", "token_ids"], ["_token_emb"], axis=0)
    add("MatMul", ["_token_emb", "embedding_proj"], ["inp_res"])
    add("MatMul", ["pos", "pos_proj"], ["pos_res"])
    add("Add", ["inp_res", "pos_res"], ["res_pi"])
    add("Add", ["res_pi", "constant_values"], ["res_0"])

    current_res = "res_0"
    for i in range(n_layers):
        current_res = _emit_cached_layer_nodes(
            nodes, i, current_res, d, d_head, n_heads
        )

    add(
        "Gather",
        [current_res, "output_gather_indices_init"],
        ["_output_emb"],
        axis=1,
    )
    # logits = output_emb @ embed_table.T
    add("Transpose", ["embed_table"], ["_embed_table_T"], perm=[1, 0])
    add("MatMul", ["_output_emb", "_embed_table_T"], ["logits"])

    # Graph I/O value infos
    token_ids_vi = helper.make_tensor_value_info(
        "token_ids", TensorProto.INT64, ["n_new"]
    )
    past_len_vi = helper.make_tensor_value_info("past_len", TensorProto.INT64, [])
    past_vis, new_vis = _kv_io_value_info(n_layers, n_heads, d_head)
    logits_vi = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, ["n_new", vocab_size]
    )

    graph = helper.make_graph(
        nodes,
        "token_transformer_cached",
        inputs=[token_ids_vi, past_len_vi, *past_vis],
        outputs=[logits_vi, *new_vis],
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

    meta_path = _write_meta(
        output_path,
        {"format": TOKEN_META_FORMAT, "vocab": list(embedding.tokenizer.vocab)},
    )

    if verbose:
        print(
            f"Phases: compile+emit {t_compile:.2f}s, "
            f"build {t_build:.2f}s, save {t_save:.2f}s"
        )
        print(
            f"{n_layers} layers, {vocab_size} vocab, "
            f"{len(sparse_inits)} sparse inits, {len(dense_inits)} dense inits"
        )
        model_size = os.path.getsize(output_path)
        print(f"Wrote {output_path} ({model_size:,} bytes)")
        print(f"Wrote {meta_path}")


# ---------------------------------------------------------------------------
# In-process headless callable: a thin adapter over HeadlessTransformer.compute()
# that presents the same (inputs) -> outputs interface as OnnxHeadlessModule.
#
# Used for dev-mode DOOM play and compiler-output tests that don't need an
# ONNX round-trip.  For production inference, export with
# compile_headless_to_onnx instead — that path streams weights and runs
# under onnxruntime.
# ---------------------------------------------------------------------------


class CompiledHeadless:
    """Callable wrapper around :class:`HeadlessTransformer`.

    Exposes the same three-method surface as
    :class:`torchwright.compiler.onnx_load.OnnxHeadlessModule`:

    - ``module(inputs)``: stateless per-query inference — runs the
      non-cached ``forward()`` path and returns outputs.
    - ``module.step(inputs, past)``: autoregressive step — runs
      ``forward_cached()`` with the given past and returns
      ``(outputs, new_past)``.
    - ``module.empty_past()``: zero-length KV cache tuple suitable as
      the initial state for a decode sequence.
    """

    def __init__(
        self,
        net,
        input_specs: List[tuple],
        output_indices: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> None:
        self._net = net
        # input_specs: list of (name, start_col, width) in input-tensor column order.
        self._input_specs = list(input_specs)
        self._output_indices = output_indices
        self.input_names: List[str] = [name for name, _, _ in input_specs]
        self.metadata: dict = dict(metadata or {})

        # KV cache shape metadata — discovered from the compiled transformer
        # so empty_past() can build zero-length tensors of the right shape.
        first_attn = net.layers[0].attn.attn
        self._n_heads = first_attn.n_heads
        self._d_head = first_attn.d_head
        self._n_layers = len(net.layers)

    def _build_res_stream(
        self, inputs: torch.Tensor, past_len: int
    ) -> torch.Tensor:
        n_new = inputs.shape[0]
        input_values = {
            name: inputs[:, start : start + width]
            for name, start, width in self._input_specs
        }
        return self._net.get_input_res_stream(
            n_new, input_values, past_len=past_len
        ).to(self._net.device)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Stateless per-query inference — uses the non-cached ``forward()``."""
        res_stream = self._build_res_stream(inputs, past_len=0)
        res = self._net.forward(res_stream)
        return res[:, self._output_indices]

    def empty_past(
        self,
    ) -> tuple:
        """Zero-length past tensors suitable for a first prefill call."""
        zeros = torch.zeros(self._n_heads, 0, self._d_head)
        past_K = tuple(zeros.clone() for _ in range(self._n_layers))
        past_V = tuple(zeros.clone() for _ in range(self._n_layers))
        return (past_K, past_V)

    def step(
        self, inputs: torch.Tensor, past: tuple
    ) -> tuple:
        """Cached forward step.

        Args:
            inputs: ``(n_new, d_input)`` float tensor for the new rows.
            past: ``(past_K_tuple, past_V_tuple)`` from a prior step or
                :meth:`empty_past`.  Each tuple has length ``n_layers``
                and each entry is ``(n_heads, n_past, d_head)``.

        Returns:
            ``(outputs, new_past)`` where ``outputs`` is
            ``(n_new, d_output)`` and ``new_past`` has the same shape
            as ``past`` but with the new rows appended.
        """
        past_K, past_V = past
        assert len(past_K) == self._n_layers
        assert len(past_V) == self._n_layers
        past_len = int(past_K[0].shape[1])

        res_stream = self._build_res_stream(inputs, past_len=past_len)
        past_kvs = [(past_K[i], past_V[i]) for i in range(self._n_layers)]
        res, new_kvs = self._net.forward_cached(res_stream, past_kvs=past_kvs)

        new_K = tuple(kv[0] for kv in new_kvs)
        new_V = tuple(kv[1] for kv in new_kvs)
        outputs = res[:, self._output_indices]
        return outputs, (new_K, new_V)

    def eval(self) -> "CompiledHeadless":
        return self


def compile_headless(
    output_node: Node,
    pos_encoding: PosEncoding,
    d: int = 1024,
    d_head: int = 16,
    max_layers: int = 100,
    verbose: bool = True,
    device: str = "cpu",
    extra_metadata: Optional[dict] = None,
    d_hidden: Optional[int] = None,
) -> CompiledHeadless:
    """Compile a headless graph to an in-process callable.

    Returns a :class:`CompiledHeadless` that evaluates the graph via
    :meth:`HeadlessTransformer.forward` behind the standard
    ``module(inputs) -> outputs`` interface.  For production use (saved
    artifact, autoregressive decode, fast startup), use
    :func:`compile_headless_to_onnx` instead.

    ``d_hidden`` is the per-layer MLP hidden width.  Defaults to ``d``
    when omitted; pass an explicit value to decouple the MLP intermediate
    width from the residual stream width.
    """
    net = forward_compile(
        d=d,
        d_head=d_head,
        output_node=output_node,
        pos_encoding=pos_encoding,
        verbose=verbose,
        max_layers=max_layers,
        device=device,
        d_hidden=d_hidden,
    )

    assert net.residual_assignment is not None
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].mlp.out_state

    input_nodes_list: List[tuple] = []  # (name, width)
    for node in net.residual_assignment.get_nodes(in_state):
        if isinstance(node, InputNode):
            indices = net.residual_assignment.get_node_indices(in_state, node)
            input_nodes_list.append((node.name, len(indices)))
    input_nodes_list.sort(key=lambda x: x[0])

    input_specs: List[tuple] = []
    offset = 0
    for name, width in input_nodes_list:
        input_specs.append((name, offset, width))
        offset += width

    # Direct residual-stream gather handles Concatenate output nodes
    # (which compute()'s per-node result dict does not populate).
    output_indices = torch.tensor(
        net.residual_assignment.get_node_indices(out_state, output_node),
        dtype=torch.long,
    )

    return CompiledHeadless(
        net, input_specs, output_indices, metadata=extra_metadata,
    )
