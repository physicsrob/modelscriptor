"""Writes weight matrices into TransformerLayer components.

Each operation directly sets matrix entries using column indices from the
ResidualStreamMap. No strategies, no ResidualAssignment — just scatter writes.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set

import torch
import torch.nn.functional as F

from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.graph import Node, Linear, Attn, Add, Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding, attention_hardness
from torchwright.graph.relu import ReLU


@dataclass
class AttnHeadOp:
    op_type: Literal[
        "compute_attn", "compute_linear", "compute_add", "cancel", "add_into",
        "delta_transfer"
    ]
    node: Node
    target_cols: List[int]
    # For delta_transfer: source columns (where the output value is)
    source_cols: List[int] = None
    # For delta_transfer: subtract columns (same as target for overlay)
    subtract_cols: List[int] = None


@dataclass
class MLPOp:
    op_type: Literal[
        "compute_relu",
        "compute_standalone_relu",
        "compute_literal_value",
        "compute_bias",
    ]
    node: Node
    target_cols: List[int]
    mlp_slots: List[int] = field(default_factory=list)


def write_attn_sublayer(
    layer: TransformerLayer,
    ops: List[AttnHeadOp],
    residual_map: ResidualStreamMap,
    pos_encoding: Optional[PosEncoding],
):
    """Write attention head operations into a layer's AttnLayerComponent."""
    attn = layer.attn.attn
    assert pos_encoding is not None, "pos_encoding required for attention ops"
    for op in ops:
        if op.op_type == "compute_attn":
            _write_compute_attn(attn, op, residual_map)
        elif op.op_type == "compute_linear":
            _write_compute_linear(attn, op, residual_map, pos_encoding)
        elif op.op_type == "cancel":
            _write_cancel(attn, op, residual_map, pos_encoding)
        elif op.op_type == "compute_add":
            _write_compute_add(attn, op, residual_map, pos_encoding)
        elif op.op_type == "add_into":
            _write_add_into(attn, op, residual_map, pos_encoding)
        elif op.op_type == "delta_transfer":
            _write_delta_transfer(attn, op, residual_map, pos_encoding)
        else:
            raise ValueError(f"Unknown attn op_type: {op.op_type}")


def write_mlp_sublayer(
    layer: TransformerLayer,
    ops: List[MLPOp],
    residual_map: ResidualStreamMap,
    biased_linears: Optional[Set[Node]] = None,
):
    """Write MLP operations into a layer's MLPSubLayer components."""
    if biased_linears is None:
        biased_linears = set()
    for op in ops:
        if op.op_type == "compute_relu":
            _write_compute_relu(layer.mlp, op, residual_map, biased_linears)
        elif op.op_type == "compute_literal_value":
            _write_compute_literal_value(layer.mlp, op)
        elif op.op_type == "compute_bias":
            _write_compute_bias(layer.mlp, op)
        elif op.op_type == "compute_standalone_relu":
            _write_compute_standalone_relu(layer.mlp, op, residual_map, biased_linears)
        else:
            raise ValueError(f"Unknown mlp op_type: {op.op_type}")


# ---------------------------------------------------------------------------
# Attention operations
# ---------------------------------------------------------------------------


def _scatter_attn_head(
    attn, head, q_idx, k_idx, v_idx, o_idx, q_mat, k_mat, v_mat, o_mat, d_head
):
    """Scatter strategy matrices into one attention head's weight tensors."""
    q_idx_t = torch.as_tensor(q_idx, dtype=torch.long)
    k_idx_t = torch.as_tensor(k_idx, dtype=torch.long)
    v_idx_t = torch.as_tensor(v_idx, dtype=torch.long)
    o_idx_t = torch.as_tensor(o_idx, dtype=torch.long)
    # Source matrices may be Long (e.g. integer Linear weights); the
    # destination is always Float.  Original scalar-loop assignment
    # auto-cast; vectorized assignment does not.
    target_dtype = attn.query_matrix.dtype
    attn.query_matrix[head, q_idx_t, :d_head] = q_mat[: len(q_idx), :d_head].to(target_dtype)
    attn.key_matrix[head, k_idx_t, :d_head] = k_mat[: len(k_idx), :d_head].to(target_dtype)
    attn.value_matrix[head, v_idx_t, :d_head] = v_mat[: len(v_idx), :d_head].to(target_dtype)
    attn.output_matrix[head, :d_head, o_idx_t] = o_mat[:d_head, : len(o_idx)].to(target_dtype)


def _allocate_head(attn):
    """Allocate the next available attention head."""
    assert attn.used_heads < attn.n_heads, "Ran out of attention heads"
    head = attn.used_heads
    attn.used_heads += 1
    return head


def _write_compute_attn(attn, op: AttnHeadOp, rmap: ResidualStreamMap):
    """Copy an Attn node's Q/K/V/O matrices into attention heads.

    When the node's d_v exceeds the layer's d_head, V/O are split across
    multiple heads that share duplicated Q/K matrices.  This is correct
    because V/O is applied after softmax (purely linear):
    sum_i(weights @ V_i @ O_i) == weights @ V @ O.
    """
    node = op.node
    assert isinstance(node, Attn)

    query_in, key_in, value_in = node.inputs
    q_idx = rmap.resolve_indices(query_in)
    k_idx = rmap.resolve_indices(key_in)
    v_idx = rmap.resolve_indices(value_in)
    o_idx = op.target_cols

    layer_d_head = attn.d_head

    assert layer_d_head >= node.d_qk, (
        f"d_head={layer_d_head} is too small for Attn node "
        f"'{node.name}' which needs d_qk={node.d_qk} for Q/K. "
        f"Use d_head>={node.d_qk} or let compile_game auto-size it."
    )

    # Q/K are shared across all V/O chunk heads, padded to layer d_head
    q_mat = F.pad(node.query_matrix, (0, layer_d_head - node.d_qk))
    k_mat = F.pad(node.key_matrix, (0, layer_d_head - node.d_qk))

    # Split V/O across ceil(d_v / layer_d_head) heads
    n_vo_heads = (node.d_v + layer_d_head - 1) // layer_d_head
    for chunk_idx in range(n_vo_heads):
        v_start = chunk_idx * layer_d_head
        v_end = min(v_start + layer_d_head, node.d_v)
        chunk_size = v_end - v_start

        v_chunk = F.pad(node.value_matrix[:, v_start:v_end], (0, layer_d_head - chunk_size))
        o_chunk = F.pad(node.output_matrix[v_start:v_end, :], (0, 0, 0, layer_d_head - chunk_size))

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn, head, q_idx, k_idx, v_idx, o_idx,
            q_mat, k_mat, v_chunk, o_chunk, layer_d_head,
        )


def _current_pos_attn_matrices(pos_encoding, d_head):
    """Build Q/K matrices for current-position attention."""
    d_pos = len(pos_encoding)
    q_mat = attention_hardness * torch.eye(d_pos, d_head)
    k_mat = torch.eye(d_pos, d_head)
    return q_mat, k_mat


def _write_compute_linear(
    attn, op: AttnHeadOp, rmap: ResidualStreamMap, pos_encoding: PosEncoding
):
    """Compile a zero-bias Linear via current-position attention.

    Q/K attend to current position via pos_encoding.
    V reads from the Linear's input columns.
    O applies the Linear's weight matrix to target columns.

    For d_input > d_head, splits across multiple heads. Each head handles
    a d_head-sized chunk of the input and applies the corresponding slice
    of the weight matrix. Attention heads are additive, so results sum
    to give the full W @ input.
    """
    node = op.node
    assert isinstance(node, Linear)

    input_node = node.inputs[0]
    d_head = attn.d_head
    d_input = len(input_node)

    pe_idx = rmap.get_indices(pos_encoding)
    v_idx = rmap.resolve_indices(input_node)  # resolves through Concatenate
    o_idx = op.target_cols

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)

    # Split input across ceil(d_input / d_head) heads
    for start in range(0, d_input, d_head):
        end = min(start + d_head, d_input)
        chunk_size = end - start

        v_chunk_idx = v_idx[start:end]
        v_mat = torch.eye(chunk_size, d_head)

        # Weight matrix slice for this chunk: rows [start:end]
        weight_slice = node.output_matrix[start:end, :]  # (chunk_size, d_output)
        o_mat = F.pad(weight_slice, (0, 0, 0, d_head - chunk_size))

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn,
            head,
            pe_idx,
            pe_idx,
            v_chunk_idx,
            o_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
        )


def _write_compute_add(
    attn, op: AttnHeadOp, rmap: ResidualStreamMap, pos_encoding: PosEncoding
):
    """Compute Add(a, b) by copying both inputs to fresh columns via attention.

    Used when neither input is dead (so add_into can't reuse columns).
    Allocates new output columns and uses two groups of attention heads:
    one to copy a, one to copy b. Since attention heads are additive,
    the result in the output columns is 0 + a + b = a + b.
    """
    node = op.node
    assert isinstance(node, Add)

    a0, a1 = node.inputs
    d_head = attn.d_head
    d_output = len(node)

    pe_idx = rmap.get_indices(pos_encoding)
    o_idx = op.target_cols
    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)

    # Copy each input to the output columns via separate heads.
    for input_node in [a0, a1]:
        v_idx = rmap.resolve_indices(input_node)
        for start in range(0, d_output, d_head):
            end = min(start + d_head, d_output)
            chunk_size = end - start

            v_chunk_idx = v_idx[start:end]
            o_chunk_idx = o_idx[start:end]

            v_mat = torch.eye(chunk_size, d_head)
            o_mat = torch.eye(d_head, chunk_size)

            head = _allocate_head(attn)
            _scatter_attn_head(
                attn,
                head,
                pe_idx,
                pe_idx,
                v_chunk_idx,
                o_chunk_idx,
                q_mat,
                k_mat,
                v_mat,
                o_mat,
                d_head,
            )


def _write_cancel(
    attn, op: AttnHeadOp, rmap: ResidualStreamMap, pos_encoding: PosEncoding
):
    """Cancel a node: V=identity, O=-identity. Skip adds x + (-x) = 0.

    Splits across multiple heads for nodes wider than d_head.
    """
    node = op.node
    d_head = attn.d_head
    d_node = len(node)

    pe_idx = rmap.get_indices(pos_encoding)
    node_idx = op.target_cols

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)

    for start in range(0, d_node, d_head):
        end = min(start + d_head, d_node)
        chunk_size = end - start

        chunk_idx = node_idx[start:end]
        v_mat = torch.eye(chunk_size, d_head)
        o_mat = -torch.eye(d_head, chunk_size)

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn,
            head,
            pe_idx,
            pe_idx,
            chunk_idx,
            chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
        )


def _write_add_into(
    attn, op: AttnHeadOp, rmap: ResidualStreamMap, pos_encoding: PosEncoding
):
    """Add(dead, live): copy live's values to dead's columns via attention.

    target_cols are the dead addend's columns (now owned by the Add node
    after reassign). The live addend is whichever input is still allocated
    in the residual map. Skip connection adds: dead + live = Add(dead, live).

    Splits across multiple heads for operands wider than d_head.
    """
    node = op.node
    assert isinstance(node, Add)

    # Determine which input is live (still has values in the residual stream).
    # After scheduler's reassign, the dead addend is no longer individually
    # allocated. The live addend is either a regular allocated node or a
    # Concatenate (whose children are allocated).
    a0, a1 = node.inputs
    if rmap.is_allocated(a0) or isinstance(a0, Concatenate):
        live_addend = a0
    else:
        live_addend = a1
    d_head = attn.d_head
    d_live = len(live_addend)

    pe_idx = rmap.get_indices(pos_encoding)
    v_idx = rmap.resolve_indices(live_addend)  # resolves through Concatenate
    o_idx = op.target_cols  # dead addend's columns

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)

    for start in range(0, d_live, d_head):
        end = min(start + d_head, d_live)
        chunk_size = end - start

        v_chunk_idx = v_idx[start:end]
        o_chunk_idx = o_idx[start:end]

        v_mat = torch.eye(chunk_size, d_head)
        o_mat = torch.eye(d_head, chunk_size)

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn,
            head,
            pe_idx,
            pe_idx,
            v_chunk_idx,
            o_chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
        )


def _write_delta_transfer(
    attn, op: AttnHeadOp, rmap: ResidualStreamMap, pos_encoding: PosEncoding
):
    """Transfer (source - subtract) to target columns via attention.

    This operation computes: target_cols += (source_cols - subtract_cols)

    After the skip connection, if target_cols originally held the same value
    as subtract_cols (i.e., the input value), then:
        result = original + (source - subtract)
               = subtract + (source - subtract)
               = source

    This enables overlaid I/O where the output replaces the input in-place.

    Uses two groups of attention heads:
    - Group 1: Copy source_cols to target_cols with +1 coefficient
    - Group 2: Copy subtract_cols to target_cols with -1 coefficient

    Net effect: target_cols += source - subtract
    """
    assert op.source_cols is not None, "delta_transfer requires source_cols"
    assert op.subtract_cols is not None, "delta_transfer requires subtract_cols"

    d_head = attn.d_head
    d_width = len(op.target_cols)

    assert len(op.source_cols) == d_width
    assert len(op.subtract_cols) == d_width

    pe_idx = rmap.get_indices(pos_encoding)
    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)

    # Group 1: +1 coefficient (copy source to target)
    for start in range(0, d_width, d_head):
        end = min(start + d_head, d_width)
        chunk_size = end - start

        v_chunk_idx = op.source_cols[start:end]
        o_chunk_idx = op.target_cols[start:end]

        v_mat = torch.eye(chunk_size, d_head)
        o_mat = torch.eye(d_head, chunk_size)  # +1 coefficient

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn,
            head,
            pe_idx,
            pe_idx,
            v_chunk_idx,
            o_chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
        )

    # Group 2: -1 coefficient (subtract subtract_cols from target)
    for start in range(0, d_width, d_head):
        end = min(start + d_head, d_width)
        chunk_size = end - start

        v_chunk_idx = op.subtract_cols[start:end]
        o_chunk_idx = op.target_cols[start:end]

        v_mat = torch.eye(chunk_size, d_head)
        o_mat = -torch.eye(d_head, chunk_size)  # -1 coefficient

        head = _allocate_head(attn)
        _scatter_attn_head(
            attn,
            head,
            pe_idx,
            pe_idx,
            v_chunk_idx,
            o_chunk_idx,
            q_mat,
            k_mat,
            v_mat,
            o_mat,
            d_head,
        )


# ---------------------------------------------------------------------------
# MLP operations
# ---------------------------------------------------------------------------


def _write_compute_relu(
    mlp, op: MLPOp, rmap: ResidualStreamMap, biased_linears: Set[Node] = frozenset()
):
    """Compile a Linear1 -> ReLU -> Linear2 chain through the MLP sublayer.

    linear1: maps input columns to mlp_slots (applying L1's weight + bias)
    relu: applied elementwise (no weights)
    linear2: maps mlp_slots to target columns (applying L2's weight + bias)
    """
    l2_node = op.node
    assert isinstance(l2_node, Linear)
    relu_node = l2_node.inputs[0]
    assert isinstance(relu_node, ReLU)
    l1_node = relu_node.inputs[0]
    assert isinstance(l1_node, Linear)

    input_node = l1_node.inputs[0]
    in_idx = rmap.resolve_indices(input_node)
    mlp_slots = op.mlp_slots
    out_idx = op.target_cols

    in_idx_t = torch.as_tensor(in_idx, dtype=torch.long)
    slots_t = torch.as_tensor(mlp_slots, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)

    target_dtype = mlp.linear1.output_matrix.dtype

    # linear1: rows=input columns, cols=mlp slots
    # L1 weight: (d_input, d_hidden), bias: (d_hidden,)
    mlp.linear1.output_matrix[in_idx_t.unsqueeze(1), slots_t.unsqueeze(0)] = (
        l1_node.output_matrix.to(target_dtype)
    )
    mlp.linear1.output_bias[slots_t] = l1_node.output_bias.to(target_dtype)

    # Fold deferred biased-Linear inputs into l1's hidden bias.
    # Biased Linears compiled in attention have their matrix applied but bias
    # deferred to compute_bias (written to linear2 output_bias). The chain
    # reads those columns in linear1, before output_bias is applied. Fold the
    # bias into l1's MLP bias so the chain sees correct values.
    if biased_linears:
        leaves = (
            flatten_concat_nodes([input_node])
            if isinstance(input_node, Concatenate)
            else [input_node]
        )
        offset = 0
        for leaf in leaves:
            if leaf in biased_linears:
                # contrib[j] = sum_i l1.W[offset+i, j] * leaf.bias[i]
                contrib = leaf.output_bias @ l1_node.output_matrix[offset : offset + len(leaf), :]
                mlp.linear1.output_bias[slots_t] += contrib.to(target_dtype)
            offset += len(leaf)

    # linear2: rows=mlp slots, cols=output columns
    # L2 weight: (d_hidden, d_output), bias: (d_output,)
    mlp.linear2.output_matrix[slots_t.unsqueeze(1), out_idx_t.unsqueeze(0)] = (
        l2_node.output_matrix.to(target_dtype)
    )
    mlp.linear2.output_bias[out_idx_t] = l2_node.output_bias.to(target_dtype)


def _write_compute_literal_value(mlp, op: MLPOp):
    """Write a constant value via MLP output bias."""
    node = op.node
    assert isinstance(node, LiteralValue)
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = mlp.linear2.output_bias.dtype
    mlp.linear2.output_bias[cols_t] = node.value[: len(op.target_cols)].to(target_dtype)


def _write_compute_bias(mlp, op: MLPOp):
    """Add bias to MLP output bias (for biased Linear split)."""
    node = op.node
    assert isinstance(node, Linear)
    cols_t = torch.as_tensor(op.target_cols, dtype=torch.long)
    target_dtype = mlp.linear2.output_bias.dtype
    mlp.linear2.output_bias[cols_t] += node.output_bias[: len(op.target_cols)].to(target_dtype)


def _write_compute_standalone_relu(
    mlp, op: MLPOp, rmap: ResidualStreamMap, biased_linears: Set[Node] = frozenset()
):
    """Compile a standalone ReLU through the MLP sublayer.

    linear1: identity mapping from input columns to MLP slots (bias=0)
    relu: applied elementwise (built-in)
    linear2: identity mapping from MLP slots to output columns (bias=0)

    The skip connection preserves the input. Output columns (initially zero)
    receive relu(input_value).
    """
    relu_node = op.node
    assert isinstance(relu_node, ReLU)

    input_node = relu_node.inputs[0]
    in_idx = rmap.resolve_indices(input_node)
    mlp_slots = op.mlp_slots
    out_idx = op.target_cols

    in_idx_t = torch.as_tensor(in_idx, dtype=torch.long)
    slots_t = torch.as_tensor(mlp_slots, dtype=torch.long)
    out_idx_t = torch.as_tensor(out_idx, dtype=torch.long)

    # linear1: input columns → MLP slots (identity, zero bias)
    mlp.linear1.output_matrix[in_idx_t, slots_t] = 1.0

    # Fold deferred biased-Linear inputs into MLP slot biases.
    # Identity mapping means slot[k] = input[k], so the bias folds directly.
    bias_dtype = mlp.linear1.output_bias.dtype
    if biased_linears:
        leaves = (
            flatten_concat_nodes([input_node])
            if isinstance(input_node, Concatenate)
            else [input_node]
        )
        offset = 0
        for leaf in leaves:
            if leaf in biased_linears:
                leaf_slots = slots_t[offset : offset + len(leaf)]
                mlp.linear1.output_bias[leaf_slots] += leaf.output_bias.to(bias_dtype)
            offset += len(leaf)

    # linear2: MLP slots → output columns (identity, zero bias)
    mlp.linear2.output_matrix[slots_t, out_idx_t] = 1.0
