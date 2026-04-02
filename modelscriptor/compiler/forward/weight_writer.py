"""Writes weight matrices into TransformerLayer components.

Each operation directly sets matrix entries using column indices from the
ResidualStreamMap. No strategies, no FeatureAssignment — just scatter writes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from modelscriptor.compiler.forward.residual_map import ResidualStreamMap
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.graph import Node, Linear, Attn, Add
from modelscriptor.graph.misc import Constant
from modelscriptor.graph.pos_encoding import PosEncoding, attention_hardness
from modelscriptor.graph.relu import ReLU


@dataclass
class AttnHeadOp:
    op_type: str  # "compute_attn", "compute_linear", "cancel", "add_into"
    node: Node
    target_cols: List[int]


@dataclass
class FFNOp:
    op_type: str  # "compute_relu", "compute_constant", "compute_bias"
    node: Node
    target_cols: List[int]
    ffn_slots: List[int] = field(default_factory=list)


def write_attn_sublayer(
    layer: TransformerLayer,
    ops: List[AttnHeadOp],
    residual_map: ResidualStreamMap,
    pos_encoding: Optional[PosEncoding],
):
    """Write attention head operations into a layer's AttnLayerComponent."""
    attn = layer.attn.attn
    for op in ops:
        if op.op_type == "compute_attn":
            _write_compute_attn(attn, op, residual_map)
        elif op.op_type == "compute_linear":
            _write_compute_linear(attn, op, residual_map, pos_encoding)
        elif op.op_type == "cancel":
            _write_cancel(attn, op, residual_map, pos_encoding)
        elif op.op_type == "add_into":
            _write_add_into(attn, op, residual_map, pos_encoding)
        else:
            raise ValueError(f"Unknown attn op_type: {op.op_type}")


def write_ffn_sublayer(
    layer: TransformerLayer,
    ops: List[FFNOp],
    residual_map: ResidualStreamMap,
):
    """Write FFN operations into a layer's FFNSubLayer components."""
    for op in ops:
        if op.op_type == "compute_relu":
            _write_compute_relu(layer.ffn, op, residual_map)
        elif op.op_type == "compute_constant":
            _write_compute_constant(layer.ffn, op)
        elif op.op_type == "compute_bias":
            _write_compute_bias(layer.ffn, op)
        elif op.op_type == "compute_standalone_relu":
            _write_compute_standalone_relu(layer.ffn, op, residual_map)
        else:
            raise ValueError(f"Unknown ffn op_type: {op.op_type}")


# ---------------------------------------------------------------------------
# Attention operations
# ---------------------------------------------------------------------------


def _scatter_attn_head(attn, head, q_idx, k_idx, v_idx, o_idx,
                       q_mat, k_mat, v_mat, o_mat, d_head):
    """Scatter strategy matrices into one attention head's weight tensors."""
    for i, idx in enumerate(q_idx):
        attn.query_matrix[head, idx, :d_head] = q_mat[i, :d_head]
    for i, idx in enumerate(k_idx):
        attn.key_matrix[head, idx, :d_head] = k_mat[i, :d_head]
    for i, idx in enumerate(v_idx):
        attn.value_matrix[head, idx, :d_head] = v_mat[i, :d_head]
    for i, idx in enumerate(o_idx):
        attn.output_matrix[head, :d_head, idx] = o_mat[:d_head, i]


def _allocate_head(attn):
    """Allocate the next available attention head."""
    assert attn.used_heads < attn.n_heads, "Ran out of attention heads"
    head = attn.used_heads
    attn.used_heads += 1
    return head


def _write_compute_attn(attn, op: AttnHeadOp, rmap: ResidualStreamMap):
    """Copy an Attn node's Q/K/V/O matrices into one head."""
    node = op.node
    assert isinstance(node, Attn)

    query_in, key_in, value_in = node.inputs
    q_idx = rmap.get_node_indices(query_in)
    k_idx = rmap.get_node_indices(key_in)
    v_idx = rmap.get_node_indices(value_in)
    o_idx = op.target_cols

    # Pad node matrices to layer d_head if needed
    node_d_head = node.d_head
    layer_d_head = attn.d_head

    q_mat = F.pad(node.query_matrix, (0, layer_d_head - node_d_head))
    k_mat = F.pad(node.key_matrix, (0, layer_d_head - node_d_head))
    v_mat = F.pad(node.value_matrix, (0, layer_d_head - node_d_head))
    o_mat = F.pad(node.output_matrix, (0, 0, 0, layer_d_head - node_d_head))

    head = _allocate_head(attn)
    _scatter_attn_head(attn, head, q_idx, k_idx, v_idx, o_idx,
                       q_mat, k_mat, v_mat, o_mat, layer_d_head)


def _current_pos_attn_matrices(pos_encoding, d_head):
    """Build Q/K matrices for current-position attention."""
    d_pos = len(pos_encoding)
    q_mat = attention_hardness * torch.eye(d_pos, d_head)
    k_mat = torch.eye(d_pos, d_head)
    return q_mat, k_mat


def _write_compute_linear(attn, op: AttnHeadOp, rmap: ResidualStreamMap,
                          pos_encoding: PosEncoding):
    """Compile a zero-bias Linear via current-position attention.

    Q/K attend to current position via pos_encoding.
    V reads from the Linear's input columns.
    O applies the Linear's weight matrix to target columns.
    """
    node = op.node
    assert isinstance(node, Linear)

    input_node = node.inputs[0]
    d_head = attn.d_head
    d_input = len(input_node)

    pe_idx = rmap.get_indices(pos_encoding)
    v_idx = rmap.get_indices(input_node)
    o_idx = op.target_cols

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)
    v_mat = torch.eye(d_input, d_head)
    # O = weight matrix padded: (d_head, d_output)
    o_mat = F.pad(node.output_matrix, (0, 0, 0, d_head - d_input))

    head = _allocate_head(attn)
    _scatter_attn_head(attn, head, pe_idx, pe_idx, v_idx, o_idx,
                       q_mat, k_mat, v_mat, o_mat, d_head)


def _write_cancel(attn, op: AttnHeadOp, rmap: ResidualStreamMap,
                  pos_encoding: PosEncoding):
    """Cancel a node: V=identity, O=-identity. Skip adds x + (-x) = 0."""
    node = op.node
    d_head = attn.d_head
    d_node = len(node)

    pe_idx = rmap.get_indices(pos_encoding)
    node_idx = op.target_cols

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)
    v_mat = torch.eye(d_node, d_head)
    o_mat = -torch.eye(d_head, d_node)

    head = _allocate_head(attn)
    _scatter_attn_head(attn, head, pe_idx, pe_idx, node_idx, node_idx,
                       q_mat, k_mat, v_mat, o_mat, d_head)


def _write_add_into(attn, op: AttnHeadOp, rmap: ResidualStreamMap,
                    pos_encoding: PosEncoding):
    """Add(dead, live): copy live's values to dead's columns via attention.

    target_cols are the dead addend's columns (now owned by the Add node
    after reassign). The live addend is whichever input is still allocated
    in the residual map. Skip connection adds: dead + live = Add(dead, live).
    """
    node = op.node
    assert isinstance(node, Add)

    # Determine which input is live (still allocated in residual map)
    if rmap.is_allocated(node.inputs[0]):
        live_addend = node.inputs[0]
    else:
        live_addend = node.inputs[1]
    d_head = attn.d_head
    d_live = len(live_addend)

    pe_idx = rmap.get_indices(pos_encoding)
    v_idx = rmap.get_indices(live_addend)
    o_idx = op.target_cols  # dead addend's columns

    q_mat, k_mat = _current_pos_attn_matrices(pos_encoding, d_head)
    v_mat = torch.eye(d_live, d_head)
    o_mat = torch.eye(d_head, d_live)

    head = _allocate_head(attn)
    _scatter_attn_head(attn, head, pe_idx, pe_idx, v_idx, o_idx,
                       q_mat, k_mat, v_mat, o_mat, d_head)


# ---------------------------------------------------------------------------
# FFN operations
# ---------------------------------------------------------------------------


def _write_compute_relu(ffn, op: FFNOp, rmap: ResidualStreamMap):
    """Compile a Linear1 -> ReLU -> Linear2 chain through the FFN.

    linear1: maps input columns to ffn_slots (applying L1's weight + bias)
    relu: applied elementwise (no weights)
    linear2: maps ffn_slots to target columns (applying L2's weight + bias)
    """
    l2_node = op.node
    assert isinstance(l2_node, Linear)
    relu_node = l2_node.inputs[0]
    assert isinstance(relu_node, ReLU)
    l1_node = relu_node.inputs[0]
    assert isinstance(l1_node, Linear)

    input_node = l1_node.inputs[0]
    in_idx = rmap.get_node_indices(input_node)
    ffn_slots = op.ffn_slots
    out_idx = op.target_cols

    # linear1: input columns -> ffn slots
    # L1 weight: (d_input, d_intermediate), bias: (d_intermediate,)
    for i, in_col in enumerate(in_idx):
        for j, slot in enumerate(ffn_slots):
            ffn.linear1.output_matrix[in_col, slot] = l1_node.output_matrix[i, j]
    for j, slot in enumerate(ffn_slots):
        ffn.linear1.output_bias[slot] = l1_node.output_bias[j]

    # linear2: ffn slots -> output columns
    # L2 weight: (d_intermediate, d_output), bias: (d_output,)
    for i, slot in enumerate(ffn_slots):
        for j, out_col in enumerate(out_idx):
            ffn.linear2.output_matrix[slot, out_col] = l2_node.output_matrix[i, j]
    for j, out_col in enumerate(out_idx):
        ffn.linear2.output_bias[out_col] = l2_node.output_bias[j]


def _write_compute_constant(ffn, op: FFNOp):
    """Write a constant value via FFN output bias."""
    node = op.node
    assert isinstance(node, Constant)
    for i, col in enumerate(op.target_cols):
        ffn.linear2.output_bias[col] = node.value[i]


def _write_compute_bias(ffn, op: FFNOp):
    """Add bias to FFN output bias (for biased Linear split)."""
    node = op.node
    assert isinstance(node, Linear)
    for i, col in enumerate(op.target_cols):
        ffn.linear2.output_bias[col] += node.output_bias[i]


def _write_compute_standalone_relu(ffn, op: FFNOp, rmap: ResidualStreamMap):
    """Compile a standalone ReLU through the FFN.

    linear1: identity mapping from input columns to FFN slots (bias=0)
    relu: applied elementwise (built-in)
    linear2: identity mapping from FFN slots to output columns (bias=0)

    The skip connection preserves the input. Output columns (initially zero)
    receive relu(input_value).
    """
    relu_node = op.node
    assert isinstance(relu_node, ReLU)

    input_node = relu_node.inputs[0]
    in_idx = rmap.get_node_indices(input_node)
    ffn_slots = op.ffn_slots
    out_idx = op.target_cols

    # linear1: input columns → FFN slots (identity, zero bias)
    for in_col, slot in zip(in_idx, ffn_slots):
        ffn.linear1.output_matrix[in_col, slot] = 1.0

    # linear2: FFN slots → output columns (identity, zero bias)
    for slot, out_col in zip(ffn_slots, out_idx):
        ffn.linear2.output_matrix[slot, out_col] = 1.0
