"""Forward compiler: wires GraphAnalyzer, ResidualStreamMap, LayerScheduler,
and WeightWriter into a complete compilation pipeline.

Produces a HeadlessTransformer that can compute the output node's value
given input values.
"""

from typing import Optional

import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.feature_assignment import FeatureAssignment
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.forward.weight_writer import (
    AttnHeadOp,
    FFNOp,
    write_attn_sublayer,
    write_ffn_sublayer,
)
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.graph import Node, Linear
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU


def _count_layer_params(
    attn_ops: list[AttnHeadOp], ffn_ops: list[FFNOp], d: int, d_head: int
) -> int:
    """Count transformer parameters used by one layer's ops.

    Attention ops consume whole heads (4 * d * d_head params each).
    FFN ops consume slots (2*d + 2 params each) or bias entries (1 each).
    """
    params_per_head = 4 * d * d_head

    heads_used = 0
    for op in attn_ops:
        if op.op_type == "compute_attn":
            heads_used += 1
        elif op.op_type == "compute_linear":
            d_input = len(op.node.inputs[0])
            heads_used += (d_input + d_head - 1) // d_head
        elif op.op_type == "compute_add":
            heads_used += 2 * ((len(op.node) + d_head - 1) // d_head)
        elif op.op_type in ("cancel", "add_into"):
            heads_used += (len(op.node) + d_head - 1) // d_head

    slots_used = 0
    bias_entries = 0
    for op in ffn_ops:
        if op.ffn_slots:
            slots_used += len(op.ffn_slots)
        if op.op_type in ("compute_literal_value", "compute_bias"):
            bias_entries += len(op.target_cols)

    params_per_slot = 2 * d + 2  # linear1 column + bias + linear2 row + bias
    return heads_used * params_per_head + slots_used * params_per_slot + bias_entries


def forward_compile(
    d: int,
    d_head: int,
    output_node: Node,
    pos_encoding: Optional[PosEncoding] = None,
    verbose: bool = True,
    max_layers: int = 100,
    device: Optional[str] = "auto",
) -> HeadlessTransformer:
    """Compile a computation graph into a HeadlessTransformer.

    Args:
        d: Residual stream dimension.
        d_head: Attention head dimension.
        output_node: The graph node whose value should appear in the output.
        pos_encoding: Positional encoding node (required for attention ops).
        verbose: Print compilation progress.
        max_layers: Safety limit on number of layers.
        device: Target device — "auto" (default) uses GPU if available,
                "cpu"/"cuda" to force, or None to skip moving.

    Returns:
        A HeadlessTransformer whose compute() method reproduces
        output_node.compute() for the same inputs.
    """
    # 1. Analyze graph
    graph = GraphAnalyzer(output_node)
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]

    # Auto-create pos_encoding if needed (required for attention ops)
    if pos_encoding is None:
        pos_encoding = PosEncoding(d_pos=d_head)

    # 2. Initialize
    net = HeadlessTransformer(d, d_head, pos_encoding)
    residual_map = ResidualStreamMap(d)
    residual_map.allocate(pos_encoding)
    for node in input_nodes:
        residual_map.allocate(node)
    computed = set(input_nodes)
    scheduler = LayerScheduler(graph, d, d_head, pos_encoding)

    # Save input indices before scheduling (scheduling may free/reassign them)
    input_indices: dict[Node, list[int]] = {
        pos_encoding: residual_map.get_indices(pos_encoding)
    }
    for node in input_nodes:
        input_indices[node] = residual_map.get_indices(node)

    graph_params = sum(n.num_params() for n in graph.get_all_nodes())

    if verbose:
        print(
            f"Compiling {len(graph.get_all_nodes())} graph nodes "
            f"({graph_params:,} params) into d={d} transformer"
        )
        print(
            f"  {'Layer':<8} {'Ops':>8}  {'Layer params':>28}  "
            f"{'Stream in':>10}  {'Stream out':>11}"
        )

    # 3. Layer loop — seed with input node params (Embedding, etc.)
    total_params = sum(n.num_params() for n in input_nodes)
    for i in range(max_layers):
        if output_node in computed:
            break

        occupied_before = d - residual_map.get_free_count()

        layer = net.add_layer(append=True)
        attn_ops, ffn_ops = scheduler.schedule_layer(residual_map, computed)
        write_attn_sublayer(layer, attn_ops, residual_map, pos_encoding)
        write_ffn_sublayer(layer, ffn_ops, residual_map)

        layer_params = _count_layer_params(attn_ops, ffn_ops, d, d_head)
        total_params += layer_params
        occupied_after = d - residual_map.get_free_count()

        if verbose:
            n_ops = len(attn_ops) + len(ffn_ops)
            layer_capacity = layer.num_params()
            pct_params = 100 * layer_params / layer_capacity if layer_capacity else 0
            pct_before = 100 * occupied_before // d
            pct_after = 100 * occupied_after // d
            print(
                f"  {i:<8} {n_ops:>5} ops  "
                f"{layer_params:>9,}/{layer_capacity:,} ({pct_params:>4.1f}%)  "
                f"{occupied_before:>6}/{d} ({pct_before:>2}%)  "
                f"{occupied_after:>6}/{d} ({pct_after:>2}%)"
            )
    else:
        raise RuntimeError(
            f"Compilation did not converge in {max_layers} layers. "
            f"{len(graph.get_all_nodes() - computed)} nodes remaining."
        )

    transformer_params = sum(layer.num_params() for layer in net.layers)
    if verbose:
        pct_used = 100 * total_params / transformer_params if transformer_params else 0
        print(
            f"\n  {len(net.layers)} layers, "
            f"{total_params:,} / {transformer_params:,} params used "
            f"({pct_used:.1f}%)"
        )

    # Ensure at least one layer exists for FeatureAssignment states
    if not net.layers:
        net.add_layer(append=True)

    # 4. Build FeatureAssignment bridge from saved input indices
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].ffn.out_state
    fa = FeatureAssignment({in_state, out_state})
    for node, indices in input_indices.items():
        fa.assign(in_state, node, indices)
    fa.assign(out_state, output_node, residual_map.get_indices(output_node))
    net.feature_assignment = fa

    if device == "auto":
        net.to(get_device(verbose=verbose))
    elif device is not None:
        net.to(torch.device(device))

    return net
