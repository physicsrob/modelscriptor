"""Forward compiler: wires GraphAnalyzer, ResidualStreamMap, LayerScheduler,
and WeightWriter into a complete compilation pipeline.

Produces a HeadlessTransformer that can compute the output node's value
given input values.
"""

from typing import Optional

from modelscriptor.compiler.feature_assignment import FeatureAssignment
from modelscriptor.compiler.forward.graph_analysis import GraphAnalyzer
from modelscriptor.compiler.forward.residual_map import ResidualStreamMap
from modelscriptor.compiler.forward.scheduler import LayerScheduler
from modelscriptor.compiler.forward.weight_writer import (
    write_attn_sublayer,
    write_ffn_sublayer,
)
from modelscriptor.compiler.transformer import HeadlessTransformer
from modelscriptor.graph import Node
from modelscriptor.graph.pos_encoding import PosEncoding


def forward_compile(
    d: int,
    d_head: int,
    output_node: Node,
    pos_encoding: Optional[PosEncoding] = None,
    verbose: bool = True,
    max_layers: int = 100,
) -> HeadlessTransformer:
    """Compile a computation graph into a HeadlessTransformer.

    Args:
        d: Residual stream dimension.
        d_head: Attention head dimension.
        output_node: The graph node whose value should appear in the output.
        pos_encoding: Positional encoding node (required for attention ops).
        verbose: Print compilation progress.
        max_layers: Safety limit on number of layers.

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
    input_indices = {pos_encoding: residual_map.get_indices(pos_encoding)}
    for node in input_nodes:
        input_indices[node] = residual_map.get_indices(node)

    if verbose:
        print(
            f"Forward compile: {len(graph.get_all_nodes())} nodes, "
            f"{len(input_nodes)} inputs, d={d}, d_head={d_head}"
        )

    # 3. Layer loop
    for i in range(max_layers):
        if output_node in computed:
            break

        layer = net.add_layer(end=True)
        attn_ops, ffn_ops = scheduler.schedule_layer(residual_map, computed)
        write_attn_sublayer(layer, attn_ops, residual_map, pos_encoding)
        write_ffn_sublayer(layer, ffn_ops, residual_map)

        if verbose:
            n_attn = len(attn_ops)
            n_ffn = len(ffn_ops)
            n_computed = len(computed)
            free = residual_map.get_free_count()
            print(
                f"  Layer {i}: {n_attn} attn ops, {n_ffn} ffn ops, "
                f"{n_computed} computed, {free} free cols"
            )
    else:
        raise RuntimeError(
            f"Compilation did not converge in {max_layers} layers. "
            f"{len(graph.get_all_nodes() - computed)} nodes remaining."
        )

    if verbose:
        print(f"  Done: {len(net.layers)} layers")

    # Ensure at least one layer exists for FeatureAssignment states
    if not net.layers:
        net.add_layer(end=True)

    # 4. Build FeatureAssignment bridge from saved input indices
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].ffn.out_state
    fa = FeatureAssignment({in_state, out_state})
    for node, indices in input_indices.items():
        fa.assign(in_state, node, indices)
    fa.assign(out_state, output_node, residual_map.get_indices(output_node))
    net.feature_assignment = fa

    return net
