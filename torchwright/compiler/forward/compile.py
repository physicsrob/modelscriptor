"""Forward compiler: wires GraphAnalyzer, ResidualStreamMap, LayerScheduler,
and WeightWriter into a complete compilation pipeline.

Produces a HeadlessTransformer that can compute the output node's value
given input values.
"""

import time
from typing import Callable, Optional

import torch

from torchwright.compiler.device import get_device
from torchwright.compiler.residual_assignment import ResidualAssignment
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.residual_map import ResidualStreamMap
from torchwright.compiler.forward.scheduler import LayerScheduler
from torchwright.compiler.forward.weight_writer import (
    AttnHeadOp,
    MLPOp,
    write_attn_sublayer,
    write_mlp_sublayer,
)
from torchwright.compiler.groups.transformer_layer import TransformerLayer
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import Node, Linear, Concatenate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU


def _count_layer_params(
    attn_ops: list[AttnHeadOp],
    mlp_ops: list[MLPOp],
    d: int,
    d_head: int,
) -> int:
    """Count transformer parameters used by one layer's ops.

    Attention ops consume whole heads (4 * d * d_head params each).
    MLP ops consume slots (2*d + 2 params each) or bias entries (1 each).
    The per-slot cost is independent of ``d_hidden`` — each occupied
    hidden slot still costs one column in linear1 plus one row in linear2
    plus two biases.
    """
    params_per_head = 4 * d * d_head

    heads_used = 0
    for op in attn_ops:
        if op.op_type == "compute_attn":
            heads_used += (op.node.d_v + d_head - 1) // d_head
        elif op.op_type == "compute_linear":
            d_input = len(op.node.inputs[0])
            heads_used += (d_input + d_head - 1) // d_head
        elif op.op_type == "compute_add":
            heads_used += 2 * ((len(op.node) + d_head - 1) // d_head)
        elif op.op_type in ("cancel", "add_into"):
            heads_used += (len(op.node) + d_head - 1) // d_head

    slots_used = 0
    bias_entries = 0
    for op in mlp_ops:
        if op.mlp_slots:
            slots_used += len(op.mlp_slots)
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
    on_layer_compiled: Optional[Callable[[int, TransformerLayer], None]] = None,
    d_hidden: Optional[int] = None,
    on_node_scheduled: Optional[Callable[[Node, int], None]] = None,
    trim_heads: bool = True,
    overlays: Optional[dict] = None,
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
        d_hidden: MLP hidden width per layer (the per-layer pool of
            ``L1->ReLU->L2`` neurons).  Independent of ``d``; defaults
            to ``d`` for backwards compatibility.
        on_layer_compiled: Optional streaming hook, called with
            ``(layer_index, layer)`` after each layer's weights are fully
            written.  The callback may extract weight tensors and then
            null the component weight attributes to reclaim memory
            before the next layer is allocated.  The residual-stream
            state objects (``layer.attn.in_state`` / ``layer.mlp.out_state``)
            stay valid regardless and are consumed later when building
            ``residual_assignment``.
        overlays: Optional dict mapping output_node -> (input_node, target_cols)
            for delta transfer. When provided, a final layer is added that
            transfers each output's value to the specified target columns
            via delta: target += (output - target). This enables overlaid
            I/O where output replaces input in-place.

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

    if d_hidden is None:
        d_hidden = d

    # 2. Initialize
    net = HeadlessTransformer(d, d_head, pos_encoding, d_hidden=d_hidden)
    residual_map = ResidualStreamMap(d)
    residual_map.allocate(pos_encoding)
    for node in input_nodes:
        residual_map.allocate(node)
    computed = set(input_nodes)
    scheduler = LayerScheduler(graph, d, d_head, pos_encoding, d_hidden=d_hidden)

    # Save input indices before scheduling (scheduling may free/reassign them)
    input_indices: dict[Node, list[int]] = {
        pos_encoding: residual_map.get_indices(pos_encoding)
    }
    for node in input_nodes:
        input_indices[node] = residual_map.get_indices(node)

    graph_params = sum(n.num_params() for n in graph.get_all_nodes())

    # Per-layer tensor capacity (Q/K/V/O attention matrices + linear1/linear2
    # weights & biases).  Computed once instead of via `layer.num_params()` so
    # the verbose log still works after `on_layer_compiled` nulls the layer's
    # weight attributes.  Decomposes as 4*d*d (attention QKVO) +
    # 2*d*d_hidden (rectangular MLP matrices) + d_hidden (linear1 bias) +
    # d (linear2 bias).
    layer_capacity = 4 * d * d + 2 * d * d_hidden + d_hidden + d

    if verbose:
        print(
            f"Compiling {len(graph.get_all_nodes())} graph nodes "
            f"({graph_params:,} params) into d={d} transformer"
        )
        print(
            f"  {'Layer':<8} {'Ops':>8}  {'Layer params':>28}  "
            f"{'Stream in':>10}  {'Stream out':>11}  {'Time':>10}"
        )

    # Per-layer snapshots of ``residual_map._node_to_indices``, one per
    # sublayer boundary.  Consumed by :mod:`torchwright.debug.probe` so
    # it can look up where each graph node lives in the compiled
    # residual stream at intermediate layers.  Each entry is
    # ``(ResidualStreamState, {Node: List[int]})`` keyed by the sublayer
    # whose post-skip tensor carries those values.  Only the
    # post-MLP-sublayer state is captured per layer — attn+mlp are both
    # scheduled inside a single ``schedule_layer`` call, so there is no
    # clean observation point between them.
    sublayer_snapshots: list = []

    # 3. Layer loop — seed with input node params (Embedding, etc.)
    total_params = sum(n.num_params() for n in input_nodes)
    total_layer_time = 0.0
    for i in range(max_layers):
        if output_node in computed:
            break

        prev_computed = set(computed) if on_node_scheduled else None
        occupied_before = d - residual_map.get_free_count()

        t_layer_start = time.perf_counter()
        layer = net.add_layer(append=True)
        t_schedule_start = time.perf_counter()
        attn_ops, mlp_ops, biased_linears = scheduler.schedule_layer(
            residual_map, computed
        )
        t_attn_start = time.perf_counter()
        write_attn_sublayer(layer, attn_ops, residual_map, pos_encoding)
        t_mlp_start = time.perf_counter()
        write_mlp_sublayer(layer, mlp_ops, residual_map, set(biased_linears))
        t_layer_end = time.perf_counter()

        layer_time = t_layer_end - t_layer_start
        alloc_time = t_schedule_start - t_layer_start
        schedule_time = t_attn_start - t_schedule_start
        attn_time = t_mlp_start - t_attn_start
        mlp_time = t_layer_end - t_mlp_start
        total_layer_time += layer_time

        # Mark Concatenate nodes as computed when all leaf inputs are done
        for node in graph.get_all_nodes():
            if isinstance(node, Concatenate) and node not in computed:
                if all(leaf in computed for leaf in flatten_concat_nodes([node])):
                    computed.add(node)

        if on_node_scheduled is not None:
            for node in computed - prev_computed:
                on_node_scheduled(node, i)

        layer_params = _count_layer_params(attn_ops, mlp_ops, d, d_head)
        total_params += layer_params
        occupied_after = d - residual_map.get_free_count()

        if verbose:
            n_ops = len(attn_ops) + len(mlp_ops)
            pct_params = 100 * layer_params / layer_capacity if layer_capacity else 0
            pct_before = 100 * occupied_before // d
            pct_after = 100 * occupied_after // d
            print(
                f"  {i:<8} {n_ops:>5} ops  "
                f"{layer_params:>9,}/{layer_capacity:,} ({pct_params:>4.1f}%)  "
                f"{occupied_before:>6}/{d} ({pct_before:>2}%)  "
                f"{occupied_after:>6}/{d} ({pct_after:>2}%)  "
                f"{layer_time*1000:>7.1f}ms "
                f"(alloc {alloc_time*1000:.0f} sch {schedule_time*1000:.0f} "
                f"attn {attn_time*1000:.0f} mlp {mlp_time*1000:.0f})",
                flush=True,
            )

        # Snapshot the live residual-column assignments at the end of
        # this layer's MLP sublayer.  Copying the dict is deliberate:
        # subsequent layers will mutate residual_map via reassign/free
        # and we need the frozen "as of this state" view.
        sublayer_snapshots.append(
            (
                layer.mlp.out_state,
                {n: list(cols) for n, cols in residual_map._node_to_indices.items()},
            )
        )

        if on_layer_compiled is not None:
            on_layer_compiled(i, layer)
    else:
        raise RuntimeError(
            f"Compilation did not converge in {max_layers} layers. "
            f"{len(graph.get_all_nodes() - computed)} nodes remaining."
        )

    # layer_capacity is constant per layer; avoids touching layer tensors,
    # which may have been freed by on_layer_compiled.
    transformer_params = layer_capacity * len(net.layers)
    if verbose:
        pct_used = 100 * total_params / transformer_params if transformer_params else 0
        print(
            f"\n  {len(net.layers)} layers, "
            f"{total_params:,} / {transformer_params:,} params used "
            f"({pct_used:.1f}%), "
            f"{total_layer_time:.2f}s total layer time"
        )

    # 3b. Delta transfer layer for overlaid I/O
    # When overlays is provided, add a final layer that transfers each output
    # value to the input's columns via delta: target += (output - target).
    if overlays:
        delta_layer = net.add_layer(append=True)
        delta_ops = []
        for out_node, (in_node, target_cols) in overlays.items():
            # Source columns: where the output value was computed
            source_cols = residual_map.get_indices(out_node)
            # Subtract columns: same as target (the input columns)
            subtract_cols = target_cols
            delta_ops.append(AttnHeadOp(
                op_type="delta_transfer",
                node=out_node,
                target_cols=target_cols,
                source_cols=source_cols,
                subtract_cols=subtract_cols,
            ))
        write_attn_sublayer(delta_layer, delta_ops, residual_map, pos_encoding)
        if verbose:
            print(f"  Delta transfer layer: {len(delta_ops)} overlays")
        if on_layer_compiled is not None:
            on_layer_compiled(len(net.layers) - 1, delta_layer)

    # Ensure at least one layer exists for ResidualAssignment states.
    # If compile produced zero layers (trivial graph), run the callback on
    # the placeholder too so every layer in net.layers is consistently in
    # the extracted state.
    if not net.layers:
        fallback_layer = net.add_layer(append=True)
        if on_layer_compiled is not None:
            on_layer_compiled(0, fallback_layer)

    # 4. Build ResidualAssignment bridge from saved input indices
    in_state = net.layers[0].attn.in_state
    out_state = net.layers[-1].mlp.out_state
    # Include the per-sublayer snapshots so the debug probe can look up
    # where each graph node lives in the residual stream at any
    # intermediate point.  The top-level in_state / out_state are still
    # populated with input + output indices for the runtime's
    # get_input_res_stream / compute paths.
    all_states = {in_state, out_state}
    for state, _ in sublayer_snapshots:
        all_states.add(state)
    ra = ResidualAssignment(all_states)
    for node, indices in input_indices.items():
        ra.assign(in_state, node, indices)
    for state, snapshot in sublayer_snapshots:
        for node, cols in snapshot.items():
            ra.assign(state, node, list(cols))
    if isinstance(output_node, Concatenate):
        for leaf in flatten_concat_nodes([output_node]):
            ra.assign(out_state, leaf, residual_map.get_indices(leaf))
    else:
        ra.assign(out_state, output_node, residual_map.get_indices(output_node))
    net.residual_assignment = ra

    if trim_heads:
        for layer in net.layers:
            layer.attn.attn.trim_unused_heads()

    if device == "auto":
        net.to(get_device(verbose=verbose))
    elif device is not None:
        net.to(torch.device(device))

    return net
