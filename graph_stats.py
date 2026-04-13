"""Print a breakdown of graph nodes and params by annotation.

Usage (via Makefile):
    make graph-stats
    make graph-stats ARGS="--chunk-size 25"
    make graph-stats ARGS="--scene multi --chunk-size 100"
    make graph-stats ARGS="--d 4096 --d-head 128"

Three levels of parameter accounting:

    Graph params       Non-zero values in the graph's small weight matrices.
                       A Linear(3→5) contributes 3×5 + 5 = 20 params.
                       This is the actual information content of the model.

    Allocated params   Transformer weight-matrix entries reserved by the
                       compiler for these graph ops.  A Linear(3→5) is
                       compiled into an attention head that reserves
                       4 × d × d_head entries — most of which are zero.
                       This is the cost you pay in memory and compute.

    Total capacity     Every entry in every layer's weight matrices
                       (attention QKVO + MLP linear1/linear2 + biases).
                       Layers × (4·d² + 2·d·d_hidden + d_hidden + d).

    The ratio graph/allocated is the density within the "used" portion.
    The ratio allocated/total is the layer utilization reported by the
    compiler (the "params used" percentage).
"""

import argparse
from collections import defaultdict
from itertools import groupby

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.graph import Add, Attn, Node
from torchwright.graph.linear import Linear
from torchwright.graph.relu import ReLU
from torchwright.reference_renderer.scenes import (
    box_room_textured,
    multi_room_textured,
)
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


# ── Helpers ──────────────────────────────────────────────────────────


def _estimate_allocated_params(node: Node, d: int, d_head: int) -> int:
    """Estimate transformer params the compiler allocates for one graph node.

    Attention sublayer:
        Each Linear or Attn op is compiled into one or more attention
        heads.  Each head reserves 4 × d × d_head entries in the QKVO
        weight matrices (most will be zero — only the graph node's
        small matrix is embedded).

    MLP sublayer:
        A Linear→ReLU→Linear chain (one "MLP op") uses d_hidden slots.
        Each slot costs 2·d + 2 entries (one column of linear1, one row
        of linear2, plus two biases).  For simplicity we count the
        intermediate ReLU as zero and attribute cost to the enclosing
        Linears.

    This is an estimate — the real compiler packs ops into layers and
    may share heads across ops — but it gives the right order of
    magnitude and makes the density ratio meaningful.
    """
    if isinstance(node, Attn):
        # Attn node compiled into ceil(d_v / d_head) heads.
        n_heads = (node.d_v + d_head - 1) // d_head
        return n_heads * 4 * d * d_head
    elif isinstance(node, Linear):
        d_input = len(node.inputs[0])
        # Linear compiled into ceil(d_input / d_head) heads.
        n_heads = (d_input + d_head - 1) // d_head
        return n_heads * 4 * d * d_head
    # ReLU, Add, Concatenate, InputNode, etc. — no direct param cost
    return 0


def _collect_stats(all_nodes, d, d_head, node_to_layer=None):
    """Collect per-annotation stats from a set of graph nodes."""
    # Build consumer map so we can detect LRL chain membership
    consumers = defaultdict(set)
    for node in all_nodes:
        for inp in node.inputs:
            consumers[inp.node_id].add(node)

    # Linears that are part of Linear→ReLU→Linear chains (compiled as MLP,
    # not attention).  Linear1 feeds a ReLU; Linear2 reads from a ReLU.
    lrl_linears: set = set()
    for node in all_nodes:
        if isinstance(node, ReLU):
            if node.inputs and isinstance(node.inputs[0], Linear):
                lrl_linears.add(node.inputs[0].node_id)
            for consumer in consumers.get(node.node_id, set()):
                if isinstance(consumer, Linear):
                    lrl_linears.add(consumer.node_id)

    stats = defaultdict(lambda: {
        "nodes": 0, "graph_params": 0, "alloc_params": 0,
        "neurons": 0, "heads": 0, "layers": set(),
    })
    for node in all_nodes:
        key = node.annotation or "(none)"
        stats[key]["nodes"] += 1
        stats[key]["graph_params"] += node.num_params()
        stats[key]["alloc_params"] += _estimate_allocated_params(node, d, d_head)

        if isinstance(node, ReLU):
            # ReLU width = hidden dimension of the LRL chain
            stats[key]["neurons"] += len(node)
        elif isinstance(node, Attn):
            stats[key]["heads"] += (node.d_v + d_head - 1) // d_head
        elif isinstance(node, Add):
            # Add compiled as attention head(s)
            stats[key]["heads"] += (len(node) + d_head - 1) // d_head
        elif isinstance(node, Linear) and node.node_id not in lrl_linears:
            # Standalone Linear (not part of LRL) → compiled as attention
            d_input = len(node.inputs[0])
            stats[key]["heads"] += (d_input + d_head - 1) // d_head

        if node_to_layer and node.node_id in node_to_layer:
            stats[key]["layers"].add(node_to_layer[node.node_id])

    return stats


def _fmt_layers(layer_set):
    """Format a set of layer indices as 'min-max (count)' or '—'."""
    if not layer_set:
        return "—"
    lo, hi = min(layer_set), max(layer_set)
    n = len(layer_set)
    return f"{lo:>3}-{hi:<3} ({n:>2})"


def _print_table(stats, has_layers=False):
    """Print the annotation breakdown table with subtotals."""
    total_nodes = sum(s["nodes"] for s in stats.values())
    total_graph = sum(s["graph_params"] for s in stats.values())
    total_alloc = sum(s["alloc_params"] for s in stats.values())
    total_neurons = sum(s["neurons"] for s in stats.values())
    total_heads = sum(s["heads"] for s in stats.values())

    rows = sorted(stats.items())

    layer_hdr = f"{'Layers':>14s} " if has_layers else ""
    layer_sep = f"{'─' * 14} " if has_layers else ""
    print(f"\n  {'Annotation':<35s} {'Nodes':>7s} "
          f"{'Neurons':>9s} {'Heads':>7s} {layer_hdr}"
          f"{'Graph params':>14s} {'Alloc params':>14s}")
    print(f"  {'─' * 35} {'─' * 7} {'─' * 9} {'─' * 7} {layer_sep}{'─' * 14} {'─' * 14}")

    def top_level(item):
        return item[0].split("/")[0]

    for group_key, group_items in groupby(rows, key=top_level):
        group_list = list(group_items)
        for key, s in group_list:
            layer_col = f"{_fmt_layers(s['layers']):>14s} " if has_layers else ""
            print(f"  {key:<35s} {s['nodes']:>7,} "
                  f"{s['neurons']:>9,} {s['heads']:>7,} {layer_col}"
                  f"{s['graph_params']:>14,} {s['alloc_params']:>14,}")
        if len(group_list) > 1:
            gn = sum(s["nodes"] for _, s in group_list)
            gneu = sum(s["neurons"] for _, s in group_list)
            gh = sum(s["heads"] for _, s in group_list)
            gg = sum(s["graph_params"] for _, s in group_list)
            ga = sum(s["alloc_params"] for _, s in group_list)
            gl = set().union(*(s["layers"] for _, s in group_list))
            layer_col = f"{_fmt_layers(gl):>14s} " if has_layers else ""
            print(f"  {'  ' + group_key + ' (total)':<35s} {gn:>7,} "
                  f"{gneu:>9,} {gh:>7,} {layer_col}"
                  f"{gg:>14,} {ga:>14,}")
        print()

    all_layers = set().union(*(s["layers"] for s in stats.values()))
    layer_col = f"{_fmt_layers(all_layers):>14s} " if has_layers else ""
    print(f"  {'TOTAL':<35s} {total_nodes:>7,} "
          f"{total_neurons:>9,} {total_heads:>7,} {layer_col}"
          f"{total_graph:>14,} {total_alloc:>14,}")

    return total_nodes, total_graph, total_alloc


def _print_summary(total_graph, total_alloc, n_layers, layer_capacity):
    """Print the three-level parameter summary."""
    total_capacity = n_layers * layer_capacity
    utilization = 100.0 * total_alloc / total_capacity if total_capacity else 0
    density = 100.0 * total_graph / total_alloc if total_alloc else 0
    overall = 100.0 * total_graph / total_capacity if total_capacity else 0

    print(f"\n  Parameter summary ({n_layers} layers estimated):\n")
    print(f"    Graph params (non-zero weights):    {total_graph:>14,}")
    print(f"    Allocated params (head/slot cost):   {total_alloc:>14,}")
    print(f"    Total transformer capacity:          {total_capacity:>14,}")
    print()
    print(f"    Density (graph / allocated):         {density:>13.2f}%"
          f"   ← how sparse each allocated head/slot is")
    print(f"    Utilization (allocated / capacity):  {utilization:>13.2f}%"
          f"   ← how full each layer is")
    print(f"    Overall (graph / capacity):          {overall:>13.2f}%"
          f"   ← fraction of transformer that is non-zero")
    print()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Print game graph stats by annotation")
    parser.add_argument("--scene", default="box", choices=["box", "multi"])
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--tex-size", type=int, default=64)
    parser.add_argument("--max-walls", type=int, default=None)
    parser.add_argument("--d", type=int, default=2048,
                        help="Model dimension (for allocated param estimates)")
    parser.add_argument("--d-head", type=int, default=None,
                        help="Head dimension (default: auto from width/tex_size)")
    parser.add_argument("--d-hidden", type=int, default=None,
                        help="MLP hidden dimension (default: 4*d)")
    args = parser.parse_args()

    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path="doom1.wad", tex_size=args.tex_size,
        )
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path="doom1.wad", tex_size=args.tex_size,
        )
        max_coord = 15.0

    max_walls = args.max_walls if args.max_walls else max(8, len(segments))

    # Compute d_head the same way compile_game does
    d = args.d
    if args.d_head is None:
        tex_w = textures[0].shape[0]
        min_d_head = compute_min_d_head(max_walls, tex_w)
        d_head = 1
        while d_head < min_d_head:
            d_head *= 2
    else:
        d_head = args.d_head

    d_hidden = args.d_hidden if args.d_hidden else 4 * d

    config = RenderConfig(
        screen_width=args.width,
        screen_height=args.height,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    print(f"Building graph: {args.scene} scene, {args.width}x{args.height}, "
          f"chunk_size={args.chunk_size}, {len(textures)} textures, "
          f"max_walls={max_walls}")
    print(f"Transformer config: d={d}, d_head={d_head}, d_hidden={d_hidden}, "
          f"n_heads={d // d_head}")

    output, pos = build_game_graph(
        config, textures, max_walls=max_walls, max_coord=max_coord,
        chunk_size=args.chunk_size,
    )

    all_nodes = get_ancestor_nodes({output, pos})

    # Compile to get actual layer assignments
    print("Compiling...")
    node_to_layer: dict = {}
    def _track(node, layer_idx):
        node_to_layer[node.node_id] = layer_idx

    forward_compile(
        d, d_head, output, pos,
        verbose=False, max_layers=400,
        d_hidden=d_hidden if d_hidden != 4 * d else None,
        device=None,
        on_node_scheduled=_track,
    )
    n_layers = max(node_to_layer.values()) + 1 if node_to_layer else 0

    stats = _collect_stats(all_nodes, d, d_head, node_to_layer)

    print(f"\nGraph: {len(all_nodes):,} nodes, compiled to {n_layers} layers")
    total_nodes, total_graph, total_alloc = _print_table(stats, has_layers=True)

    layer_capacity = 4 * d * d + 2 * d * d_hidden + d_hidden + d
    _print_summary(total_graph, total_alloc, n_layers, layer_capacity)


if __name__ == "__main__":
    main()
