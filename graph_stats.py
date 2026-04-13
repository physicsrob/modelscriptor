"""Print a breakdown of graph nodes and params by annotation.

Usage (via Makefile):
    make graph-stats
    make graph-stats ARGS="--rows-per-patch 25"
    make graph-stats ARGS="--scene multi --rows-per-patch 100"
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

from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.doom.game_graph import build_game_graph
from torchwright.graph import Attn, Node
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


def _collect_stats(all_nodes, d, d_head):
    """Collect per-annotation stats from a set of graph nodes."""
    stats = defaultdict(lambda: {"nodes": 0, "graph_params": 0, "alloc_params": 0})
    for node in all_nodes:
        key = node.annotation or "(none)"
        stats[key]["nodes"] += 1
        stats[key]["graph_params"] += node.num_params()
        stats[key]["alloc_params"] += _estimate_allocated_params(node, d, d_head)
    return stats


def _print_table(stats):
    """Print the annotation breakdown table with subtotals."""
    total_nodes = sum(s["nodes"] for s in stats.values())
    total_graph = sum(s["graph_params"] for s in stats.values())
    total_alloc = sum(s["alloc_params"] for s in stats.values())

    rows = sorted(stats.items())

    print(f"\n  {'Annotation':<35s} {'Nodes':>7s} "
          f"{'Graph params':>14s} {'Alloc params':>14s} {'Density':>8s}")
    print(f"  {'─' * 35} {'─' * 7} {'─' * 14} {'─' * 14} {'─' * 8}")

    def top_level(item):
        return item[0].split("/")[0]

    for group_key, group_items in groupby(rows, key=top_level):
        group_list = list(group_items)
        for key, s in group_list:
            density = (100.0 * s["graph_params"] / s["alloc_params"]
                       if s["alloc_params"] else 0)
            density_str = f"{density:.1f}%" if s["alloc_params"] else "—"
            print(f"  {key:<35s} {s['nodes']:>7,} "
                  f"{s['graph_params']:>14,} {s['alloc_params']:>14,} "
                  f"{density_str:>8s}")
        if len(group_list) > 1:
            gn = sum(s["nodes"] for _, s in group_list)
            gg = sum(s["graph_params"] for _, s in group_list)
            ga = sum(s["alloc_params"] for _, s in group_list)
            density = 100.0 * gg / ga if ga else 0
            density_str = f"{density:.1f}%" if ga else "—"
            print(f"  {'  ' + group_key + ' (total)':<35s} {gn:>7,} "
                  f"{gg:>14,} {ga:>14,} {density_str:>8s}")
        print()

    density = 100.0 * total_graph / total_alloc if total_alloc else 0
    print(f"  {'TOTAL':<35s} {total_nodes:>7,} "
          f"{total_graph:>14,} {total_alloc:>14,} {density:>7.1f}%")

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
    parser.add_argument("--rows-per-patch", type=int, default=10)
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
        W = args.width
        tex_w = textures[0].shape[0]
        tex_d_qk = 8 + tex_w + 1
        render_d_qk = W + 2
        min_d_head = max(render_d_qk, tex_d_qk)
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
          f"rows_per_patch={args.rows_per_patch}, {len(textures)} textures, "
          f"max_walls={max_walls}")
    print(f"Transformer config: d={d}, d_head={d_head}, d_hidden={d_hidden}, "
          f"n_heads={d // d_head}")

    output, pos = build_game_graph(
        config, textures, max_walls=max_walls, max_coord=max_coord,
        rows_per_patch=args.rows_per_patch,
    )

    all_nodes = get_ancestor_nodes({output, pos})
    stats = _collect_stats(all_nodes, d, d_head)

    print(f"\nGraph: {len(all_nodes):,} nodes")
    total_nodes, total_graph, total_alloc = _print_table(stats)

    # Estimate layer count: total allocated / layer capacity, rounded up.
    # This is a rough estimate — the actual compiler may use more or fewer
    # layers depending on scheduling constraints.
    layer_capacity = 4 * d * d + 2 * d * d_hidden + d_hidden + d
    n_layers_est = max(1, -(-total_alloc // layer_capacity))  # ceil division

    _print_summary(total_graph, total_alloc, n_layers_est, layer_capacity)


if __name__ == "__main__":
    main()
