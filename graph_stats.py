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

Critical path analysis:

    The critical path is the longest chain of sequential dependencies in the
    graph. This determines the minimum number of layers needed. Optimizations
    that shorten the critical path directly reduce layer count.

    Contiguous chains of the same annotation on the critical path are often
    the best optimization targets - they represent a single logical operation
    that might be restructurable.
"""

import argparse
from collections import defaultdict, deque
from itertools import groupby
from typing import Dict, List, Set, Tuple

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.graph import Add, Attn, Concatenate, Node
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


# ── Critical Path Analysis ───────────────────────────────────────────


class CriticalPathAnalyzer:
    """Analyzes critical paths through the computation graph.

    The critical path is the longest chain of sequential dependencies.
    This determines the minimum number of transformer layers needed.
    """

    def __init__(self, output_nodes: Set[Node], all_nodes: Set[Node]):
        self._output_nodes = output_nodes
        self._all_nodes = all_nodes

        # Build forward edges: node -> set of consumers
        self._consumers: Dict[Node, Set[Node]] = defaultdict(set)
        for node in all_nodes:
            for inp in node.inputs:
                if inp in all_nodes:
                    self._consumers[inp].add(node)

        # Distance from each node to nearest output (0 for outputs)
        self._dist_to_output: Dict[Node, int] = {}
        self._compute_distances()

        # The maximum distance = critical path length
        self._max_depth = max(self._dist_to_output.values()) if self._dist_to_output else 0

    def _compute_distances(self):
        """BFS from outputs backward to compute distance to output for each node."""
        # Start from output nodes
        queue = deque()
        for node in self._output_nodes:
            self._dist_to_output[node] = 0
            queue.append(node)

        # BFS backward through inputs
        while queue:
            node = queue.popleft()
            dist = self._dist_to_output[node]
            for inp in node.inputs:
                if inp in self._all_nodes and inp not in self._dist_to_output:
                    self._dist_to_output[inp] = dist + 1
                    queue.append(inp)

    def get_max_depth(self) -> int:
        """The length of the critical path (longest input-to-output chain)."""
        return self._max_depth

    def get_depth_histogram(self) -> Dict[int, int]:
        """Count of nodes at each depth level."""
        hist: Dict[int, int] = defaultdict(int)
        for dist in self._dist_to_output.values():
            hist[dist] += 1
        return dict(hist)

    def get_critical_path_nodes(self) -> Set[Node]:
        """All nodes that lie on some critical path."""
        # A node is on a critical path if its depth equals the max depth
        # and there's a path of decreasing depth to an output
        critical: Set[Node] = set()
        for node, dist in self._dist_to_output.items():
            if dist == self._max_depth:
                # This is a critical path start, trace forward
                self._trace_critical_path_forward(node, critical)
        return critical

    def _trace_critical_path_forward(self, node: Node, critical: Set[Node]):
        """Trace from a critical path start to output, marking all nodes."""
        critical.add(node)
        dist = self._dist_to_output[node]
        if dist == 0:
            return
        # Find a consumer with dist = current dist - 1
        for consumer in self._consumers.get(node, set()):
            if self._dist_to_output.get(consumer, -1) == dist - 1:
                self._trace_critical_path_forward(consumer, critical)
                break  # Only need one path

    def trace_critical_paths(self, max_paths: int = 5) -> List[List[Node]]:
        """Trace up to max_paths distinct critical paths.

        Returns list of paths, each path is a list of nodes from input to output.
        """
        # Find all critical path starts (nodes at max depth)
        starts = [n for n, d in self._dist_to_output.items() if d == self._max_depth]

        paths: List[List[Node]] = []
        seen_path_signatures: Set[Tuple[Tuple[str, str | None], ...]] = set()

        for start in starts:
            if len(paths) >= max_paths:
                break
            path = self._trace_one_path(start)
            # Use annotation signature to deduplicate similar paths
            sig = tuple((n.node_type(), n.annotation) for n in path)
            if sig not in seen_path_signatures:
                seen_path_signatures.add(sig)
                paths.append(path)

        return paths

    def _trace_one_path(self, start: Node) -> List[Node]:
        """Trace one critical path from start to output."""
        path = [start]
        node = start
        while True:
            dist = self._dist_to_output[node]
            if dist == 0:
                break
            # Find a consumer with dist - 1 (prefer staying on annotation)
            candidates = [
                c for c in self._consumers.get(node, set())
                if self._dist_to_output.get(c, -1) == dist - 1
            ]
            if not candidates:
                break
            # Prefer same annotation for cleaner traces
            same_ann = [c for c in candidates if c.annotation == node.annotation]
            next_node = same_ann[0] if same_ann else candidates[0]
            path.append(next_node)
            node = next_node
        return path

    def get_annotation_critical_contribution(self) -> Dict[str, int]:
        """Count how many critical path nodes have each annotation."""
        critical = self.get_critical_path_nodes()
        counts: Dict[str, int] = defaultdict(int)
        for node in critical:
            ann = node.annotation or "(none)"
            counts[ann] += 1
        return dict(counts)

    def find_contiguous_chains(self, path: List[Node]) -> List[Tuple[str, int, int, List[Node]]]:
        """Find contiguous runs of the same annotation in a path.

        Returns list of (annotation, start_idx, length, nodes).
        Sorted by length descending.
        """
        if not path:
            return []

        chains = []
        i = 0
        while i < len(path):
            ann = path[i].annotation or "(none)"
            start = i
            while i < len(path) and (path[i].annotation or "(none)") == ann:
                i += 1
            length = i - start
            chains.append((ann, start, length, path[start:i]))

        # Sort by length descending
        chains.sort(key=lambda x: -x[2])
        return chains


def _count_real_ops(nodes: List[Node]) -> int:
    """Count nodes that consume layer capacity (exclude Concatenate)."""
    return sum(1 for n in nodes if not isinstance(n, Concatenate))


def _real_ops_type_summary(nodes: List[Node]) -> str:
    """Summarize real op types (excluding Concatenate)."""
    real = [n for n in nodes if not isinstance(n, Concatenate)]
    if not real:
        return "(none)"

    condensed = []
    for t, group in groupby(n.node_type() for n in real):
        count = len(list(group))
        if count > 1:
            condensed.append(f"{t}×{count}")
        else:
            condensed.append(t)

    if len(condensed) > 6:
        return " → ".join(condensed[:3]) + " → ... → " + " → ".join(condensed[-2:])
    return " → ".join(condensed)


def _print_critical_path_analysis(all_nodes: Set[Node], output_nodes: Set[Node]):
    """Print critical path analysis."""
    analyzer = CriticalPathAnalyzer(output_nodes, all_nodes)

    max_depth = analyzer.get_max_depth()
    print(f"\n{'─' * 72}")
    print(f"  CRITICAL PATH ANALYSIS")
    print(f"{'─' * 72}")
    print(f"\n  Critical path length: {max_depth} sequential ops")
    print(f"  (This determines the minimum transformer layers needed)")

    # Trace paths first so we can use them for the optimization targets
    paths = analyzer.trace_critical_paths(max_paths=3)

    # Depth histogram (condensed)
    hist = analyzer.get_depth_histogram()
    depths = sorted(hist.keys())
    if depths:
        # Show a condensed sparkline-style histogram
        max_count = max(hist.values())
        bars = []
        for d in range(0, max(depths) + 1, max(1, len(depths) // 20)):
            # Average count in this bucket
            bucket = [hist.get(i, 0) for i in range(d, min(d + max(1, len(depths) // 20), max(depths) + 1))]
            avg = sum(bucket) / len(bucket) if bucket else 0
            level = int(8 * avg / max_count) if max_count > 0 else 0
            bars.append("▁▂▃▄▅▆▇█"[min(level, 7)])
        print(f"\n  Depth distribution: [{''.join(bars)}] 0→{max(depths)}")

    # Show one representative critical path breakdown
    if paths:
        path = paths[0]
        real_ops = _count_real_ops(path)
        print(f"\n  One critical path: {real_ops} real ops (excluding Concatenate)")

        # Break down by annotation for this path
        chains = analyzer.find_contiguous_chains(path)
        print(f"\n  {'Annotation':<35s} {'Ops':>8s} {'%':>8s}")
        print(f"  {'─' * 35} {'─' * 8} {'─' * 8}")

        for ann, start, length, nodes in sorted(chains, key=lambda x: -_count_real_ops(x[3])):
            real_count = _count_real_ops(nodes)
            if real_count > 0:
                pct = 100.0 * real_count / real_ops if real_ops else 0
                print(f"  {ann:<35s} {real_count:>8d} {pct:>7.1f}%")

    # Optimization targets - find the longest contiguous chains across all paths
    print(f"\n  {'─' * 68}")
    print(f"  OPTIMIZATION TARGETS (longest contiguous chains)")
    print(f"  {'─' * 68}")
    print(f"  Contiguous chains of the same annotation are often single logical")
    print(f"  operations that could potentially be restructured for parallelism.")

    all_chains = []
    for path in paths:
        chains = analyzer.find_contiguous_chains(path)
        for ann, start, length, nodes in chains:
            real_count = _count_real_ops(nodes)
            if real_count >= 3:  # Only show chains with 3+ real ops
                all_chains.append((ann, real_count, nodes))

    # Deduplicate and sort by real op count
    seen = set()
    unique_chains = []
    for ann, real_count, nodes in sorted(all_chains, key=lambda x: -x[1]):
        key = (ann, real_count, tuple(n.node_type() for n in nodes))
        if key not in seen:
            seen.add(key)
            unique_chains.append((ann, real_count, nodes))

    if unique_chains:
        print(f"\n  {'Annotation':<30s} {'Ops':>6s}  {'Pattern':<30s}")
        print(f"  {'─' * 30} {'─' * 6}  {'─' * 30}")
        for ann, real_count, nodes in unique_chains[:10]:  # Top 10
            pattern = _real_ops_type_summary(nodes)
            if len(pattern) > 30:
                pattern = pattern[:27] + "..."
            print(f"  {ann:<30s} {real_count:>6d}  {pattern:<30s}")
    else:
        print(f"\n  No contiguous chains with 3+ ops found.")

    # Detailed path trace (optional, show first path only)
    if paths:
        print(f"\n  Example critical path trace:")
        path = paths[0]
        real_ops_count = _count_real_ops(path)
        print(f"  ({len(path)} nodes, {real_ops_count} real ops)")

        chains = analyzer.find_contiguous_chains(path)
        print()
        for ann, start, length, nodes in sorted(chains, key=lambda x: x[1]):
            real_count = _count_real_ops(nodes)
            pattern = _real_ops_type_summary(nodes)
            print(f"    [{ann}] {real_count} ops: {pattern}")

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

    graph_io, pos = build_game_graph(
        config, textures, max_walls=max_walls, max_coord=max_coord,
        chunk_size=args.chunk_size,
    )
    output = graph_io.concat_output()

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

    # Critical path analysis
    _print_critical_path_analysis(all_nodes, {output, pos})


if __name__ == "__main__":
    main()
