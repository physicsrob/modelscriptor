"""Compare what graph annotations land on which layers under default vs legacy.

Goal: identify which annotation chains stretch in default but compress in
legacy. The tail of default has 23 layers of 1-13 ops/layer with 1-3 attn
heads used — far below capacity in every dimension. That's dependency-bound
serialization. This script identifies which subgraph is doing it.
"""

from collections import Counter, defaultdict
from typing import Dict

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


def compile_with_tracking(output, pos, d, d_head, d_hidden, policy):
    node_to_layer: Dict[int, int] = {}

    def _track(node, layer_idx):
        node_to_layer[node.node_id] = layer_idx

    net = forward_compile(
        d,
        d_head,
        output,
        pos,
        verbose=False,
        max_layers=400,
        d_hidden=d_hidden,
        device=None,
        on_node_scheduled=_track,
        policy=policy,
    )
    n_layers = len(net.layers)
    del net
    return node_to_layer, n_layers


def main():
    segments, textures = box_room_textured(wad_path="doom1.wad", tex_size=64)
    config = RenderConfig(
        screen_width=120,
        screen_height=100,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    max_walls = max(8, len(segments))

    print("Building headless graph...", flush=True)
    graph_io, pos = build_game_graph(
        config,
        textures,
        max_walls=max_walls,
        max_coord=10.0,
        chunk_size=20,
        render_pixels=False,
    )
    output = graph_io.concat_output()
    all_nodes = get_ancestor_nodes({output, pos})

    tex_w = textures[0].shape[0]
    min_d_head = compute_min_d_head(max_walls, tex_w)
    d_head = 1
    while d_head < min_d_head:
        d_head *= 2

    d, d_hidden = 3072, 8192

    print(f"Compiling default (None policy) at d={d}, d_hidden={d_hidden}...", flush=True)
    n2l_default, nl_default = compile_with_tracking(output, pos, d, d_head, d_hidden, None)
    print(f"  {nl_default} layers", flush=True)

    print(f"Compiling legacy (LEGACY_POLICY)...", flush=True)
    n2l_legacy, nl_legacy = compile_with_tracking(output, pos, d, d_head, d_hidden, LEGACY_POLICY)
    print(f"  {nl_legacy} layers", flush=True)

    # Per-annotation last-layer (max layer index where any node of that
    # annotation lands).  This shows which annotation chains stretch out.
    by_id = {n.node_id: n for n in all_nodes}

    def per_ann_layer_range(n2l):
        ranges = defaultdict(lambda: [None, None, 0])  # [min, max, count]
        for nid, layer in n2l.items():
            node = by_id.get(nid)
            if node is None:
                continue
            ann = node.annotation or "(none)"
            r = ranges[ann]
            if r[0] is None or layer < r[0]:
                r[0] = layer
            if r[1] is None or layer > r[1]:
                r[1] = layer
            r[2] += 1
        return ranges

    rng_default = per_ann_layer_range(n2l_default)
    rng_legacy = per_ann_layer_range(n2l_legacy)
    all_anns = sorted(set(rng_default) | set(rng_legacy))

    print()
    print("=" * 100)
    print("PER-ANNOTATION LAYER RANGE  (sorted by default's max layer descending)")
    print("=" * 100)
    print(
        f"  {'Annotation':<42s}  "
        f"{'def_min':>7} {'def_max':>7} {'def_n':>5}  "
        f"{'leg_min':>7} {'leg_max':>7} {'leg_n':>5}  "
        f"{'Δmax':>5}"
    )
    print("  " + "-" * 95)
    rows = []
    for ann in all_anns:
        d_min, d_max, d_n = rng_default.get(ann, (None, None, 0))
        l_min, l_max, l_n = rng_legacy.get(ann, (None, None, 0))
        rows.append((ann, d_min, d_max, d_n, l_min, l_max, l_n))
    rows.sort(key=lambda r: -(r[2] if r[2] is not None else -1))
    for ann, d_min, d_max, d_n, l_min, l_max, l_n in rows:
        d_max_v = d_max if d_max is not None else -1
        l_max_v = l_max if l_max is not None else -1
        delta = d_max_v - l_max_v
        delta_str = f"{delta:+d}" if d_max is not None and l_max is not None else "—"
        print(
            f"  {ann:<42s}  "
            f"{str(d_min):>7} {str(d_max):>7} {d_n:>5}  "
            f"{str(l_min):>7} {str(l_max):>7} {l_n:>5}  "
            f"{delta_str:>5}"
        )

    # Tail-layer occupancy by annotation.  For each policy, list the
    # annotations whose nodes land in the LAST 5 layers.
    print()
    print("=" * 100)
    print("WHAT'S IN THE TAIL — annotations contributing nodes in last 5 layers")
    print("=" * 100)

    def tail_anns(n2l, n_layers, k=5):
        c = Counter()
        for nid, layer in n2l.items():
            if layer >= n_layers - k:
                node = by_id.get(nid)
                ann = node.annotation if node else None
                c[ann or "(none)"] += 1
        return c

    print(
        f"\n  Default last {min(5, nl_default)} layers (layers {nl_default - 5}–{nl_default - 1}):"
    )
    for ann, count in tail_anns(n2l_default, nl_default).most_common(15):
        print(f"    {ann:<45s} {count:>4d} nodes")

    print(
        f"\n  Legacy last {min(5, nl_legacy)} layers (layers {nl_legacy - 5}–{nl_legacy - 1}):"
    )
    for ann, count in tail_anns(n2l_legacy, nl_legacy).most_common(15):
        print(f"    {ann:<45s} {count:>4d} nodes")


if __name__ == "__main__":
    main()
