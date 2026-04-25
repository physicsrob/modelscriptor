"""Count standalone Linears in the headless DOOM graph and compare
attention-routing vs MLP-bypass cost.

Standalone Linear = Linear that's NOT part of a Linear→ReLU→Linear MLP chain.
These are the ops affected by the ``local_in_attention`` policy lever.

For each Linear:
- Attention cost: ceil(d_input / d_head) heads.
- MLP bypass cost: 2 * d_output slots.

Reports total cost across the graph at d_head=128 (matching the headless
sweep) and the bottleneck distribution.
"""

from collections import Counter, defaultdict

from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph.linear import Linear
from torchwright.graph.relu import ReLU
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


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
    print(f"  {len(all_nodes)} nodes, d_head = {d_head}", flush=True)

    # Identify Linears that are part of Linear→ReLU→Linear chains.
    # Linear1 feeds a ReLU; Linear2 reads from a ReLU.
    consumers = defaultdict(set)
    for node in all_nodes:
        for inp in node.inputs:
            consumers[inp.node_id].add(node)

    chain_linears = set()
    for node in all_nodes:
        if isinstance(node, ReLU):
            if node.inputs and isinstance(node.inputs[0], Linear):
                chain_linears.add(node.inputs[0].node_id)
            for c in consumers.get(node.node_id, set()):
                if isinstance(c, Linear):
                    chain_linears.add(c.node_id)

    standalone = [
        n
        for n in all_nodes
        if isinstance(n, Linear) and n.node_id not in chain_linears
    ]

    print(f"\nStandalone Linears: {len(standalone)}")

    # Bucket by (d_input, d_output)
    shapes = Counter()
    for n in standalone:
        shapes[(len(n.inputs[0]), n.d_output)] += 1

    # Aggregate cost
    total_attn_heads = sum(
        (len(n.inputs[0]) + d_head - 1) // d_head for n in standalone
    )
    total_mlp_slots = sum(2 * n.d_output for n in standalone)

    print(f"\n  Total attention heads (legacy):  {total_attn_heads}")
    print(f"  Total MLP slots (default bypass): {total_mlp_slots}")
    print(
        f"  Ratio MLP/attn: {total_mlp_slots / max(total_attn_heads, 1):.1f}x more slots than heads"
    )

    print(f"\n  At d_head=128, d_hidden=2048, n_heads = d/d_head:")
    for d_model in (2048, 2560, 3072):
        for d_hidden in (2048, 4096, 8192):
            n_heads_per_layer = d_model // d_head
            attn_layers = (total_attn_heads + n_heads_per_layer - 1) // n_heads_per_layer
            mlp_layers = (total_mlp_slots + d_hidden - 1) // d_hidden
            print(
                f"    d={d_model:>4} d_hidden={d_hidden:>4}: "
                f"attn-routing fits in {attn_layers} layers, "
                f"MLP-routing fits in {mlp_layers} layers (per-resource lower bound)"
            )

    # Top shapes by aggregate cost
    print(f"\nTop shapes by attention head cost:")
    by_attn_cost = sorted(
        shapes.items(),
        key=lambda kv: -((kv[0][0] + d_head - 1) // d_head) * kv[1],
    )
    print(f"  {'d_input':>8} {'d_output':>8} {'count':>6} {'attn_heads':>11} {'mlp_slots':>10}")
    for (d_in, d_out), count in by_attn_cost[:15]:
        attn = ((d_in + d_head - 1) // d_head) * count
        mlp = 2 * d_out * count
        print(f"  {d_in:>8} {d_out:>8} {count:>6} {attn:>11} {mlp:>10}")

    print(f"\nTop shapes by MLP slot cost:")
    by_mlp_cost = sorted(shapes.items(), key=lambda kv: -2 * kv[0][1] * kv[1])
    print(f"  {'d_input':>8} {'d_output':>8} {'count':>6} {'attn_heads':>11} {'mlp_slots':>10}")
    for (d_in, d_out), count in by_mlp_cost[:15]:
        attn = ((d_in + d_head - 1) // d_head) * count
        mlp = 2 * d_out * count
        print(f"  {d_in:>8} {d_out:>8} {count:>6} {attn:>11} {mlp:>10}")


if __name__ == "__main__":
    main()
