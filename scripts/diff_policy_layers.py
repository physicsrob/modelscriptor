"""Per-layer diff of default vs legacy scheduling policy.

For each policy, capture per-layer:
- Attn ops by type (compute_attn / compute_linear / compute_add / cancel / add_into)
- Heads used (vs n_heads_per_layer max)
- MLP slots used (vs d_hidden)
- Residual occupancy after the layer

Then print side-by-side and flag layers where one policy is densely packed
while the other is not (the source of the layer-count delta).
"""

import io
import contextlib
from typing import Dict, List, Tuple

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.scheduling_policy import LEGACY_POLICY
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


def compile_capture(output, pos, d, d_head, d_hidden, policy):
    """Run forward_compile with verbose=True and capture stdout into a list."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net = forward_compile(
            d,
            d_head,
            output,
            pos,
            verbose=True,
            max_layers=400,
            d_hidden=d_hidden,
            device=None,
            policy=policy,
        )
    n_layers = len(net.layers)

    # Per-layer head pruning (post-compile pruning is reflected in layer.attn.attn.n_heads)
    n_heads_per_layer: List[int] = []
    for layer in net.layers:
        n_heads_per_layer.append(layer.attn.attn.n_heads)

    del net
    return buf.getvalue(), n_layers, n_heads_per_layer


def parse_verbose(log: str) -> List[Dict]:
    """Parse the verbose per-layer log into structured records.

    Format example (one line per layer):
      0       12 ops  123/456 (5.0%)  100/2048 ( 5%)   234/2048 (11%)  MLP  500/2048   ...

    The three "X/Y (Z%)" triples are: params, occupied_before, occupied_after.
    """
    rows = []
    for line in log.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        try:
            layer = int(toks[0])
        except ValueError:
            continue
        if toks[2] != "ops":
            continue
        n_ops = int(toks[1])
        try:
            mlp_idx = toks.index("MLP")
            mlp_used, mlp_total = toks[mlp_idx + 1].split("/")
            mlp_used = int(mlp_used)
            mlp_total = int(mlp_total)
        except (ValueError, IndexError):
            mlp_used = mlp_total = 0
        # Three slash pairs in order: layer_params, occupied_before,
        # occupied_after. Index 3 is the params slash; indices 5 and 7
        # are the occupancy slashes (count on tokens 'X/Y' between '(' tokens).
        slash_pairs = [
            t for t in toks if "/" in t and t.replace(",", "").replace("/", "").isdigit()
        ]
        # Drop the MLP one — it's the last
        slash_pairs = [p for p in slash_pairs if p != f"{mlp_used}/{mlp_total}"]
        occ_before = occ_after = 0
        if len(slash_pairs) >= 3:
            try:
                occ_before = int(slash_pairs[1].split("/")[0])
                occ_after = int(slash_pairs[2].split("/")[0])
            except ValueError:
                pass
        rows.append(
            {
                "layer": layer,
                "n_ops": n_ops,
                "mlp_slots": mlp_used,
                "mlp_total": mlp_total,
                "occ_before": occ_before,
                "occ_after": occ_after,
            }
        )
    return rows


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

    tex_w = textures[0].shape[0]
    min_d_head = compute_min_d_head(max_walls, tex_w)
    d_head = 1
    while d_head < min_d_head:
        d_head *= 2

    d, d_hidden = 3072, 8192
    n_heads_max = d // d_head
    print(
        f"Compiling at d={d}, d_head={d_head}, d_hidden={d_hidden} "
        f"(max heads/layer = {n_heads_max})",
        flush=True,
    )

    print("\n--- DEFAULT policy ---", flush=True)
    log_default, nl_default, heads_default = compile_capture(
        output, pos, d, d_head, d_hidden, None
    )
    rows_default = parse_verbose(log_default)
    print(f"  {nl_default} layers", flush=True)

    print("\n--- LEGACY policy ---", flush=True)
    log_legacy, nl_legacy, heads_legacy = compile_capture(
        output, pos, d, d_head, d_hidden, LEGACY_POLICY
    )
    rows_legacy = parse_verbose(log_legacy)
    print(f"  {nl_legacy} layers", flush=True)

    print("\n" + "=" * 90)
    print(f"PER-LAYER COMPARISON (d={d}, d_hidden={d_hidden})")
    print("=" * 90)
    print(
        f"  {'L':>3} | "
        f"{'def_ops':>7} {'def_h':>5} {'def_mlp':>9} {'def_occ':>9} | "
        f"{'leg_ops':>7} {'leg_h':>5} {'leg_mlp':>9} {'leg_occ':>9}"
    )
    print(f"  {'-' * 3} | " + ("-" * 33 + " | ") * 2)
    n_max = max(nl_default, nl_legacy)
    for i in range(n_max):
        d_row = rows_default[i] if i < len(rows_default) else None
        l_row = rows_legacy[i] if i < len(rows_legacy) else None
        d_h = heads_default[i] if i < len(heads_default) else None
        l_h = heads_legacy[i] if i < len(heads_legacy) else None

        def _cell(row, h):
            if row is None:
                return f"{'—':>7} {'—':>5} {'—':>9} {'—':>9}"
            ops = f"{row['n_ops']:>7d}"
            heads = f"{h}/{n_heads_max}" if h is not None else "—"
            mlp = (
                f"{row['mlp_slots']}/{row['mlp_total']}"
                if row["mlp_total"]
                else "—"
            )
            occ = f"{row['occ_after']}/{d}"
            return f"{ops:>7} {heads:>5} {mlp:>9} {occ:>9}"

        print(f"  {i:>3} | {_cell(d_row, d_h)} | {_cell(l_row, l_h)}")

    # Summary stats
    print("\n--- AGGREGATES ---")
    def_total_ops = sum(r["n_ops"] for r in rows_default)
    leg_total_ops = sum(r["n_ops"] for r in rows_legacy)
    def_total_mlp = sum(r["mlp_slots"] for r in rows_default)
    leg_total_mlp = sum(r["mlp_slots"] for r in rows_legacy)
    def_total_heads = sum(heads_default)
    leg_total_heads = sum(heads_legacy)
    print(
        f"  Total ops:    default={def_total_ops}  legacy={leg_total_ops}  "
        f"diff={def_total_ops - leg_total_ops:+d}"
    )
    print(
        f"  Total MLP slots: default={def_total_mlp}  legacy={leg_total_mlp}  "
        f"diff={def_total_mlp - leg_total_mlp:+d}"
    )
    print(
        f"  Total heads (post-prune): default={def_total_heads}  "
        f"legacy={leg_total_heads}  diff={def_total_heads - leg_total_heads:+d}"
    )
    print(
        f"  Avg ops/layer:  default={def_total_ops / nl_default:.1f}  "
        f"legacy={leg_total_ops / nl_legacy:.1f}"
    )


if __name__ == "__main__":
    main()
