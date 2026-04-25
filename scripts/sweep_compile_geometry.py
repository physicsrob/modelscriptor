"""Sweep d_model x d_hidden x scheduler-policy on the headless DOOM graph.

Builds the game graph once with ``render_pixels=False`` (saves the
~30s thinking_wall build), then runs ``forward_compile`` for each
combination and reports compiled layer count, head pruning, and
layer-capacity utilization.

Usage:
    uv run python -m scripts.sweep_compile_geometry

Results print incrementally and are summarized as a table at the end.
"""

import time
from typing import Dict, List, Optional, Tuple

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


D_MODELS = [2048, 2560, 3072]
D_HIDDENS = [2048, 4096, 8192]
POLICIES: Dict[str, Optional[SchedulingPolicy]] = {
    "default": None,
    "legacy": LEGACY_POLICY,
}


def main():
    # Build graph once.
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

    print(
        f"Building graph: render_pixels=False, max_walls={max_walls}, "
        f"chunk_size=20",
        flush=True,
    )
    t0 = time.perf_counter()
    graph_io, pos = build_game_graph(
        config,
        textures,
        max_walls=max_walls,
        max_coord=10.0,
        chunk_size=20,
        render_pixels=False,
    )
    output = graph_io.concat_output()
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    tex_w = textures[0].shape[0]
    min_d_head = compute_min_d_head(max_walls, tex_w)
    d_head = 1
    while d_head < min_d_head:
        d_head *= 2
    print(f"  d_head = {d_head} (min={min_d_head})", flush=True)
    print()

    rows: List[Tuple[int, int, str, Optional[int], Optional[float], Optional[float], float]] = []

    for d in D_MODELS:
        for d_hidden in D_HIDDENS:
            for pname, policy in POLICIES.items():
                node_to_layer: Dict[int, int] = {}

                def _track(node, layer_idx):
                    node_to_layer[node.node_id] = layer_idx

                t1 = time.perf_counter()
                try:
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
                    n_layers = (
                        max(node_to_layer.values()) + 1 if node_to_layer else 0
                    )
                    max_heads_per_layer = d // d_head
                    total_heads = sum(l.attn.attn.n_heads for l in net.layers)
                    head_pct = (
                        100.0 * total_heads / (max_heads_per_layer * n_layers)
                        if n_layers
                        else 0.0
                    )
                    layer_capacity = (
                        4 * d * d + 2 * d * d_hidden + d_hidden + d
                    )
                    attn_params = sum(l.attn.attn.num_params() for l in net.layers)
                    # Rough utilization: attn + MLP weights vs total capacity.
                    # The MLP weights aren't pruned, so they count fully.
                    mlp_params_per_layer = 2 * d * d_hidden + d_hidden + d
                    total_alloc = attn_params + mlp_params_per_layer * n_layers
                    util_pct = (
                        100.0 * total_alloc / (layer_capacity * n_layers)
                        if n_layers
                        else 0.0
                    )
                    elapsed = time.perf_counter() - t1
                    rows.append(
                        (d, d_hidden, pname, n_layers, head_pct, util_pct, elapsed)
                    )
                    print(
                        f"  d={d:>4} d_hidden={d_hidden:>4} policy={pname:<16}  "
                        f"layers={n_layers:>3}  heads={head_pct:>5.1f}%  "
                        f"util={util_pct:>5.1f}%  ({elapsed:>5.1f}s)",
                        flush=True,
                    )
                    del net
                except Exception as e:
                    elapsed = time.perf_counter() - t1
                    rows.append((d, d_hidden, pname, None, None, None, elapsed))
                    print(
                        f"  d={d:>4} d_hidden={d_hidden:>4} policy={pname:<16}  "
                        f"FAILED ({elapsed:.1f}s): {type(e).__name__}: {e}",
                        flush=True,
                    )

    # ---- Summary tables ----
    print()
    print("=" * 78)
    print("LAYER COUNT")
    print("=" * 78)
    _print_layer_table(rows)

    print()
    print("=" * 78)
    print("HEAD PRUNING (% of max heads used)")
    print("=" * 78)
    _print_metric_table(rows, lambda r: r[4], unit="%")

    print()
    print("=" * 78)
    print("LAYER UTILIZATION (% of layer capacity)")
    print("=" * 78)
    _print_metric_table(rows, lambda r: r[5], unit="%")


def _print_layer_table(rows):
    """One row per (d, d_hidden), one column per policy."""
    pnames = list(POLICIES.keys())
    print(
        f"  {'d_model':>7} {'d_hidden':>9} | "
        + " ".join(f"{p:>16}" for p in pnames)
    )
    print(f"  {'-' * 7} {'-' * 9} | " + " ".join(f"{'-' * 16}" for _ in pnames))
    for d in D_MODELS:
        for dh in D_HIDDENS:
            cells = []
            for p in pnames:
                row = next(
                    (r for r in rows if r[0] == d and r[1] == dh and r[2] == p),
                    None,
                )
                if row is None or row[3] is None:
                    cells.append(f"{'FAIL':>16}")
                else:
                    cells.append(f"{row[3]:>16d}")
            print(f"  {d:>7} {dh:>9} | " + " ".join(cells))


def _print_metric_table(rows, getter, unit=""):
    pnames = list(POLICIES.keys())
    print(
        f"  {'d_model':>7} {'d_hidden':>9} | "
        + " ".join(f"{p:>16}" for p in pnames)
    )
    print(f"  {'-' * 7} {'-' * 9} | " + " ".join(f"{'-' * 16}" for _ in pnames))
    for d in D_MODELS:
        for dh in D_HIDDENS:
            cells = []
            for p in pnames:
                row = next(
                    (r for r in rows if r[0] == d and r[1] == dh and r[2] == p),
                    None,
                )
                val = getter(row) if row is not None else None
                if val is None:
                    cells.append(f"{'—':>16}")
                else:
                    cells.append(f"{val:>14.1f}{unit:<2}")
            print(f"  {d:>7} {dh:>9} | " + " ".join(cells))


if __name__ == "__main__":
    main()
