"""Probe script for the CP-SAT scheduler in
`torchwright.compiler.forward.cpsat_scheduler`.

Synthetic sanity check + headless-DOOM Pareto sweep. Use this to
explore the (layer count, attention head count) Pareto front under
different `Costs` weights before committing to a configuration in
`forward_compile`.

Run:
    # Synthetic sanity check (~1s, local)
    uv run python -m scripts.cpsat_schedule --synthetic

    # Full DOOM headless run with default Pareto sweep (heavy, can run
    # local on a CPU-only host — no GPU needed for the solver)
    uv run python -m scripts.cpsat_schedule

    # Custom Pareto sweep
    uv run python -m scripts.cpsat_schedule --pareto-betas 0,5,50,500
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import (
    ATTN,
    MLP,
    Costs,
    GraphModel,
    ScheduleAssignment,
    SolveStats,
    solve_schedule,
    build_graph_model,
    heads_for,
    is_flex,
    routing,
    slots_for,
    uses_residual,
)
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
from torchwright.graph import Add, InputNode, Linear, Node
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


# ---------------------------------------------------------------------------
# Heuristic baseline (for warm-starting CP-SAT)
# ---------------------------------------------------------------------------


@dataclass
class HintSchedule:
    """Heuristic baseline schedule. Maps node_id -> layer."""

    node_to_layer: Dict[int, int]
    n_layers: int


def baseline_hint(
    output_node: Node,
    pos_encoding: PosEncoding,
    d: int,
    d_head: int,
    d_hidden: int,
    policy: SchedulingPolicy,
) -> HintSchedule:
    """Run `forward_compile` under `policy` and return its node->layer map.

    Used as a warm-start hint for CP-SAT so the solver starts from a
    known feasible incumbent rather than building one from scratch.

    Lives in the script (not the production module) because it
    imports `forward_compile`, which would create a cycle once
    `forward_compile` itself imports `cpsat_scheduler` (Stage 2).
    """
    node_to_layer: Dict[int, int] = {}

    def _track(node: Node, layer_idx: int) -> None:
        node_to_layer[node.node_id] = layer_idx

    net = forward_compile(
        d,
        d_head,
        output_node,
        pos_encoding,
        verbose=False,
        max_layers=400,
        d_hidden=d_hidden,
        device=None,
        on_node_scheduled=_track,
        policy=policy,
    )
    n_layers = len(net.layers)
    del net
    return HintSchedule(node_to_layer=node_to_layer, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Synthetic sanity check
# ---------------------------------------------------------------------------


def synthetic_check() -> None:
    """A small, hand-checkable graph: an L->R->L chain on InputNode `in1`,
    then an Add with another InputNode `in2`, then output via Linear.

    With d=256, d_head=16, d_hidden=64: chain at layer 0, Add at
    layer 1, output Linear at layer 2 — 3 layers total. The chain
    dependency serializes them; nothing else fits in parallel because
    the only other schedulable nodes ARE this chain. So the optimum
    is dictated by the critical-path length and we can verify it.

    d=256 leaves 256 - 32(in1) - 32(in2) - 16(pos_encoding) = 176 cols
    of headroom for L2/s/out, all 32 wide — comfortable.
    """
    print("=" * 70)
    print("SYNTHETIC SANITY CHECK")
    print("=" * 70)
    from torchwright.graph.session import fresh_graph_session

    with fresh_graph_session():
        d_in = 32
        d_hidden = 32
        in1 = InputNode(d_in, name="in1", value_range=(-1.0, 1.0))
        in2 = InputNode(d_in, name="in2", value_range=(-1.0, 1.0))
        W1 = torch.randn(d_in, d_hidden) * 0.1
        l1 = Linear(in1, W1)
        relu = ReLU(l1)
        W2 = torch.randn(d_hidden, d_in) * 0.1
        l2 = Linear(relu, W2)
        s = Add(l2, in2)
        Wout = torch.randn(d_in, d_in) * 0.1
        out = Linear(s, Wout)
        pos = PosEncoding(d_pos=16)
        gm = build_graph_model(out, pos)

    print(
        f"  Schedulable: {len(gm.schedulable)}, "
        f"chains: {len(gm.chains)}, "
        f"edges: {len(gm.edges)}"
    )
    for c in gm.chains:
        print(
            f"    chain c{c.chain_id}: L1={c.l1.node_id} R={c.relu.node_id} "
            f"L2={c.l2.node_id} exclusive={c.exclusive} width={c.width}"
        )

    assignment, stats = solve_schedule(
        out,
        pos,
        d=256,
        d_head=16,
        d_hidden=64,
        costs=Costs(alpha=1, beta=0),
        flex_routing=False,
        time_budget_s=30.0,
        max_layers=10,
        policy=LEGACY_POLICY,
        log_search_progress=False,
    )
    print(
        f"  layer-min: status={stats.status_name}, "
        f"n_layers={assignment.n_layers if assignment else '-'}, "
        f"heads={stats.total_attn_heads}, "
        f"wall={stats.wall_time_s:.2f}s"
    )
    assert assignment is not None, f"expected an assignment; status={stats.status_name}"
    print("  per-node layer:")
    for n in gm.schedulable:
        L = assignment.node_to_layer.get(n.node_id, "?")
        ann = type(n).__name__
        print(f"    n{n.node_id:>3} {ann:<14} d={len(n):>3}  layer={L}")

    assert stats.is_optimal, f"expected OPTIMAL, got {stats.status_name}"
    # Chain must run before the Add (L2 -> s) and Add before output
    # (s -> out). That's a strict 3-layer chain in the dependency
    # DAG (chain=L0, Add=L1, output=L2). The optimum is exactly 3.
    assert assignment.n_layers == 3, (
        f"expected 3 layers, got {assignment.n_layers}"
    )
    print("  PASS")
    print()


# ---------------------------------------------------------------------------
# Headless-DOOM probe
# ---------------------------------------------------------------------------


def build_doom_model() -> Tuple[GraphModel, int, int, int]:
    """Build the headless DOOM graph and the standard solver geometry
    (`d=3072, d_head=128, d_hidden=8192`).
    """
    print("Building headless DOOM graph...", flush=True)
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
    d = 3072
    d_hidden = 8192

    gm = build_graph_model(output, pos)
    print(
        f"  schedulable: {len(gm.schedulable)}, "
        f"chains: {len(gm.chains)}, edges: {len(gm.edges)}",
        flush=True,
    )
    print(
        f"  inputs (pre-allocated): {len(gm.input_nodes)} nodes, "
        f"{sum(len(n) for n in gm.input_nodes)} cols",
        flush=True,
    )
    print(f"  d={d}, d_head={d_head}, d_hidden={d_hidden}", flush=True)
    print(f"  max heads per layer = {d // d_head}", flush=True)

    return gm, d_head, d, d_hidden


def doom_run(
    time_budget_s: float,
    max_layers: int,
    pareto_betas: Optional[List[int]] = None,
) -> None:
    print("=" * 70)
    print("FULL DOOM HEADLESS RUN")
    print("=" * 70)

    gm, d_head, d, d_hidden = build_doom_model()

    # Baseline hints from the heuristic (used to warm-start CP-SAT).
    print()
    print("Baseline (heuristic) hints:", flush=True)
    t0 = time.perf_counter()
    legacy_hint = baseline_hint(
        gm.output_node, gm.pos_encoding, d, d_head, d_hidden, LEGACY_POLICY
    )
    print(
        f"  legacy: {legacy_hint.n_layers} layers "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    t0 = time.perf_counter()
    default_hint = baseline_hint(
        gm.output_node, gm.pos_encoding, d, d_head, d_hidden, SchedulingPolicy()
    )
    print(
        f"  default: {default_hint.n_layers} layers "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    # ---- Static-routing layer-min (one per policy) ----
    static_results = []
    for policy_name, policy, hint in [
        ("legacy (local=always)", LEGACY_POLICY, legacy_hint),
        ("default (local=never)", SchedulingPolicy(), default_hint),
    ]:
        print()
        print(
            f"--- Layer-min CP-SAT, policy={policy_name} (static routing) ---",
            flush=True,
        )
        assignment, stats = solve_schedule(
            gm.output_node,
            gm.pos_encoding,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
            costs=Costs(alpha=1, beta=0),
            flex_routing=False,
            time_budget_s=time_budget_s,
            max_layers=max_layers,
            hint_layers=hint.node_to_layer,
            policy=policy,
            log_search_progress=False,
        )
        n_layers = assignment.n_layers if assignment else -1
        print(
            f"  status: {stats.status_name}, "
            f"n_layers={n_layers}, "
            f"total_attn_heads={stats.total_attn_heads}, "
            f"wall={stats.wall_time_s:.1f}s",
            flush=True,
        )
        print(f"  vs. heuristic: {hint.n_layers} layers", flush=True)
        static_results.append((policy_name, assignment, stats, hint.n_layers))

    # ---- Flex-routing Pareto sweep ----
    if pareto_betas is None:
        pareto_betas = [0, 1, 10, 100, 1000]

    print()
    print("=" * 70)
    print("FLEX-ROUTING PARETO SWEEP")
    print("Objective: alpha=1 * n_layers + beta * total_attn_heads")
    print("=" * 70)

    pareto_results: List[
        Tuple[int, Optional[ScheduleAssignment], SolveStats]
    ] = []
    flex_hint = (
        legacy_hint
        if legacy_hint.n_layers <= default_hint.n_layers
        else default_hint
    )
    for beta in pareto_betas:
        print()
        print(f"--- beta={beta} (flex routing) ---", flush=True)
        assignment, stats = solve_schedule(
            gm.output_node,
            gm.pos_encoding,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
            costs=Costs(alpha=1, beta=beta),
            flex_routing=True,
            time_budget_s=time_budget_s,
            max_layers=max_layers,
            hint_layers=flex_hint.node_to_layer,
            log_search_progress=False,
        )
        n_layers = assignment.n_layers if assignment else -1
        print(
            f"  status: {stats.status_name}, "
            f"n_layers={n_layers}, "
            f"total_attn_heads={stats.total_attn_heads}, "
            f"wall={stats.wall_time_s:.1f}s",
            flush=True,
        )
        pareto_results.append((beta, assignment, stats))

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Static routing (per heuristic policy):")
    print(
        f"  {'policy':<24} {'heuristic':>10} {'CP-SAT LB':>10} "
        f"{'attn heads':>11}"
    )
    for name, assignment, stats, h_layers in static_results:
        n_layers = assignment.n_layers if assignment else -1
        print(
            f"  {name:<24} {h_layers:>10} {n_layers:>10} "
            f"{stats.total_attn_heads:>11}"
        )
    print()
    print("Flex routing Pareto sweep (CP-SAT picks per-Linear routing):")
    print(
        f"  {'beta':>6} {'n_layers':>10} {'attn_heads':>11} "
        f"{'mlp_bypass':>10}"
    )
    for beta, assignment, stats in pareto_results:
        n_layers = assignment.n_layers if assignment else -1
        opt_marker = "" if stats.is_optimal else " (FEASIBLE)"
        print(
            f"  {beta:>6} {n_layers:>10} {stats.total_attn_heads:>11} "
            f"{stats.total_mlp_bypass_slots:>10}{opt_marker}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the small synthetic-graph sanity check only.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=300.0,
        help="Per-solve time budget in seconds (default 300).",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=60,
        help="Search horizon in layers (default 60).",
    )
    parser.add_argument(
        "--pareto-betas",
        type=str,
        default=None,
        help="Comma-separated beta values for the Pareto sweep "
        "(default: 0,1,10,100,1000).",
    )
    args = parser.parse_args()

    if args.synthetic:
        synthetic_check()
        return

    pareto_betas = None
    if args.pareto_betas:
        pareto_betas = [int(b) for b in args.pareto_betas.split(",")]

    synthetic_check()
    doom_run(args.time_budget, args.max_layers, pareto_betas=pareto_betas)


if __name__ == "__main__":
    main()
