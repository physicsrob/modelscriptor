"""CP-SAT scheduler for `forward_compile`.

Produces an optimal placement of every graph node into transformer
layers, an optimal cancellation timing for every node's residual
columns, and (under `flex_routing=True`) an optimal attention-versus-
MLP routing for every standalone `Linear`.

See `docs/cpsat_scheduler.md` for the architecture spec — this module
is its implementation.

Public API:

- `Costs` — objective weights `(alpha, beta, gamma)`.
- `ScheduleAssignment` — solver output contract: per-node layer,
  cancel layer, and routing.
- `solve_schedule(...)` — build and solve, return `ScheduleAssignment`.
  Raises `RuntimeError` on infeasibility, time-out without a feasible
  solution, or `FEASIBLE`-but-not-`OPTIMAL` unless
  `allow_suboptimal=True`.

The probe script `scripts/cpsat_schedule.py` uses `_solve_full` for
richer diagnostic output (solver status string, objective value,
wall time, solver log).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.compiler.forward.sibling_clusters import SiblingClusterAnalyzer
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.graph import (
    Add,
    Attn,
    Concatenate,
    Embedding,
    InputNode,
    Linear,
    Node,
)
from torchwright.graph.misc import LiteralValue
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.graph.relu import ReLU


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Costs:
    """Objective weights for the CP-SAT solver.

    Total objective = `alpha * n_layers + beta * total_attn_heads +
    gamma * total_mlp_bypass_slots`.

    Defaults: `alpha=1, beta=0, gamma=0` — pure layer minimization.

    `beta` (long-sequence regime). Per-token attention compute scales
    as `O(L · d_head)` per head for sequence length `L`; per-layer
    compute (the full `d × d_hidden` MLP matmul plus the `4 · d²`
    attention QKVO matmuls) is independent of `L`. For long sequences,
    set `beta` above zero to push routing toward MLP. Rule of thumb:
    `beta ≈ L` makes one attention head equivalent to one extra layer.

    `gamma` (MLP bypass slot pressure). Less commonly useful — the
    per-layer MLP matmul costs the full `d × d_hidden` regardless of
    how many slots are used. `gamma=0` is the normal case.
    """

    alpha: int = 1
    beta: int = 0
    gamma: int = 0


@dataclass(frozen=True)
class ScheduleAssignment:
    """Per-node placement, cancellation, and routing decisions.

    Returned by `solve_schedule`. Consumed by `DirectedLayerScheduler`
    to replay the schedule through the existing per-layer code path.

    `node_to_layer[n]` is the transformer layer where node `n`
    executes. `node_to_cancel_layer[n]` is the layer where `n`'s
    residual columns are reclaimed (set to `n_layers` for nodes that
    stay alive forever — inputs, output, output-cone leaves).
    `node_to_routing[n]` is `"attn"` or `"mlp"` — which sublayer of
    `node_to_layer[n]` runs the op.

    Every schedulable node — every non-`Concatenate`, non-input node
    in the ancestor cone of `output_node` — appears in all three
    dicts.
    """

    node_to_layer: Dict[int, int]
    node_to_cancel_layer: Dict[int, int]
    node_to_routing: Dict[int, str]
    n_layers: int


@dataclass(frozen=True)
class SolveStats:
    """Solver metadata for diagnostics.

    Returned alongside an optional `ScheduleAssignment` from
    `_solve_full`. Useful for the probe script and for logging the
    solver gap when `allow_suboptimal=True`.
    """

    status_name: str
    objective_value: int        # -1 if no feasible solution
    best_objective_bound: float # tight LB the solver proved
    wall_time_s: float
    solver_log: str
    total_attn_heads: int       # -1 if no feasible solution
    total_mlp_bypass_slots: int # -1 if no feasible solution
    is_optimal: bool
    n_symmetry_constraints: int = 0  # lex-min constraints added before solve


# Routing constants used both by this module and by the probe script.
ATTN = "attn"
MLP = "mlp"


# ---------------------------------------------------------------------------
# Static graph preprocessing
# ---------------------------------------------------------------------------


@dataclass
class Chain:
    """One detected `L1 -> ReLU -> L2` chain.

    `exclusive` = `L1` has no effective consumers other than `R`, so
    `L1` doesn't need its own residual position (it's simulated inline
    inside `linear1` from its input's residual columns).
    """

    chain_id: int
    l1: Linear
    relu: ReLU
    l2: Linear
    exclusive: bool

    @property
    def width(self) -> int:
        return len(self.relu)


@dataclass
class GraphModel:
    """Static analysis of the graph that the CP-SAT model is built over."""

    graph: GraphAnalyzer
    schedulable: List[Node]              # nodes that need a time slot
    edges: List[Tuple[Node, Node]]       # (u, v) Concatenate-transparent
    consumers_eff: Dict[Node, Set[Node]] # effective consumers (Concat-transparent)
    chains: List[Chain]
    node_to_chain: Dict[Node, Chain]     # any of L1/R/L2 -> Chain
    output_node: Node
    pos_encoding: PosEncoding
    input_nodes: List[Node]              # pre-allocated inputs (incl. LiteralValue)
    pinned_nodes: Set[Node]              # never freed


def _effective_consumers(
    graph: GraphAnalyzer, node: Node, cache: Dict[Node, Set[Node]]
) -> Set[Node]:
    """Mirror `LayerScheduler._get_effective_consumers`."""
    if node in cache:
        return cache[node]
    result: Set[Node] = set()
    for consumer in graph.get_consumers(node):
        if isinstance(consumer, Concatenate):
            downstream = _effective_consumers(graph, consumer, cache)
            if downstream:
                result |= downstream
            else:
                # Terminal Concatenate (output): keep as consumer so
                # children stay alive.
                result.add(consumer)
        else:
            result.add(consumer)
    cache[node] = result
    return result


def _detect_chains_static(
    graph: GraphAnalyzer,
    schedulable: Set[Node],
    consumers_eff: Dict[Node, Set[Node]],
) -> List[Chain]:
    """Mirror `LayerScheduler._detect_chains` over the entire graph.

    Each ReLU matches at most one chain. Iteration is in node-id order
    for determinism.
    """
    chains: List[Chain] = []
    seen_relus: Set[Node] = set()

    linears = sorted(
        (n for n in schedulable if isinstance(n, Linear)),
        key=lambda n: n.node_id,
    )
    for l1 in linears:
        for consumer in graph.get_consumers(l1):
            if not isinstance(consumer, ReLU) or consumer in seen_relus:
                continue
            relu = consumer
            relu_eff = consumers_eff.get(relu, set())
            l2_candidates = [c for c in relu_eff if isinstance(c, Linear)]
            if len(relu_eff) != 1 or len(l2_candidates) != 1:
                continue
            l2 = l2_candidates[0]
            if l2.inputs[0] is not relu:
                continue
            l1_eff = consumers_eff.get(l1, set())
            exclusive = l1_eff == {relu}
            chains.append(
                Chain(
                    chain_id=len(chains),
                    l1=l1,
                    relu=relu,
                    l2=l2,
                    exclusive=exclusive,
                )
            )
            seen_relus.add(relu)
            break

    return chains


def build_graph_model(output_node: Node, pos_encoding: PosEncoding) -> GraphModel:
    """Run all the static preprocessing the CP-SAT builder needs."""
    graph = GraphAnalyzer(output_node)
    output_node = graph.get_output_node()
    all_nodes = graph.get_all_nodes()

    # Inputs are pre-allocated by compile.py's initialization; they
    # don't need a time slot. LiteralValue is included by
    # is_input_node; treat the same way.
    input_nodes: List[Node] = [
        n for n in all_nodes if graph.is_input_node(n)
    ]

    schedulable: List[Node] = sorted(
        (
            n
            for n in all_nodes
            if not isinstance(n, Concatenate) and n not in set(input_nodes)
        ),
        key=lambda n: n.node_id,
    )

    # Effective-consumers cache (Concat-transparent).
    consumers_eff: Dict[Node, Set[Node]] = {}
    for n in all_nodes:
        if isinstance(n, Concatenate):
            continue
        _effective_consumers(graph, n, consumers_eff)

    # Edges: every direct dependency, traversing Concatenate inputs.
    edges_set: Set[Tuple[int, int]] = set()
    edges: List[Tuple[Node, Node]] = []
    for v in schedulable:
        for inp in v.inputs:
            if isinstance(inp, Concatenate):
                leaves = flatten_concat_nodes([inp])
            else:
                leaves = [inp]
            for u in leaves:
                key = (u.node_id, v.node_id)
                if key not in edges_set:
                    edges_set.add(key)
                    edges.append((u, v))
        for pred in v.scheduling_predecessors:
            key = (pred.node_id, v.node_id)
            if key not in edges_set:
                edges_set.add(key)
                edges.append((pred, v))

    chains = _detect_chains_static(graph, set(schedulable), consumers_eff)
    node_to_chain: Dict[Node, Chain] = {}
    for c in chains:
        node_to_chain[c.l1] = c
        node_to_chain[c.relu] = c
        node_to_chain[c.l2] = c

    pinned_nodes: Set[Node] = set(input_nodes)
    pinned_nodes.add(output_node)

    return GraphModel(
        graph=graph,
        schedulable=schedulable,
        edges=edges,
        consumers_eff=consumers_eff,
        chains=chains,
        node_to_chain=node_to_chain,
        output_node=output_node,
        pos_encoding=pos_encoding,
        input_nodes=input_nodes,
        pinned_nodes=pinned_nodes,
    )


# ---------------------------------------------------------------------------
# Routing and cost helpers
# ---------------------------------------------------------------------------
#
# Used internally by the model builder, and by the probe script for
# diagnostic prints. Treat as implementation detail of this module —
# external callers should use `solve_schedule` instead.


def routing(node: Node, gm: GraphModel, policy: SchedulingPolicy) -> str:
    """Static routing decision under the given policy.

    Used when `flex_routing=False`: every node has a fixed sublayer.
    With `flex_routing=True`, only `is_flex(n)` nodes' modes become
    CP-SAT decision variables; others still use this routing.
    """
    if isinstance(node, Attn):
        return ATTN
    if isinstance(node, Add):
        return ATTN
    if isinstance(node, ReLU):
        return MLP  # standalone or chain-internal — both MLP
    if isinstance(node, LiteralValue):
        return MLP
    if isinstance(node, Linear):
        if node in gm.node_to_chain:
            return MLP
        if policy.local_in_attention == "always":
            return ATTN
        return MLP
    raise TypeError(f"Unknown schedulable node type: {type(node).__name__}")


def is_flex(node: Node, gm: GraphModel) -> bool:
    """True iff this node's routing is a CP-SAT decision variable
    when `flex_routing=True`.

    Standalone Linears (not part of any L1->R->L2 chain) can run in
    attention (`heads = ⌈d_input/d_head⌉`) or in MLP bypass (`slots
    = 2 · d_output`). The heuristic picks one statically per policy;
    CP-SAT can pick per-node.

    Chain L1/R/L2 stay locked to MLP — splitting a chain into
    separate ops would need a different model structure.

    `Attn` / `Add` / standalone `ReLU` / `LiteralValue` stay locked
    because they have only one valid sublayer.
    """
    if isinstance(node, Linear) and node not in gm.node_to_chain:
        return True
    return False


def heads_for(node: Node, d_head: int) -> int:
    """Heads consumed if attention-routed.

    Mirrors `LayerScheduler._heads_*`. `Add` cost uses the OPTIMISTIC
    free-add count (`⌈d_out/d_head⌉` — one head per `d_head`-wide
    chunk of the live addend, copied into the dead addend's cols).
    The model precondition (see `docs/cpsat_scheduler.md` §3) requires
    no `Add` to have a `Concatenate` input, so the dynamic free-vs-
    compute classification always resolves to free-add.
    """
    if isinstance(node, Attn):
        return (node.d_v + d_head - 1) // d_head
    if isinstance(node, Linear):
        d_in = len(node.inputs[0])
        return (d_in + d_head - 1) // d_head
    if isinstance(node, Add):
        d_out = len(node)
        return (d_out + d_head - 1) // d_head
    return 0


def slots_for(node: Node, gm: GraphModel) -> int:
    """MLP slots consumed if MLP-routed.

    Chain-internal nodes (L1/R/L2) return 0 individually — the chain
    composite (modeled separately) carries the `len(R)` slot demand.
    """
    if node in gm.node_to_chain:
        return 0
    if isinstance(node, ReLU):
        return len(node)
    if isinstance(node, Linear):
        return 2 * node.d_output  # MLP bypass
    if isinstance(node, LiteralValue):
        return 0
    return 0


def uses_residual(node: Node, gm: GraphModel) -> bool:
    """True iff this node gets its own residual-stream column allocation.

    Chain `R`: no — its activations live in MLP hidden slots, not residual.
    Chain `L1` (exclusive): no — computed inline inside `linear1` from
        its input's residual cols, never written back to the stream.
    Chain `L1` (non-exclusive): yes — has consumers besides `R`, so
        needs a residual position alongside the chain.
    Chain `L2`: yes — chain output writes to residual.
    All other schedulable nodes (standalone Linear/Add/Attn/ReLU/
        LiteralValue): yes.
    """
    chain = gm.node_to_chain.get(node)
    if chain is None:
        return True
    if node is chain.relu:
        return False
    if node is chain.l1:
        return not chain.exclusive
    return True  # chain.l2


# ---------------------------------------------------------------------------
# Symmetry breaking
# ---------------------------------------------------------------------------


def _chain_fingerprint(chain) -> Tuple:
    """Hashable structural signature of a sibling-cluster chain.

    Two chains with identical fingerprints have the same multiset of
    (op type, output width) pairs over their nodes — a strong proxy
    for "functionally equivalent pipeline" when the chains feed the
    same join.  The schedule cost (n_layers, total heads, total slots)
    is invariant under any permutation of equally-fingerprinted chains
    in the same cluster, so we can safely add a lex-min constraint
    between their layer assignments to break that combinatorial
    symmetry in CP-SAT's search.
    """
    return tuple(sorted((type(n).__name__, len(n)) for n in chain.nodes))


def _add_symmetry_breaking(
    model: cp_model.CpModel,
    gm: GraphModel,
    layer_var: Dict[int, cp_model.IntVar],
    graph: GraphAnalyzer,
    hint_layers: Optional[Dict[int, int]] = None,
) -> int:
    """Add lex-min constraints across structurally-equivalent sibling chains.

    Detects parallel chains feeding common ``Concatenate`` joins via
    :class:`SiblingClusterAnalyzer`, groups them by
    ``_chain_fingerprint``, and chains
    ``layer_var[chain_i.terminal] <= layer_var[chain_{i+1}.terminal]``
    within each fingerprint group.  For a group of N equally-shaped
    chains this eliminates ``N!`` permutations from the search tree.

    **Status:** experimental, off by default.  On the DOOM headless
    graph (4440 nodes, 196 lex-min constraints across 36 fingerprint
    groups) the constraints push CP-SAT into a regime where it tightens
    the lower bound (46 → 49) but spends so long on bound proofs that
    it never finds a feasible incumbent within reasonable budgets
    (10–600s) — even though the heuristic warm-start hint is verifiably
    feasible against every constraint.  Limiting to large groups only
    (≥ 8 chains) didn't help.  The likely cause is CP-SAT's worker
    allocation shifting toward ``objective_lb_search`` when
    propagation is denser, starving incumbent-finding workers.  Worth
    revisiting with a different solver-parameter mix (LNS-heavy,
    ``use_lns_only``, or a fixed search strategy that respects the
    hint).

    Returns the number of constraints added (for diagnostics).
    """
    analyzer = SiblingClusterAnalyzer(graph, min_chains=2, min_peak_width=1)
    clusters = analyzer.analyze()

    def _canonical_key(chain):
        if hint_layers is not None:
            hl = hint_layers.get(chain.terminal.node_id)
            if hl is not None:
                return (hl, chain.chain_id)
        return (0, chain.chain_id)

    n_added = 0
    for cluster in clusters.clusters.values():
        groups: Dict[Tuple, List] = {}
        for chain in cluster.chains:
            groups.setdefault(_chain_fingerprint(chain), []).append(chain)
        for chains in groups.values():
            if len(chains) < 2:
                continue
            chains.sort(key=_canonical_key)
            for i in range(len(chains) - 1):
                a_term = chains[i].terminal
                b_term = chains[i + 1].terminal
                if (
                    a_term.node_id in layer_var
                    and b_term.node_id in layer_var
                ):
                    model.Add(
                        layer_var[a_term.node_id] <= layer_var[b_term.node_id]
                    )
                    n_added += 1
    return n_added


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_schedule(
    output_node: Node,
    pos_encoding: PosEncoding,
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    time_budget_s: float = 60.0,
    max_layers: int = 60,
    allow_suboptimal: bool = False,
    hint_layers: Optional[Dict[int, int]] = None,
    hint_routing: Optional[Dict[int, str]] = None,
    hint_cancel: Optional[Dict[int, int]] = None,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    reserve_heads: int = 0,
    assume_zero_init: bool = False,
    symmetry_breaking: bool = False,
) -> ScheduleAssignment:
    """Build and solve the CP-SAT scheduling model.

    Returns a `ScheduleAssignment` for the graph rooted at `output_node`.

    Args:
        output_node: graph output. Defines the ancestor cone the
            scheduler operates over.
        pos_encoding: positional encoding node (always allocated by
            `forward_compile`; subtracted from the residual budget).
        d, d_head, d_hidden: transformer geometry. `n_heads_per_layer
            = d // d_head`. Residual budget is `d - input_residual_cols`.
        costs: objective weights. See `Costs`.
        flex_routing: if True, CP-SAT picks attention vs MLP for each
            standalone `Linear`. If False, standalone Linears use the
            static routing dictated by `policy.local_in_attention`.
        time_budget_s: per-solve wall-clock cap.
        max_layers: search horizon. Should be at least the heuristic's
            layer count.
        allow_suboptimal: if False (default), raise on `FEASIBLE` (only
            return on proven `OPTIMAL`). If True, accept a
            feasible-but-not-proven-optimal schedule.
        hint_layers: optional warm-start mapping `node_id -> layer`.
        hint_routing: optional warm-start mapping
            `node_id -> "attn"|"mlp"` for flex Linears.  When the
            heuristic placed a standalone Linear in attention vs
            MLP-bypass, hinting the same routing lets CP-SAT
            reconstruct the heuristic's solution as a starting
            incumbent.
        hint_cancel: optional warm-start mapping `node_id -> layer`
            for the cancel layer.  Captures when the heuristic freed
            each node's columns; combined with `hint_layers` this
            gives a complete schedule the solver can verify and
            improve from.
        cancel_slack: when not None, restrict each non-pinned node's
            cancel layer to `[earliest_dead, earliest_dead + K]`
            where `earliest_dead = max(layer[c] + 1)` over consumers
            and ``K == cancel_slack``.  Cuts the cancel-decision
            search space ~30x at K=2 with negligible loss of
            optimality (the heuristic almost always cancels within
            1–2 layers of the last consumer).  Set to None to keep
            the wide `[layer[n]+1, max_layers]` domain.  Default 2.
        policy: only consulted when `flex_routing=False`. Defaults to
            `LEGACY_POLICY`.
        reserve_heads: per-layer attention-head budget reserved
            beyond the modeled compute + cancel + dirty terms.
            Defaults to 0.  Raise it for graphs whose attention heads
            are saturated by ops outside the model.
        assume_zero_init: if True, the model assumes the runtime
            zero-initialises the residual stream (so the heuristic
            emits no BIRTH-layer dirty-column cancels for fresh
            allocations on the initially-free pool).  Pair this with
            ``forward_compile(assume_zero_init=True)`` so the heuristic
            and CP-SAT model agree.  Defaults to False — the
            conservative model that mirrors the heuristic's defensive
            cancellation behaviour from commit cf4af42.

    Raises `RuntimeError` on `INFEASIBLE`, time-out without a feasible
    solution, or `FEASIBLE` when `allow_suboptimal=False`. Raises also
    when the graph violates a model precondition (see
    `docs/cpsat_scheduler.md` §3): currently, an `Add` whose input is a
    `Concatenate`.
    """
    assignment, stats = _solve_full(
        output_node,
        pos_encoding,
        d=d,
        d_head=d_head,
        d_hidden=d_hidden,
        costs=costs,
        flex_routing=flex_routing,
        time_budget_s=time_budget_s,
        max_layers=max_layers,
        hint_layers=hint_layers,
        hint_routing=hint_routing,
        hint_cancel=hint_cancel,
        cancel_slack=cancel_slack,
        policy=policy,
        log_search_progress=False,
        reserve_heads=reserve_heads,
        assume_zero_init=assume_zero_init,
        symmetry_breaking=symmetry_breaking,
    )
    if assignment is None:
        raise RuntimeError(
            f"CP-SAT returned {stats.status_name}; no schedule produced "
            f"(best_objective_bound={stats.best_objective_bound}, "
            f"n_symmetry_constraints={stats.n_symmetry_constraints})"
        )
    if not stats.is_optimal and not allow_suboptimal:
        raise RuntimeError(
            f"CP-SAT returned {stats.status_name} but did not prove "
            f"optimality: objective={stats.objective_value}, "
            f"best_objective_bound={stats.best_objective_bound}; "
            f"pass allow_suboptimal=True to accept this schedule"
        )
    return assignment


def _solve_full(
    output_node: Node,
    pos_encoding: PosEncoding,
    *,
    d: int,
    d_head: int,
    d_hidden: int,
    costs: Costs = Costs(),
    flex_routing: bool = True,
    time_budget_s: float = 60.0,
    max_layers: int = 60,
    hint_layers: Optional[Dict[int, int]] = None,
    hint_routing: Optional[Dict[int, str]] = None,
    hint_cancel: Optional[Dict[int, int]] = None,
    cancel_slack: Optional[int] = 2,
    policy: Optional[SchedulingPolicy] = None,
    log_search_progress: bool = False,
    reserve_heads: int = 0,
    assume_zero_init: bool = False,
    symmetry_breaking: bool = False,
) -> Tuple[Optional[ScheduleAssignment], SolveStats]:
    """Internal solve that returns both the assignment and the solver stats.

    Used by `solve_schedule` (which converts non-`OPTIMAL` results into
    exceptions per its `allow_suboptimal` policy) and by the probe
    script in `scripts/cpsat_schedule.py` (which wants the rich stats
    for diagnostic output regardless of solver outcome).

    Returns `(None, stats)` when no feasible solution exists; otherwise
    `(assignment, stats)`.

    Raises `RuntimeError` for graph-precondition violations only —
    these are not solver outcomes, they're prerequisites the solver
    cannot recover from.
    """
    if policy is None:
        policy = LEGACY_POLICY

    if costs.alpha == 0 and costs.beta == 0 and costs.gamma == 0:
        raise ValueError("alpha=beta=gamma=0 — no objective.")

    gm = build_graph_model(output_node, pos_encoding)

    # Precondition: no Add input is a Concatenate (see architecture
    # doc §3 Model preconditions). The model assumes free-add cost;
    # Concatenate-input Adds force the heuristic into compute-add and
    # would invalidate the LB.
    for n in gm.schedulable:
        if not isinstance(n, Add):
            continue
        for inp in n.inputs:
            if isinstance(inp, Concatenate):
                raise RuntimeError(
                    f"CP-SAT precondition violation: Add node "
                    f"id={n.node_id} has a Concatenate input "
                    f"({type(inp).__name__} id={inp.node_id}). The "
                    f"CP-SAT model assumes free-add cost; "
                    f"Concatenate-input Adds force compute-add and "
                    f"would invalidate the resource budget. See "
                    f"docs/cpsat_scheduler.md §3 Model preconditions."
                )

    n_heads_per_layer = d // d_head
    input_residual = sum(len(n) for n in gm.input_nodes)
    if gm.pos_encoding not in set(gm.input_nodes):
        input_residual += len(gm.pos_encoding)
    available_residual = d - input_residual
    if available_residual <= 0:
        raise RuntimeError(
            f"Inputs alone require {input_residual} residual columns, "
            f"but d={d}. No room for intermediate nodes."
        )

    model = cp_model.CpModel()

    # ---- layer_var per schedulable node ----
    layer_var: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        layer_var[n.node_id] = model.NewIntVar(
            0, max_layers - 1, f"L_n{n.node_id}"
        )

    # Chain composites: L1, R, L2 share one layer.
    for c in gm.chains:
        model.Add(layer_var[c.l1.node_id] == layer_var[c.relu.node_id])
        model.Add(layer_var[c.relu.node_id] == layer_var[c.l2.node_id])

    # ---- Routing: is_attn[n] BoolVar (or fixed literal) per node ----
    # is_attn[n] == 1 means the node runs in the attention sublayer
    # at its layer; is_attn[n] == 0 means it runs in MLP.
    is_attn: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        if flex_routing and is_flex(n, gm):
            v = model.NewBoolVar(f"is_attn_n{n.node_id}")
        else:
            r = routing(n, gm, policy)
            v = model.NewBoolVar(f"is_attn_n{n.node_id}_pinned")
            if r == ATTN:
                model.Add(v == 1)
            else:
                model.Add(v == 0)
        is_attn[n.node_id] = v

    # ---- Dependency constraints ----
    # Edge u->v: same-layer ok iff u is_attn AND v is mlp (i.e., NOT
    # v is_attn). Otherwise layer[v] > layer[u].
    input_ids = {n.node_id for n in gm.input_nodes}
    for u, v in gm.edges:
        if u.node_id in input_ids:
            continue
        if (
            u in gm.node_to_chain
            and v in gm.node_to_chain
            and gm.node_to_chain[u] is gm.node_to_chain[v]
        ):
            continue
        u_attn = is_attn[u.node_id]
        v_attn = is_attn[v.node_id]
        # same_layer_ok = u_attn AND (NOT v_attn)
        same_ok = model.NewBoolVar(f"so_n{u.node_id}_n{v.node_id}")
        model.AddBoolAnd([u_attn, v_attn.Not()]).OnlyEnforceIf(same_ok)
        model.AddBoolOr([u_attn.Not(), v_attn]).OnlyEnforceIf(same_ok.Not())
        model.Add(layer_var[v.node_id] >= layer_var[u.node_id]).OnlyEnforceIf(same_ok)
        model.Add(
            layer_var[v.node_id] >= layer_var[u.node_id] + 1
        ).OnlyEnforceIf(same_ok.Not())

    # ---- Cancel layer per schedulable node ----
    # The natural lower bound on cancel_layer[n] is
    # ``max(layer[c] + 1)`` over consumers — the columns must outlive
    # every reader.  The natural upper bound is ``max_layers``, which
    # leaves ~60 candidate values per node on a DOOM-scale graph.
    # When ``cancel_slack`` is set, restrict to a small window above
    # the lower bound: the heuristic almost always cancels within 1–2
    # layers of the last consumer, so K=2 cuts the cancel decision
    # space ~30x with negligible loss of optimality.
    cancel_layer: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        cl = model.NewIntVar(0, max_layers, f"cl_n{n.node_id}")
        cancel_layer[n.node_id] = cl
        model.Add(cl >= layer_var[n.node_id] + 1)
        if n in gm.pinned_nodes:
            model.Add(cl == max_layers)
            continue
        keep_forever = False
        consumer_layer_vars: List[cp_model.IntVar] = []
        for c in gm.consumers_eff.get(n, set()):
            if isinstance(c, Concatenate):
                model.Add(cl == max_layers)
                keep_forever = True
                break
            if c.node_id in layer_var:
                model.Add(cl >= layer_var[c.node_id] + 1)
                consumer_layer_vars.append(layer_var[c.node_id])
        if keep_forever:
            continue
        if cancel_slack is not None and consumer_layer_vars:
            last_cons = model.NewIntVar(
                0, max_layers - 1, f"last_cons_n{n.node_id}"
            )
            model.AddMaxEquality(last_cons, consumer_layer_vars)
            model.Add(cl <= last_cons + 1 + cancel_slack)
        elif cancel_slack is not None and not consumer_layer_vars:
            # No layer-bound consumers — cancel can fire right after
            # the node's own birth layer.
            model.Add(cl <= layer_var[n.node_id] + 1 + cancel_slack)

    # ---- Combined attn-heads + cancel-cols cumulative ----
    # Per-node attn interval is OPTIONAL (gated by is_attn[n]) when
    # the node could run in either sublayer; for pinned nodes, the
    # bool is constant and CP-SAT presolve drops the unreachable
    # branch.
    attn_intervals: List = []
    attn_demands: List[int] = []
    for n in gm.schedulable:
        h = heads_for(n, d_head)
        if h <= 0:
            continue
        end = model.NewIntVar(1, max_layers, f"aend_n{n.node_id}")
        model.Add(end == layer_var[n.node_id] + 1)
        iv = model.NewOptionalIntervalVar(
            layer_var[n.node_id], 1, end, is_attn[n.node_id], f"aiv_n{n.node_id}"
        )
        attn_intervals.append(iv)
        attn_demands.append(h * d_head)

    cancel_intervals: List = []
    cancel_demands: List[int] = []
    dirty_intervals: List = []
    dirty_demands: List[int] = []
    for n in gm.schedulable:
        if n in gm.pinned_nodes:
            continue
        if not uses_residual(n, gm):
            # Chain-internal exclusive L1 and chain-internal ReLU live
            # in MLP hidden slots, never in the residual stream — no
            # columns to cancel.  Skipping prevents the cumulative from
            # over-counting on graphs with wide chain hidden widths
            # (e.g. d_hidden_chain > n_heads_per_layer * d_head).
            continue
        c_end = model.NewIntVar(1, max_layers + 1, f"cend_n{n.node_id}")
        model.Add(c_end == cancel_layer[n.node_id] + 1)
        iv = model.NewIntervalVar(
            cancel_layer[n.node_id], 1, c_end, f"civ_n{n.node_id}"
        )
        cancel_intervals.append(iv)
        cancel_demands.append(len(n))

        # Birth-layer dirty-column cancel.  When `assume_zero_init` is
        # False (the default, mirroring the heuristic's defensive
        # behaviour from commit cf4af42), every fresh allocation pays
        # a cancel head to clear the column's prior value before its
        # additive write.  When True, the runtime is contracted to
        # zero-initialise the residual stream and the heuristic skips
        # these cancels — so we skip them in the model too.
        # `Add` is always exempt: under the model precondition (no
        # `Concatenate`-input `Add`) the heuristic always reaches the
        # `Add` via the free-add path, reusing the dead addend's
        # already-allocated columns — no fresh allocation, no dirty
        # bits to clear.
        if assume_zero_init or isinstance(n, Add):
            continue
        d_end = model.NewIntVar(1, max_layers, f"dend_n{n.node_id}")
        model.Add(d_end == layer_var[n.node_id] + 1)
        d_iv = model.NewIntervalVar(
            layer_var[n.node_id], 1, d_end, f"div_n{n.node_id}"
        )
        dirty_intervals.append(d_iv)
        dirty_demands.append(len(n))

    # `reserve_heads` is a safety knob for graphs whose attention
    # heads are saturated by ops outside the model (e.g. bias writes
    # folded into deferred Linears); default 0.
    effective_capacity = max(0, n_heads_per_layer - reserve_heads) * d_head
    if attn_intervals or cancel_intervals or dirty_intervals:
        model.AddCumulative(
            attn_intervals + cancel_intervals + dirty_intervals,
            attn_demands + cancel_demands + dirty_demands,
            effective_capacity,
        )

    # ---- MLP slots cumulative ----
    # For flex nodes, MLP demand is gated by NOT(is_attn). Chain
    # composites and standalone ReLUs are always MLP-routed.
    mlp_intervals: List = []
    mlp_demands: List[int] = []
    for n in gm.schedulable:
        s = slots_for(n, gm)
        if s <= 0:
            continue
        end = model.NewIntVar(1, max_layers, f"mend_n{n.node_id}")
        model.Add(end == layer_var[n.node_id] + 1)
        iv = model.NewOptionalIntervalVar(
            layer_var[n.node_id], 1, end, is_attn[n.node_id].Not(), f"miv_n{n.node_id}"
        )
        mlp_intervals.append(iv)
        mlp_demands.append(s)
    for c in gm.chains:
        end = model.NewIntVar(1, max_layers, f"mend_c{c.chain_id}")
        model.Add(end == layer_var[c.relu.node_id] + 1)
        iv = model.NewIntervalVar(
            layer_var[c.relu.node_id], 1, end, f"miv_c{c.chain_id}"
        )
        mlp_intervals.append(iv)
        mlp_demands.append(c.width)
    if mlp_intervals:
        model.AddCumulative(mlp_intervals, mlp_demands, d_hidden)

    # ---- Residual cumulative ----
    residual_nodes = [n for n in gm.schedulable if uses_residual(n, gm)]
    resid_intervals: List = []
    resid_demands: List[int] = []
    for n in residual_nodes:
        size = model.NewIntVar(1, max_layers + 1, f"rsz_n{n.node_id}")
        model.Add(size == cancel_layer[n.node_id] - layer_var[n.node_id])
        iv = model.NewIntervalVar(
            layer_var[n.node_id],
            size,
            cancel_layer[n.node_id],
            f"riv_n{n.node_id}",
        )
        resid_intervals.append(iv)
        resid_demands.append(len(n))
    if resid_intervals:
        model.AddCumulative(resid_intervals, resid_demands, available_residual)

    # ---- Aggregate counters for the objective ----
    attn_term: List = []
    mlp_bypass_term: List = []
    fixed_attn_heads = 0
    for n in gm.schedulable:
        h = heads_for(n, d_head)
        if h == 0:
            continue
        if flex_routing and is_flex(n, gm):
            attn_term.append(h * is_attn[n.node_id])
        else:
            r = routing(n, gm, policy)
            if r == ATTN:
                fixed_attn_heads += h
    total_attn_heads = model.NewIntVar(
        0,
        fixed_attn_heads + sum(heads_for(n, d_head) for n in gm.schedulable),
        "total_attn_heads",
    )
    if attn_term:
        model.Add(total_attn_heads == fixed_attn_heads + sum(attn_term))
    else:
        model.Add(total_attn_heads == fixed_attn_heads)

    fixed_mlp_bypass = 0
    for n in gm.schedulable:
        if not is_flex(n, gm):
            continue
        if flex_routing:
            mlp_bypass_term.append((2 * n.d_output) * is_attn[n.node_id].Not())
        else:
            r = routing(n, gm, policy)
            if r == MLP:
                fixed_mlp_bypass += 2 * n.d_output
    total_mlp_bypass = model.NewIntVar(
        0,
        fixed_mlp_bypass + sum(2 * n.d_output for n in gm.schedulable if is_flex(n, gm)),
        "total_mlp_bypass",
    )
    if mlp_bypass_term:
        model.Add(total_mlp_bypass == fixed_mlp_bypass + sum(mlp_bypass_term))
    else:
        model.Add(total_mlp_bypass == fixed_mlp_bypass)

    # ---- Objective ----
    makespan_layer = model.NewIntVar(0, max_layers, "makespan_layer")
    if gm.schedulable:
        model.AddMaxEquality(
            makespan_layer, [layer_var[n.node_id] for n in gm.schedulable]
        )
    else:
        model.Add(makespan_layer == 0)
    n_layers_var = model.NewIntVar(0, max_layers + 1, "n_layers")
    model.Add(n_layers_var == makespan_layer + 1)

    model.Minimize(
        costs.alpha * n_layers_var
        + costs.beta * total_attn_heads
        + costs.gamma * total_mlp_bypass
    )

    # ---- Hint ----
    # A complete hint (layer + routing + cancel) gives CP-SAT a
    # full feasible incumbent it can verify and improve from, which
    # is much faster than reconstructing routing and cancel timing
    # from a layer-only hint.  Hints are soft — CP-SAT is free to
    # discard them and explore alternatives.
    if hint_layers is not None:
        for nid, L in hint_layers.items():
            if nid in layer_var and 0 <= L < max_layers:
                model.AddHint(layer_var[nid], L)
    if hint_routing is not None:
        for nid, route in hint_routing.items():
            if nid in is_attn:
                model.AddHint(is_attn[nid], 1 if route == ATTN else 0)
    if hint_cancel is not None:
        for nid, L in hint_cancel.items():
            if nid in cancel_layer and 0 <= L <= max_layers:
                model.AddHint(cancel_layer[nid], L)

    # ---- Symmetry breaking ----
    # Sibling chains feeding common ``Concatenate`` joins are
    # interchangeable when their internal structures match — adding
    # lex-min between their terminals' layer assignments cuts entire
    # ``N!`` permutation slices out of CP-SAT's search tree.  Pass the
    # hint so chain ordering matches the warm-start incumbent.
    n_sym = 0
    if symmetry_breaking:
        n_sym = _add_symmetry_breaking(
            model, gm, layer_var, gm.graph, hint_layers=hint_layers
        )

    # ---- Decision strategy: schedule by critical path first ----
    nodes_by_cp = sorted(
        gm.schedulable,
        key=lambda n: -gm.graph.get_critical_path_length(n),
    )
    model.AddDecisionStrategy(
        [layer_var[n.node_id] for n in nodes_by_cp],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MIN_VALUE,
    )

    # ---- Solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_budget_s
    solver.parameters.log_search_progress = log_search_progress
    solver.parameters.num_search_workers = 16

    log_buf: List[str] = []

    def _log(msg: str) -> None:
        log_buf.append(msg)

    solver.log_callback = _log

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if has_solution:
        node_to_layer: Dict[int, int] = {}
        node_to_cancel_layer: Dict[int, int] = {}
        node_to_routing: Dict[int, str] = {}
        for n in gm.schedulable:
            node_to_layer[n.node_id] = solver.Value(layer_var[n.node_id])
            node_to_cancel_layer[n.node_id] = solver.Value(cancel_layer[n.node_id])
            node_to_routing[n.node_id] = (
                ATTN if solver.Value(is_attn[n.node_id]) else MLP
            )
        n_layers = solver.Value(n_layers_var)
        total_heads = solver.Value(total_attn_heads)
        total_bypass = solver.Value(total_mlp_bypass)
        objective = int(solver.ObjectiveValue())
        assignment: Optional[ScheduleAssignment] = ScheduleAssignment(
            node_to_layer=node_to_layer,
            node_to_cancel_layer=node_to_cancel_layer,
            node_to_routing=node_to_routing,
            n_layers=n_layers,
        )
    else:
        total_heads = -1
        total_bypass = -1
        objective = -1
        assignment = None

    stats = SolveStats(
        status_name=solver.StatusName(status),
        objective_value=objective,
        best_objective_bound=float(solver.BestObjectiveBound()),
        wall_time_s=elapsed,
        solver_log="\n".join(log_buf),
        total_attn_heads=total_heads,
        total_mlp_bypass_slots=total_bypass,
        is_optimal=status == cp_model.OPTIMAL,
        n_symmetry_constraints=n_sym,
    )
    return assignment, stats
