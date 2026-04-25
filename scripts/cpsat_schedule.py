"""CP-SAT prototype for the forward_compile scheduler.

Builds a CP-SAT model of the headless DOOM graph and reports lower
bounds for layer count and total attention heads. Intent: see how
close the existing greedy heuristic is to the optimum and decide
whether to productionize an optimization-driven scheduler (per the
phase-B plan, this answers "Check-in 1").

The scheduling problem mirrors what `forward_compile` solves:

- Each schedulable graph node has a time variable t = 2*layer + sublayer
  (sublayer 0 = attention, 1 = MLP).
- Dependencies: t[v] > t[u] for every directed edge u->v
  (Concatenate-transparent).
- Routing is type-driven (static, per-policy): Attn / Add -> attention
  sublayer (even t), ReLU / chain composites -> MLP sublayer (odd t),
  standalone Linears split per policy mode (`local_in_attention`).
- Cumulative resource constraints at each time slot:
   * heads at even t <= d / d_head
   * MLP slots at odd t <= d_hidden
   * residual columns at every t <= d - input_residual  (aggregate, not
     2D-pack: ResidualStreamMap.allocate is scatter/gather, so
     fragmentation is not an issue)
- Cancel decisions: each schedulable node has a cancel_time variable
  bounded below by the latest effective-consumer time. Cancel cost is
  folded into the head budget at cancel_time as col-fractional heads
  (a relaxation, slightly optimistic).

Simplifications that bias the model toward LOWER bounds (the real
heuristic can only be worse than the LB):

- Add nodes are always treated as compute_add (worst case for heads),
  not free_add (which would reuse a dead input's columns).
- Cancel cost uses fractional heads instead of ceil(cols/d_head).
- Sibling-cluster admission is not modeled.
- The `pressure_threshold` sort-order flip in the heuristic is irrelevant
  to a global optimizer.

Run:
    # synthetic-graph sanity check (small, fast)
    uv run python -m scripts.cpsat_schedule --synthetic

    # full DOOM headless run (heavy, prefer Modal)
    make modal-run MODULE=scripts.cpsat_schedule

    # adjust solver budget per Pareto point
    make modal-run MODULE=scripts.cpsat_schedule ARGS="--time-budget 600"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch

from ortools.sat.python import cp_model

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)
from torchwright.compiler.residual_assignment import flatten_concat_nodes
from torchwright.doom.compile import compute_min_d_head
from torchwright.doom.game_graph import build_game_graph
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
from torchwright.reference_renderer.scenes import box_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


# ---------------------------------------------------------------------------
# Static graph preprocessing
# ---------------------------------------------------------------------------


@dataclass
class Chain:
    """One L1->ReLU->L2 chain. Mirrors LayerScheduler._detect_chains.

    `exclusive` = L1 has no effective consumers other than R, so L1
    doesn't need its own residual position (it's simulated inside
    linear1 from its input's columns).
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
    """Static analysis output consumed by the CP-SAT builder."""

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
    """Mirror LayerScheduler._get_effective_consumers."""
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
    """Mirror LayerScheduler._detect_chains over the entire graph.

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
# Routing and cost lookups
# ---------------------------------------------------------------------------


ATTN = "attn"
MLP = "mlp"


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
    attention (heads = ceil(d_input/d_head)) or in MLP bypass (slots
    = 2 * d_output). The heuristic picks one statically per policy;
    CP-SAT can pick per-node.

    Chain L1/R/L2 stay locked to MLP — splitting a chain into
    separate ops would need a different model structure.

    Attn / Add / standalone-ReLU / LiteralValue stay locked because
    they have only one valid sublayer.
    """
    if isinstance(node, Linear) and node not in gm.node_to_chain:
        return True
    return False


def heads_for(node: Node, d_head: int) -> int:
    """Heads consumed if attention-routed. Mirrors scheduler._heads_*.

    Add cost uses the OPTIMISTIC free_add count (ceil(d_out/d_head) — one
    head per d_head-wide chunk of the live addend, copied into the dead
    addend's cols). The heuristic falls back to compute_add (potentially
    twice the heads) when neither input is dead. We use the optimistic
    count so the LB stays a true LB (relaxation: assumes the dynamic Add
    classification always produces a free_add). The heuristic-produced
    schedule will fit this relaxation, so AddHint stays useful.
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
    composite (modeled separately) carries the len(R) slot demand.
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

    Chain R: no — its activations live in MLP hidden slots, not residual.
    Chain L1 (exclusive): no — computed inline inside linear1 from its
        input's residual cols, never written back to the stream.
    Chain L1 (non-exclusive): yes — has consumers besides R, so needs a
        residual position alongside the chain.
    Chain L2: yes — chain output writes to residual.
    All other schedulable nodes (standalone Linear/Add/Attn/ReLU/Literal):
        yes.
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
# CP-SAT model
# ---------------------------------------------------------------------------


@dataclass
class Costs:
    """Objective weights for the CP-SAT solver.

    Total objective = alpha * n_layers + beta * total_attn_heads +
                      gamma * total_mlp_bypass_slots

    Defaults: alpha=1, beta=0, gamma=0 — pure layer minimization.
    Tie-broken arbitrarily; in flex_routing mode CP-SAT picks
    whatever routing minimizes layers.

    `beta` knob (long-sequence regime). Attention compute scales
    roughly as O(L^2) per head (QK^T plus softmax-V), MLP compute
    scales as O(L) per layer. So per-token cost has shape:
        attn_cost  = total_attn_heads * d_head * L
        mlp_cost   = n_layers * (full d * d_hidden matmul)
    For large L, attn dominates and `beta > 0` pushes routing toward
    MLP bypass. As a rough starting point, beta ≈ L (heads-per-layer
    of attn become "as expensive" as one layer when L = beta).

    `gamma` knob (MLP bypass slot pressure). Less commonly useful
    because the per-layer matmul cost is the same whether MLP slots
    are 1% or 100% used (zero-padded slots still cost the full
    d * d_hidden multiply). Provided so you can express "I want to
    push routing toward attn" or "every used slot costs me X" if
    that maps to your deployment cost model.
    """

    alpha: int = 1
    beta: int = 0
    gamma: int = 0


@dataclass
class SolveResult:
    status_name: str
    n_layers: int
    total_attn_heads: int
    total_mlp_bypass_slots: int
    objective_value: int
    wall_time: float
    solver_log: str
    node_to_layer: Dict[int, int]
    node_to_cancel_layer: Dict[int, int]
    node_to_routing: Dict[int, str]   # final routing (post-solve), per-node
    flex_routing: bool
    is_optimal: bool


@dataclass
class HintSchedule:
    """Heuristic baseline for warm-starting CP-SAT. Maps node_id -> layer."""

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
    """Run forward_compile under `policy` and return its node->layer map.

    Used as an `AddHint` for CP-SAT so the solver starts from a known
    feasible incumbent rather than building one from scratch.
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


def solve(
    gm: GraphModel,
    d: int,
    d_head: int,
    d_hidden: int,
    policy: SchedulingPolicy,
    max_layers: int,
    costs: Costs = Costs(),
    flex_routing: bool = False,
    time_budget_s: float = 60.0,
    hint: Optional[HintSchedule] = None,
    log_search_progress: bool = True,
) -> SolveResult:
    """Build and solve the CP-SAT model.

    Objective: minimize `costs.alpha * n_layers + costs.beta * total_attn_heads
    + costs.gamma * total_mlp_bypass_slots`.

    `flex_routing=False` (default): every node has a fixed routing per
    `policy` (matches the heuristic). Layer-min LB is the proven minimum
    layer count under that policy's routing.

    `flex_routing=True`: each `is_flex(n)` node (standalone Linear) gets
    a CP-SAT BoolVar choosing ATTN or MLP. The objective drives the
    routing trade-off — `costs.beta` penalizes attn heads (helpful when
    sequence length makes attention expensive).

    Cumulative constraints: combined attn-heads + cancel-cols (capacity
    = n_heads * d_head per layer), MLP slots (capacity = d_hidden),
    residual columns (capacity = d - input cols). Per-edge dependency
    is `layer[v] >= layer[u]` if u is attn-routed and v mlp-routed
    (same-layer attn->mlp ok), otherwise strict.
    """
    n_heads_per_layer = d // d_head
    input_residual = sum(len(n) for n in gm.input_nodes)
    if gm.pos_encoding not in set(gm.input_nodes):
        input_residual += len(gm.pos_encoding)
    available_residual = d - input_residual
    if available_residual <= 0:
        raise ValueError(
            f"Inputs alone require {input_residual} cols, but d={d}. "
            f"No room for intermediate nodes."
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
    # is_attn[n] == 1 means the node runs in the attention sublayer at
    # its layer; is_attn[n] == 0 means it runs in MLP. For non-flex
    # nodes (or when flex_routing=False), the value is pinned.
    is_attn: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        if flex_routing and is_flex(n, gm):
            v = model.NewBoolVar(f"is_attn_n{n.node_id}")
        else:
            # Non-flex: hard-pin via a constant-domain bool.
            r = routing(n, gm, policy)
            v = model.NewBoolVar(f"is_attn_n{n.node_id}_pinned")
            if r == ATTN:
                model.Add(v == 1)
            else:
                model.Add(v == 0)
        is_attn[n.node_id] = v

    # ---- Dependency constraints ----
    # Edge u->v: same-layer ok iff u is_attn AND v is mlp (i.e., NOT v is_attn).
    # Otherwise layer[v] > layer[u].
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
    cancel_layer: Dict[int, cp_model.IntVar] = {}
    for n in gm.schedulable:
        cl = model.NewIntVar(0, max_layers, f"cl_n{n.node_id}")
        cancel_layer[n.node_id] = cl
        model.Add(cl >= layer_var[n.node_id] + 1)
        if n in gm.pinned_nodes:
            model.Add(cl == max_layers)
            continue
        keep_forever = False
        for c in gm.consumers_eff.get(n, set()):
            if isinstance(c, Concatenate):
                model.Add(cl == max_layers)
                keep_forever = True
                break
            if c.node_id in layer_var:
                model.Add(cl >= layer_var[c.node_id] + 1)
        if keep_forever:
            continue

    # ---- Combined attn-heads + cancel-cols cumulative ----
    # Per-node attn interval is OPTIONAL (gated by is_attn[n]) when the
    # node could run in either sublayer; for pinned nodes, the bool is
    # constant and CP-SAT presolve drops the unreachable branch.
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
    for n in gm.schedulable:
        if n in gm.pinned_nodes:
            continue
        c_end = model.NewIntVar(1, max_layers + 1, f"cend_n{n.node_id}")
        model.Add(c_end == cancel_layer[n.node_id] + 1)
        iv = model.NewIntervalVar(
            cancel_layer[n.node_id], 1, c_end, f"civ_n{n.node_id}"
        )
        cancel_intervals.append(iv)
        cancel_demands.append(len(n))

    if attn_intervals or cancel_intervals:
        model.AddCumulative(
            attn_intervals + cancel_intervals,
            attn_demands + cancel_demands,
            n_heads_per_layer * d_head,
        )

    # ---- MLP slots cumulative ----
    # For flex nodes, MLP demand is gated by NOT(is_attn).
    # Chain composites and standalone ReLUs are always MLP-routed.
    mlp_intervals: List = []
    mlp_demands: List[int] = []
    for n in gm.schedulable:
        s = slots_for(n, gm)
        if s <= 0:
            continue
        end = model.NewIntVar(1, max_layers, f"mend_n{n.node_id}")
        model.Add(end == layer_var[n.node_id] + 1)
        # is_mlp = NOT is_attn
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
    # uses_residual is a static fact (chain.exclusive, etc.), so no
    # routing dependence here.
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
    # total_attn_heads = sum over attn-routed nodes of heads_for(n).
    # total_mlp_bypass_slots = sum over (flex && mlp-routed) of 2*d_output.
    # (Chain composite slots are constant; standalone ReLU slots are constant.
    # Only the FLEX choice contributes to the objective term.)
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
        0, fixed_attn_heads + sum(heads_for(n, d_head) for n in gm.schedulable),
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
        0, fixed_mlp_bypass + sum(2 * n.d_output for n in gm.schedulable if is_flex(n, gm)),
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

    if costs.alpha == 0 and costs.beta == 0 and costs.gamma == 0:
        raise ValueError("alpha=beta=gamma=0 — no objective.")
    model.Minimize(
        costs.alpha * n_layers_var
        + costs.beta * total_attn_heads
        + costs.gamma * total_mlp_bypass
    )

    # ---- Hint ----
    if hint is not None:
        for nid, L in hint.node_to_layer.items():
            if nid in layer_var and 0 <= L < max_layers:
                model.AddHint(layer_var[nid], L)

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
    solver.parameters.num_search_workers = 8

    log_buf: List[str] = []

    class _Logger:
        def __call__(self, msg: str) -> None:
            log_buf.append(msg)

    solver.log_callback = _Logger()

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    node_to_layer: Dict[int, int] = {}
    node_to_cancel_layer: Dict[int, int] = {}
    node_to_routing: Dict[int, str] = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for n in gm.schedulable:
            node_to_layer[n.node_id] = solver.Value(layer_var[n.node_id])
            node_to_cancel_layer[n.node_id] = solver.Value(cancel_layer[n.node_id])
            node_to_routing[n.node_id] = (
                ATTN if solver.Value(is_attn[n.node_id]) else MLP
            )
        n_layers = solver.Value(n_layers_var)
        total_heads = solver.Value(total_attn_heads)
        total_bypass = solver.Value(total_mlp_bypass)
        obj = int(solver.ObjectiveValue())
    else:
        n_layers = -1
        total_heads = -1
        total_bypass = -1
        obj = -1

    return SolveResult(
        status_name=solver.StatusName(status),
        n_layers=n_layers,
        total_attn_heads=total_heads,
        total_mlp_bypass_slots=total_bypass,
        objective_value=obj,
        wall_time=elapsed,
        solver_log="\n".join(log_buf),
        node_to_layer=node_to_layer,
        node_to_cancel_layer=node_to_cancel_layer,
        node_to_routing=node_to_routing,
        flex_routing=flex_routing,
        is_optimal=status == cp_model.OPTIMAL,
    )


# ---------------------------------------------------------------------------
# Synthetic-graph sanity check (B3)
# ---------------------------------------------------------------------------


def synthetic_check() -> None:
    """A small, hand-checkable graph: an L->R->L chain on InputNode `in1`,
    then an Add with another InputNode `in2`, then output via Linear.

    With d=256, d_head=16, d_hidden=64: chain at layer 0, Add at layer 1,
    output Linear at layer 2 — 3 layers total. The chain dependency
    serializes them; nothing else fits in parallel because the only other
    schedulable nodes ARE this chain. So the optimum is dictated by the
    critical-path length and we can verify it.

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
        # L1 -> R -> L2 chain on in1.
        W1 = torch.randn(d_in, d_hidden) * 0.1
        l1 = Linear(in1, W1)
        relu = ReLU(l1)
        W2 = torch.randn(d_hidden, d_in) * 0.1
        l2 = Linear(relu, W2)
        # Add l2 + in2.
        s = Add(l2, in2)
        # Output: Linear over the add.
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
        print(f"    chain c{c.chain_id}: L1={c.l1.node_id} R={c.relu.node_id} "
              f"L2={c.l2.node_id} exclusive={c.exclusive} width={c.width}")

    res = solve(
        gm,
        d=256,
        d_head=16,
        d_hidden=64,
        policy=LEGACY_POLICY,
        max_layers=10,
        costs=Costs(alpha=1, beta=0),
        flex_routing=False,
        time_budget_s=30.0,
        hint=None,
        log_search_progress=False,
    )
    print(
        f"  layer-min: status={res.status_name}, "
        f"n_layers={res.n_layers}, heads={res.total_attn_heads}, "
        f"wall={res.wall_time:.2f}s"
    )
    print("  per-node layer:")
    for n in gm.schedulable:
        L = res.node_to_layer.get(n.node_id, "?")
        ann = type(n).__name__
        print(f"    n{n.node_id:>3} {ann:<14} d={len(n):>3}  layer={L}")

    assert res.is_optimal, f"expected OPTIMAL, got {res.status_name}"
    # Chain must run before the Add (L2 -> s) and Add before output (s -> out).
    # That's a strict 3-layer chain in the dependency DAG (chain=L0, Add=L1,
    # output=L2). The optimum is exactly 3 layers.
    assert res.n_layers == 3, f"expected 3 layers, got {res.n_layers}"
    print("  PASS")
    print()


# ---------------------------------------------------------------------------
# Full DOOM run (B4)
# ---------------------------------------------------------------------------


def build_doom_model() -> Tuple[GraphModel, int, int, int]:
    """Build the headless DOOM graph model and return (gm, d_head, d, d_hidden)
    with the standard headless geometry (d=3072, d_hidden=8192).
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

    # Baseline hints from the heuristic.
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
        print(f"--- Layer-min CP-SAT, policy={policy_name} (static routing) ---", flush=True)
        res = solve(
            gm,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
            policy=policy,
            max_layers=max_layers,
            costs=Costs(alpha=1, beta=0),
            flex_routing=False,
            time_budget_s=time_budget_s,
            hint=hint,
            log_search_progress=False,
        )
        print(
            f"  status: {res.status_name}, "
            f"n_layers={res.n_layers}, "
            f"total_attn_heads={res.total_attn_heads}, "
            f"wall={res.wall_time:.1f}s",
            flush=True,
        )
        print(f"  vs. heuristic: {hint.n_layers} layers", flush=True)
        static_results.append((policy_name, res, hint.n_layers))

    # ---- Flex-routing Pareto sweep ----
    # Default sweep: beta = 0, 1, 10, 100, 1000.
    # beta=0 -> pure layer-min (CP-SAT picks routing freely);
    # beta=1000 -> push routing strongly toward MLP.
    if pareto_betas is None:
        pareto_betas = [0, 1, 10, 100, 1000]

    print()
    print("=" * 70)
    print("FLEX-ROUTING PARETO SWEEP")
    print("Objective: alpha=1 * n_layers + beta * total_attn_heads")
    print("=" * 70)

    pareto_results: List[Tuple[int, "SolveResult"]] = []
    # Use the better of the two static hints for warm-starting flex.
    flex_hint = legacy_hint if legacy_hint.n_layers <= default_hint.n_layers else default_hint
    for beta in pareto_betas:
        print()
        print(f"--- beta={beta} (flex routing) ---", flush=True)
        res = solve(
            gm,
            d=d,
            d_head=d_head,
            d_hidden=d_hidden,
            policy=LEGACY_POLICY,  # only used for non-flex pinning; ignored when flex_routing=True
            max_layers=max_layers,
            costs=Costs(alpha=1, beta=beta),
            flex_routing=True,
            time_budget_s=time_budget_s,
            hint=flex_hint,
            log_search_progress=False,
        )
        print(
            f"  status: {res.status_name}, "
            f"n_layers={res.n_layers}, "
            f"total_attn_heads={res.total_attn_heads}, "
            f"wall={res.wall_time:.1f}s",
            flush=True,
        )
        pareto_results.append((beta, res))

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Static routing (per heuristic policy):")
    print(f"  {'policy':<24} {'heuristic':>10} {'CP-SAT LB':>10} {'attn heads':>11}")
    for name, res, h_layers in static_results:
        print(f"  {name:<24} {h_layers:>10} {res.n_layers:>10} {res.total_attn_heads:>11}")
    print()
    print("Flex routing Pareto sweep (CP-SAT picks per-Linear routing):")
    print(f"  {'beta':>6} {'n_layers':>10} {'attn_heads':>11} {'mlp_bypass_slots':>18}")
    for beta, res in pareto_results:
        opt_marker = "" if res.is_optimal else " (FEASIBLE)"
        print(
            f"  {beta:>6} {res.n_layers:>10} {res.total_attn_heads:>11} "
            f"{res.total_mlp_bypass_slots:>18}{opt_marker}"
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
