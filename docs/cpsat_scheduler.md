# cpsat_scheduler

Optimization-driven scheduler for `forward_compile`. Produces an optimal
placement of every graph node into transformer layers, an optimal
cancellation timing for every node's residual columns, and (optionally)
an optimal attention-versus-MLP routing for every standalone `Linear`.

## 1. Overview

`forward_compile` turns a computation graph into a
`HeadlessTransformer` (the compiler's output type — a transformer with
no embedding layer and no LM head, just a residual-stream and stacked
attention-plus-MLP layers) by walking the graph layer by layer,
placing nodes into either the attention sublayer (which costs attention
heads) or the MLP sublayer (which costs MLP hidden slots), and
cancelling residual columns when nodes become dead.

Two scheduler implementations exist:

- **`LayerScheduler`** (heuristic). Greedy, layer by layer, with no
  lookahead. Picks the next layer's contents based on local pressure
  (residual occupancy, critical-path length, ready set). Fast — no
  solver overhead — and good enough for many graphs.

- **`DirectedLayerScheduler`** (this subsystem). A subclass of
  `LayerScheduler` that takes a precomputed `ScheduleAssignment` from
  the CP-SAT solver in `cpsat_scheduler.py` and replays it. The solver
  considers all layers simultaneously and proves an optimal schedule
  under a configurable cost objective. Because `DirectedLayerScheduler`
  is a subclass that overrides only the macro decisions (which node
  goes in which layer, in which sublayer, when each is cancelled) and
  inherits the parent's allocator and op-emission code, every
  micro-decision and every runtime invariant the parent enforces
  (cancel batching, source column capture, dirty-bit tracking, the
  four allocator invariants I1–I4 documented in `CLAUDE.md`) holds
  unchanged.

The CP-SAT scheduler exists because the heuristic's local decisions —
which `Linear` goes to attention versus MLP, when to cancel a dead
node — are globally suboptimal on the `(layer count, attention head
count)` Pareto front. The solver enumerates that front under a
configurable cost objective. The user navigates it via
`Costs(alpha, beta, gamma)` (see §4); two notable points on it are
"layer-min" (matches or beats every heuristic policy on layer count)
and "heads-min" (matches the lowest-attention heuristic policy with
fewer layers).

## 2. Architecture

### Code map

- `torchwright/compiler/forward/cpsat_scheduler.py` — the solver.
  Exports `solve_schedule()` and `ScheduleAssignment`.
- `torchwright/compiler/forward/scheduler.py` — adds
  `DirectedLayerScheduler` next to the existing `LayerScheduler`.
- `torchwright/compiler/forward/compile.py` — adds the `use_cpsat`
  parameter to `forward_compile` and the cache for solver results.
- `scripts/cpsat_schedule.py` — probe script for ad-hoc Pareto
  exploration. Imports from the production module.

### Data flow

```
   computation graph
   (output_node, pos_encoding)
              │
              ▼
   ┌───────────────────────────────┐
   │  cpsat_scheduler.py           │
   │    solve_schedule(            │
   │      graph, d, d_head,        │
   │      d_hidden, costs,         │
   │      flex_routing,            │
   │      time_budget,             │
   │    )                          │
   └───────────────────────────────┘
              │
              ▼
   ScheduleAssignment
     node_to_layer
     node_to_cancel_layer
     node_to_routing
              │
              ▼
   ┌───────────────────────────────┐
   │  forward_compile(use_cpsat=…) │
   │    │                          │
   │    ▼                          │
   │  DirectedLayerScheduler       │
   │    .schedule_layer(L)         │
   │    for L in 0..n_layers       │
   └───────────────────────────────┘
              │
              ▼
   HeadlessTransformer
```

### `ScheduleAssignment`

The contract between the solver and the replay. A frozen dataclass:

```python
@dataclass(frozen=True)
class ScheduleAssignment:
    node_to_layer: Dict[int, int]
    node_to_cancel_layer: Dict[int, int]
    node_to_routing: Dict[int, str]   # "attn" or "mlp"
    n_layers: int
```

Every schedulable node — every non-`Concatenate`, non-input node in the
ancestor cone of `output_node` — appears in all three dicts.
(`Concatenate` is the graph's "view" op: it has no value of its own and
is never placed in the residual stream; consumers reference its leaves
directly.) When the solver returns `OPTIMAL` or `FEASIBLE`, the
assignment is fully populated and respects every constraint in §3. On
`INFEASIBLE` or unrecoverable time-out, no assignment is returned and
`forward_compile` raises (see §5).

The solver guarantees:

- `node_to_layer[n]` is the transformer layer where `n` executes.
- `node_to_cancel_layer[n]` is the layer where `n`'s residual columns
  are reclaimed. Set to `n_layers` for nodes that stay alive forever
  (inputs, output, output-cone leaves).
- `node_to_routing[n]` is either `"attn"` or `"mlp"` — which sublayer
  of `node_to_layer[n]` runs the op.

A **chain** is a triple `(L1, R, L2)` where `L1` is a `Linear`, `R` is
a `ReLU` reading from `L1`, and `L2` is a `Linear` reading from `R`,
with each link being the unique consumer/producer at that point. Chain
detection runs once at the start of `solve_schedule`; once detected, a
chain is **atomic** — the solver always schedules its three nodes into
one MLP sublayer at the same layer (`L1`'s compute happens inline
inside the MLP's `linear1`, `R` is the chain's hidden activation living
in `linear1`'s slots, and `L2` is the chain's residual-stream output).
**Chain-internal** nodes (any of `L1`, `R`, `L2` of a detected chain)
share a `node_to_layer` value and all carry routing `"mlp"`.

### `DirectedLayerScheduler`

Subclass of `LayerScheduler`. Three things change relative to the
heuristic; everything else inherits from the parent and runs unchanged.

What it overrides:

- **Ready filter.** Only nodes with `assignment.node_to_layer[n] ==
  current_layer` are eligible to schedule this layer. Nodes whose
  layer has not arrived yet stay deferred.
- **Routing.** Each `Linear` is forced into the attention sublayer or
  the MLP bypass per `assignment.node_to_routing[n]`. The
  `policy.local_in_attention` setting is ignored.
- **Cancellation.** At each layer `L`, cancels are queued for every
  node where `assignment.node_to_cancel_layer[n] == L`. The
  heuristic's eager freeing of dead nodes is suppressed.

What it preserves (by inheriting the parent's per-layer code path):

- Cancel coalescing into a single batched
  `AttnHeadOp("cancel", None, cancel_cols)`. Cancels queued by the
  override flow into the parent's existing batching machinery, which
  emits one cancel op per layer.
- Dirty-bit tracking and same-batch cancellation of dirty target
  columns from fresh allocations.
- Source column capture (`q_source_cols`, `k_source_cols`, etc.) via
  `_require_live` at schedule time.
- All four allocator invariants I1–I4 (see `CLAUDE.md`). These are
  runtime assertions inside `ResidualStreamMap` and the
  weight-writer, which `DirectedLayerScheduler` doesn't touch.

## 3. The optimization model

The solver builds a CP-SAT model from the graph and minimizes the
configured objective.

### Variables

Per schedulable node `n`:

- `layer_var[n]` ∈ [0, max_layers−1] — the transformer layer where
  `n` executes.
- `cancel_layer[n]` ∈ [layer_var[n]+1, max_layers] — the layer
  where `n`'s residual columns are reclaimed. The value `max_layers`
  is the sentinel for "never freed."
- `is_attn[n]` — boolean. Pinned to 1 for `Attn` and `Add`. Pinned to
  0 for standalone `ReLU`, `LiteralValue`, and chain-internal Linears.
  For standalone `Linear` nodes (Linears outside any chain) the value
  is free under `flex_routing=True` and pinned per `policy` under
  `flex_routing=False`.

Per chain (defined in §2) the three nodes' `layer_var` are constrained
equal — the chain runs as a single composite in the MLP sublayer of
its layer.

A chain's `L1` is **exclusive** if its only graph consumer is the
chain's `R`. An exclusive `L1` has no residual position of its own —
its value is computed inline inside `linear1` from its input's
residual columns and only ever lives in the MLP hidden slots. A
non-exclusive `L1` has additional consumers and so needs a residual
position alongside the chain.

### Dependency constraints

For each directed edge `u → v` in the graph (after walking
`Concatenate` inputs to their leaves):

```
same_layer_ok = is_attn[u] ∧ ¬is_attn[v]

if  same_layer_ok:  layer_var[v] ≥ layer_var[u]
else:               layer_var[v] ≥ layer_var[u] + 1
```

This encodes the within-layer sublayer ordering: attention writes
happen first within a layer, then MLP reads. The only same-layer
producer/consumer pair that fits is `u` writing in attention and `v`
reading in MLP.

### Resource cumulatives

Three `AddCumulative` constraints, one per resource pool. The cost
function `heads_for(n)` returns the integer number of attention heads
the op consumes if scheduled in the attention sublayer:
`⌈d_v/d_head⌉` for `Attn`, `⌈d_input/d_head⌉` for `Linear`,
`⌈d_output/d_head⌉` for `Add` (the optimistic free-add cost — see
*Model preconditions* below).

**Attention budget — heads plus cancel columns combined.** Capacity
`n_heads_per_layer · d_head` per layer.

- For each node `n` with attention cost `heads_for(n) > 0`: an
  optional unit-width interval at `layer_var[n]` gated by
  `is_attn[n]`. Demand `heads_for(n) · d_head` (the column footprint
  of those heads).
- For each non-pinned schedulable node `n`: a unit-width cancel
  interval at `cancel_layer[n]`. Demand `len(n)` columns.

The combined-column form `H_a · d_head + cancel_cols ≤ n_heads ·
d_head` is mathematically equivalent to the heuristic's per-layer
`H_a + ⌈cancel_cols/d_head⌉ ≤ n_heads`. Both `H_a` and `n_heads` are
non-negative integers, and `d_head > 0`. Forward: from `H_a · d_head +
cancel_cols ≤ n_heads · d_head`, divide by `d_head` and use the fact
that `(n_heads − H_a)` is an integer to conclude
`⌈cancel_cols/d_head⌉ ≤ n_heads − H_a`. Reverse: from
`H_a + ⌈cancel_cols/d_head⌉ ≤ n_heads`, multiply by `d_head` and use
`cancel_cols ≤ ⌈cancel_cols/d_head⌉ · d_head`.

**MLP slot budget.** Capacity `d_hidden` per layer.

- For each MLP-routed standalone `Linear` `n`: an optional unit-width
  interval at `layer_var[n]` gated by `¬is_attn[n]`. Demand
  `2 · n.d_output` (MLP bypass slots).
- For each standalone `ReLU` `n`: an optional unit-width interval
  gated by `¬is_attn[n]`. Demand `len(n)`.
- For each L1→ReLU→L2 chain: a regular unit-width interval at
  `layer_var[chain.relu]`, demand `len(chain.relu)` (the chain's
  hidden width).

**Residual column budget.** Capacity `d − input_residual_cols`,
where `input_residual_cols` is the sum of widths of pre-allocated
input nodes plus `pos_encoding`.

- For each residual-using node `n`: a regular interval
  `[layer_var[n], cancel_layer[n])`, demand `len(n)`.

`uses_residual(n)` is `False` for chain-internal `ReLU` (lives in
MLP hidden slots, not residual) and exclusive chain L1 (computed
inline inside `linear1` from its input's residual columns, never
written back to the stream). True for everything else.

### Objective

```
minimize  alpha · n_layers
        + beta  · total_attn_heads
        + gamma · total_mlp_bypass_slots
```

where:

- `n_layers = max(layer_var) + 1`.
- `total_attn_heads = Σ heads_for(n)` over attention-routed nodes.
  For pinned-attention nodes this is a constant; for flex Linears
  it depends on `is_attn[n]`.
- `total_mlp_bypass_slots = Σ 2 · n.d_output` over MLP-routed flex
  Linears. Chain composite slots and standalone ReLU slots are
  constants and therefore don't appear in the objective.

### Model preconditions

The heuristic `LayerScheduler` distinguishes two `Add` scheduling
modes: **free_add** (one input is dead — the `Add` reuses the dead
input's residual columns and only needs to copy the live input,
costing `⌈len(n)/d_head⌉` heads) and **compute_add** (both inputs
still alive — the `Add` allocates fresh columns and copies both
inputs, costing up to twice that). The heuristic also disallows
free_add when an input is a `Concatenate`, because `Concatenate` nodes
do not own residual columns and so cannot have their cols reused.

The model is sound for graphs and configurations satisfying:

- `admission_control=False` in `forward_compile`. The model does not
  represent the sibling-cluster admission constraint described in
  `torchwright/compiler/forward/sibling_clusters.py`. With admission
  control on, the solver may produce schedules the replay cannot
  honor.
- No `Add` node has a `Concatenate` input. The model uses the
  free-add cost for every `Add`. `solve_schedule` rejects graphs
  containing `Concatenate`-input `Add` nodes by raising at solve time.
- Chains are atomic. A chain `(L1, R, L2)` (defined in §2) is always
  scheduled as one MLP composite; the model does not consider
  splitting a chain into separate ops.
- Standalone Linears are the only flex-routing-eligible node type.
  `Attn`, `Add`, standalone `ReLU`, and `LiteralValue` have routing
  fixed by their type.

## 4. API

### `Costs`

```python
@dataclass(frozen=True)
class Costs:
    alpha: int = 1
    beta: int = 0
    gamma: int = 0
```

The objective is
`alpha · n_layers + beta · total_attn_heads + gamma · total_mlp_bypass_slots`.
All three are non-negative integers.

`alpha` weights the layer count. The default `alpha=1` always
penalizes adding a layer.

`beta` weights the total attention head count. Set this above zero
when sequence length makes attention compute expensive: per-token
attention compute scales as `O(L · d_head)` per head for sequence
length `L`, while per-layer compute (the full
`d × d_hidden` MLP matmul plus the `4 · d²` attention QKVO matmuls)
is independent of `L`. For long autoregressive sequences this means
attention dominates total compute and a small reduction in head
count pays back larger than a small reduction in layer count. As a
starting point, `beta ≈ L` makes one attention head equivalent to
one extra layer.

`gamma` weights MLP bypass slot usage. The per-layer MLP matmul
costs the full `d × d_hidden` regardless of how many slots are used
— zero-padded slots still pay — so `gamma = 0` is the normal case.
Provided so that deployments with deployment-time slot pruning can
express a non-trivial slot cost.

### `flex_routing`

Short for "flexible routing" — whether the standalone-`Linear`
sublayer choice (attention versus MLP-bypass) is a CP-SAT decision
variable rather than fixed by the policy.

When `True` (the default), each standalone `Linear` (a `Linear`
outside any chain) gets its own `is_attn` decision variable and the
solver picks attention versus MLP per node. When `False`, standalone
Linears are pinned per `policy.local_in_attention` and only the
placement and cancellation decisions are optimized.

`flex_routing=True` weakly dominates `flex_routing=False` on the
solver objective: anything `flex_routing=False` can produce is also
producible under `flex_routing=True`, and the larger search space
admits strictly better optima for objectives where the routing choice
matters. The `flex_routing=False` mode exists to support comparing
CP-SAT against a specific heuristic policy's routing choice.

### `forward_compile` integration

```python
forward_compile(
    d, d_head, output_node, pos_encoding,
    ...,
    use_cpsat: bool = True,
    cpsat_costs: Costs = Costs(),
    cpsat_flex_routing: bool = True,
    cpsat_time_budget_s: float = 60.0,
    cpsat_allow_suboptimal: bool = False,
)
```

`use_cpsat=True` is the default. The flow:

1. `solve_schedule` runs once, producing a `ScheduleAssignment`.
2. The assignment is cached on the cache key described in §5.
3. `DirectedLayerScheduler` consumes the assignment for the per-layer
   loop.
4. The compile produces a `HeadlessTransformer` with the same token
   semantics as the heuristic compile would have produced — the
   schedule is a placement decision, not a value-changing
   transformation.

`use_cpsat=False` falls back to the heuristic `LayerScheduler` and
runs `forward_compile` exactly as it did before this subsystem
existed. Use this to bypass the solver overhead in workflows where
compile latency matters more than schedule quality.

The `policy` argument is honored only when `cpsat_flex_routing=False`,
where it pins the routing of standalone Linears. With
`cpsat_flex_routing=True` (the default), `policy` is ignored.

`cpsat_allow_suboptimal=False` (the default) raises `RuntimeError`
when the solver returns a feasible-but-not-proven-optimal schedule.
Set to `True` to accept the suboptimal schedule and proceed with
the compile. See §5 for details.

## 5. Runtime behavior

### When the solver runs

`solve_schedule` runs at the start of `forward_compile`, before the
first layer is allocated. It does not run during the layer loop. The
layer loop is then deterministic: `DirectedLayerScheduler` reads the
assignment and emits ops.

### Caching

The solver result is cached keyed on
`(graph_hash, d, d_head, d_hidden, costs, flex_routing)`. Subsequent
compiles with the same key reuse the cached `ScheduleAssignment`. The
cache lives in process memory; cold compiles pay the solve cost once
per `(graph, geometry, costs)` tuple.

`graph_hash` is computed from the graph's node ID set, edge set, and
per-node widths. Mutating any of these between compiles invalidates
the cache.

### Time budget

`cpsat_time_budget_s` caps wall time per solve. The default of 60
seconds is sufficient to prove optimality for graphs with low
thousands of schedulable nodes at moderate dimensions. Larger
graphs or more sharply mixed cost trade-offs (small `beta` values
that produce dense Pareto fronts) may need a larger budget.

If the budget expires without a proven optimum, the solver returns
the best feasible solution found so far (status `FEASIBLE` instead
of `OPTIMAL`). The compile's response depends on
`cpsat_allow_suboptimal` — see *Failure modes* below.

### Determinism

CP-SAT runs with `num_search_workers=8` and uses parallel worker
strategies. Different runs may produce different `ScheduleAssignment`
values for the same model — different worker discovery orders find
different optima of equal objective value. The compiled
`HeadlessTransformer` differs across runs only in scheduling: token
outputs are bitwise identical (modulo float-point ordering effects
that already affect every compile, see *FP nondeterminism at
tolerance boundaries* in `CLAUDE.md`).

### Failure modes

**Precondition violation.** `solve_schedule` raises `RuntimeError`
*before* invoking CP-SAT if the graph or configuration violates a §3
precondition (admission control on, an `Add` with a `Concatenate`
input, etc.). No `ScheduleAssignment` is produced.

**`INFEASIBLE`.** The solver ran and proved no schedule fits the
constraints — typically because `max_layers` is too small for the
graph. No `ScheduleAssignment` is produced; `forward_compile` raises
`RuntimeError` carrying the solver status.

**Time limit exceeded with no feasible solution.** The solver
neither proved optimality nor found a feasible schedule within the
budget (CP-SAT status `UNKNOWN`). Same response as `INFEASIBLE` —
`RuntimeError`, no `ScheduleAssignment`.

**Time limit exceeded with feasible solution** (CP-SAT status
`FEASIBLE`). The solver returns a `ScheduleAssignment` that is
feasible but possibly non-optimal. By default, `forward_compile`
raises `RuntimeError` carrying the gap (`best_objective_bound` versus
`objective_value`) so callers know they did not get the proven
optimum. Setting `cpsat_allow_suboptimal=True` instead accepts the
schedule, logs the gap, and proceeds. The compiled model is correct
in either case; the only difference is whether the caller is willing
to ship a schedule that is possibly larger than necessary.
