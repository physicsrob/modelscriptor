# Forward Compiler Implementation Plan

## Context

The backward compiler works from outputs toward inputs, creating "zero nodes" at every skip connection that accumulate and exhaust the residual stream. The 3-digit adder fails — 14+ layers without terminating, 40+ zero nodes consuming 300+ dimensions vs d=256.

The forward compiler works from inputs toward outputs. It starts with a known-empty residual stream and builds up, computing nodes when their inputs become available and cancelling dead nodes to reclaim space. This eliminates zero nodes entirely, replaces the CP-SAT constraint solver with a greedy column allocator, and replaces the beam search with local scheduling decisions.

**End goal**: Compile the 3-digit adder (`examples/adder.py`) and verify arithmetic correctness.

## Architecture

New module: `modelscriptor/compiler/forward/` (does NOT modify existing backward compiler).

Four new abstractions:
1. **GraphAnalyzer** — pre-computes reverse deps, topological order, critical path lengths
2. **ResidualStreamMap** — set-based column allocator (any free columns, no contiguity required)
3. **WeightWriter** — writes weight matrices into TransformerLayer components given scheduling decisions
4. **LayerScheduler** — decides what to compute/cancel at each layer

Reuses existing: `HeadlessTransformer`, `TransformerLayer`, `AttnSubLayer`, `FFNSubLayer`, all component classes (weight matrix shapes and forward passes), `FeatureAssignment` (as output bridge for `compute()`).

## Key Design Decisions

**No contiguity requirement.** Nodes are assigned arbitrary free columns, not contiguous ranges. Weight matrices scatter/gather via index lists, so physical adjacency is never needed. This eliminates fragmentation entirely.

**Add nodes are deferred.** An Add(A,B) is only computed when one addend is dead (all its other consumers computed). The dead addend's columns are reused via `add_into`. Since the graph is a DAG, this always terminates — non-Add ready nodes can always be computed, eventually freeing addends.

**Cancel only via attention.** Negation costs 1 head per node (for nodes with `len(node) <= d_head`). FFN cancel is omitted — attention heads are plentiful (64 per layer at d=1024, d_head=16) and all nodes in the adder graph are <= 16 columns wide.

## Key Operations

The transformer skip connection `out = in + f(in)` enables three primitives:

| Operation | Mechanism | Cost |
|-----------|-----------|------|
| **Write** to free columns | `0 + f(in) = f(in)` | 1 head or FFN slots |
| **Add** into dead addend's columns | `dead + live = Add(dead,live)` via skip | 1 head |
| **Cancel** dead node | `v + (-v) = 0` | 1 head |

**Biased Linear** = zero-bias Linear via attention head + bias via FFN `linear2.output_bias` in same layer.

---

## Phase 0: Graph Analysis

**Goal**: Build graph metadata needed by the scheduler.

**Create**:
- `modelscriptor/compiler/forward/__init__.py`
- `modelscriptor/compiler/forward/graph_analysis.py`
- `tests/compile/forward/__init__.py`
- `tests/compile/forward/test_graph_analysis.py`

**GraphAnalyzer class** (`graph_analysis.py`):
- `__init__(output_node, input_nodes)` — builds reverse dep map, topo order, critical paths
- `get_consumers(node) -> Set[Node]` — reverse dependency map
- `get_topological_order() -> List[Node]` — inputs first
- `get_critical_path_length(node) -> int` — longest chain to output
- `get_all_nodes() -> Set[Node]` — reuses `get_ancestor_nodes` from `compiler/utils.py`
- `is_input_node(node) -> bool` — isinstance check for Embedding/PosEncoding/InputNode/Constant
- `is_ready(node, available) -> bool` — all inputs (resolving through Concatenate) in available set
- `get_ready_nodes(available) -> Set[Node]` — all nodes ready to compute

Concatenate nodes are transparent — never placed in the residual stream, resolved to leaf children via `simplify_nodes` from `compiler/feature_assignment.py`.

**Tests**:
1. `test_simple_chain` — Input->Linear->ReLU->Linear: verify topo order, critical path (3,2,1,0), consumers
2. `test_diamond_graph` — Input->A, Input->B, Add(A,B): verify readiness progression
3. `test_concatenate_transparency` — Concat([A,B])->Linear: readiness depends on A,B not Concat
4. `test_adder_graph` — Load 3-digit adder, verify topo order valid, all nodes reachable
5. `test_ready_nodes_progression` — iteratively add ready nodes, verify all eventually available

**Gate**: All tests pass.

---

## Phase 1: Residual Stream Map

**Goal**: Set-based column allocator. No contiguity required — eliminates fragmentation.

**Create**:
- `modelscriptor/compiler/forward/residual_map.py`
- `tests/compile/forward/test_residual_map.py`

**ResidualStreamMap class**:
- `__init__(d: int)` — initializes free set `{0, 1, ..., d-1}`
- `allocate(node) -> List[int]` — pops `len(node)` columns from free set, returns sorted list. Raises if insufficient.
- `free(node)` — returns columns to free set
- `reassign(old_node, new_node)` — transfers columns without freeing (for add_into)
- `get_indices(node) -> List[int]`
- `is_allocated(node) -> bool`
- `get_free_count() -> int`
- `get_allocated_nodes() -> Set[Node]`
- `build_feature_assignment(in_state, out_state, input_nodes, output_node) -> FeatureAssignment`

**`build_feature_assignment`** creates a FeatureAssignment with exactly two states populated:
- `in_state` with all `input_nodes` at their allocated columns (read by `get_input_res_stream`)
- `out_state` with `output_node` at its allocated columns (read by `compute`)

Intermediate states (each sublayer's in/out) don't need population — the forward compiler writes weight matrices directly using ResidualStreamMap indices, bypassing `apply_strategy`.

**Tests**:
1. `test_allocate_and_free` — allocate d=8 node in d=64, verify indices, free, verify recovery
2. `test_multiple_allocations` — 3 nodes, verify non-overlapping index sets
3. `test_full_stream` — fill d=64 exactly, verify next allocation raises
4. `test_reassign` — allocate A, reassign to B, verify B has A's old columns
5. `test_build_feature_assignment` — allocate nodes, build FeatureAssignment, verify `get_node_indices` correct
6. `test_no_fragmentation` — allocate A(8), B(8), C(8), free B, allocate D(16): succeeds because columns need not be contiguous

**Gate**: All tests pass.

---

## Phase 2: Weight Writer

**Goal**: Given scheduling decisions + column assignments, write weights into a TransformerLayer's components. This is the bridge from abstract operations to physical weight matrices.

**Create**:
- `modelscriptor/compiler/forward/weight_writer.py`
- `tests/compile/forward/test_weight_writer.py`

**Operations** (dataclasses):
```python
@dataclass
class AttnHeadOp:
    op_type: str  # "compute_attn", "compute_linear", "cancel", "add_into"
    node: Node
    target_cols: List[int]

@dataclass
class FFNOp:
    op_type: str  # "compute_relu", "compute_constant", "compute_bias"
    node: Node
    target_cols: List[int]
    ffn_slots: List[int]  # which internal FFN dimensions this op uses
```

**Functions**:
- `write_attn_sublayer(layer, ops, residual_map, pos_encoding)` — sets Q/K/V/O matrices on `layer.attn.attn` (the AttnLayerComponent)
- `write_ffn_sublayer(layer, ops, residual_map)` — sets linear1/linear2 matrices and bias on `layer.ffn.linear1`, `layer.ffn.linear2`

Weight matrix shapes (from existing components):
- `layer.attn.attn.query_matrix[head, :, :]` — shape (d, d_head)
- `layer.attn.attn.output_matrix[head, :, :]` — shape (d_head, d)
- `layer.ffn.linear1.output_matrix` — shape (d, d)
- `layer.ffn.linear2.output_bias` — shape (d,)

**How each operation writes weights**:

*Attention — compute_attn*: Copy Attn node's Q/K/V/O matrices into one head, scattering to correct columns via `residual_map.get_indices()` for each input node. If node's `d_head < layer d_head`, pad with zeros (matching `components/attn.py:150` pattern).

*Attention — compute_linear (zero-bias)*: Q/K matrices use pos_encoding columns + `attention_hardness` for current-position attention. V reads from input node's columns. O applies node's weight matrix to target columns. Requires `len(node) <= d_head`.

*Attention — cancel*: Current-position attention. V=identity on node's columns, O=-identity on same columns. Requires `len(node) <= d_head`.

*Attention — add_into(Add, dead_addend, live_addend)*: Current-position attention. V reads live_addend's columns, O writes to dead_addend's columns. Skip adds: `dead + live = Add(dead, live)`. Requires `len(live_addend) <= d_head`.

*FFN — compute_relu*: For a `Linear1 -> ReLU -> Linear2` chain: `linear1.output_matrix` reads from input columns to FFN slots (applying Linear1's weight), `linear1.output_bias` at those slots from Linear1's bias, `linear2.output_matrix` maps from FFN slots to target columns (applying Linear2's weight), `linear2.output_bias` at target columns from Linear2's bias.

*FFN — compute_constant*: Set `linear2.output_bias[target_cols] = constant.value`.

*FFN — compute_bias*: Set `linear2.output_bias[target_cols] += bias_vector`. Used for the bias part of biased Linear nodes (zero-bias part computed via attention in same layer).

**Tests** — each builds a small graph, creates one TransformerLayer, writes weights, runs the forward pass, and verifies output with `torch.allclose` against `node.compute()`:

1. `test_attn_compute` — Attn node compiled on one head, verify output
2. `test_linear_zero_bias` — Linear(zero-bias) compiled via attention head
3. `test_cancel` — place values, cancel them, verify columns become zero
4. `test_add_into` — Add(A,B) where A is dead, write B to A's columns, verify A+B
5. `test_ffn_relu_chain` — Linear->ReLU->Linear compiled on FFN
6. `test_ffn_constant` — Constant via bias, verify values appear
7. `test_biased_linear_split` — Linear(bias!=0) split: attention for Wx + FFN for b, verify Wx+b

**Gate**: All 7 tests pass.

---

## Phase 3: Layer Scheduler

**Goal**: Given current residual stream state and graph metadata, decide what to compute/cancel in one layer.

**Create**:
- `modelscriptor/compiler/forward/scheduler.py`
- `tests/compile/forward/test_scheduler.py`

**LayerScheduler class**:
- `__init__(graph, d, d_head, pos_encoding)`
- `schedule_layer(residual_map, computed_nodes) -> (List[AttnHeadOp], List[FFNOp])`

**Algorithm per layer**:

```
1. Identify:
   ready      = nodes whose inputs are all in computed_nodes
                (exclude Add where neither addend is dead — these are deferred)
   dead       = nodes in residual_map whose consumers are all in computed_nodes
   free_adds  = Add nodes where one addend is dead and the other is computed

2. Attention sublayer (budget: n_heads):
   a. Free Adds (add_into) — highest priority, saves space
   b. Attn nodes from ready — must use attention, critical-path priority
   c. Zero-bias Linear nodes from ready (len <= d_head) — critical-path priority
   d. Biased Linear nodes from ready (len <= d_head) — attention for Wx part
   e. Cancellations of dead nodes — largest first
   If allocation fails for a ready node, promote cancellations above (b-d)

3. FFN sublayer (budget: d internal slots):
   a. Linear->ReLU->Linear chains from ready — critical-path priority
   b. Constants from ready — free (bias only, no slot cost)
   c. Bias writes for biased Linears scheduled in step 2d — no slot cost

4. Update residual_map and computed_nodes

5. Progress check: if no nodes computed or cancelled this layer, raise error
```

**Tests**:
1. `test_schedule_attn_node` — Attn node gets assigned to attention head
2. `test_schedule_relu_chain` — L->R->L gets assigned to FFN
3. `test_schedule_cancellation` — dead node gets cancelled
4. `test_schedule_free_add` — Add with dead addend becomes add_into
5. `test_schedule_deferred_add` — Add with no dead addend is NOT scheduled
6. `test_schedule_critical_path` — longer chain scheduled before shorter

**Gate**: All 6 tests pass.

---

## Phase 4: Forward Compiler (Small Graphs)

**Goal**: Wire everything together. Test on small graphs that the backward compiler handles.

**Create**:
- `modelscriptor/compiler/forward/compile.py`
- `tests/compile/forward/test_forward_compile.py`

**`forward_compile()` function**:
```python
def forward_compile(d, d_head, output_node, pos_encoding=None,
                    verbose=True, max_layers=100) -> HeadlessTransformer:
    # 1. Analyze graph
    graph = GraphAnalyzer(output_node)
    input_nodes = [n for n in graph.get_all_nodes() if graph.is_input_node(n)]

    # 2. Initialize
    net = HeadlessTransformer(d, d_head, pos_encoding)
    residual_map = ResidualStreamMap(d)
    for node in input_nodes:
        residual_map.allocate(node)
    computed = set(input_nodes)
    scheduler = LayerScheduler(graph, d, d_head, pos_encoding)

    # 3. Layer loop
    for i in range(max_layers):
        if output_node in computed:
            break
        layer = net.add_layer(end=True)
        attn_ops, ffn_ops = scheduler.schedule_layer(residual_map, computed)
        write_attn_sublayer(layer, attn_ops, residual_map, pos_encoding)
        # update residual_map and computed from attn_ops
        write_ffn_sublayer(layer, ffn_ops, residual_map)
        # update residual_map and computed from ffn_ops

    # 4. Build FeatureAssignment bridge
    net.feature_assignment = residual_map.build_feature_assignment(
        in_state=net.layers[0].attn.in_state,
        out_state=net.layers[-1].ffn.out_state,
        input_nodes=input_nodes,
        output_node=output_node,
    )
    return net
```

**Tests** — mirror existing `test_compile_ffn_network.py` patterns:
1. `test_compile_constant` — single Constant
2. `test_compile_linear` — Input->Linear (with bias)
3. `test_compile_relu_chain` — Input->Linear->ReLU->Linear
4. `test_compile_add` — Add(input1, input2)
5. `test_compile_select` — select(cond, true, false) pattern
6. `test_compile_get_last_value` — pos_encoding.get_last_value() (Attn node)
7. `test_compile_map_to_table` — table lookup (large FFN)

Each test:
```python
expected = output_node.compute(n_pos, input_values)
net = forward_compile(d=256, d_head=16, output_node=output_node, ...)
result = net.compute(n_pos, input_values)
assert torch.allclose(expected, result[output_node], atol=1e-4)
```

**Gate**: All 7 tests pass, plus all existing tests still pass (`pytest tests/ -v`).

---

## Phase 5: Adder Tests

**Goal**: Compile 1-digit and 3-digit adders, verify arithmetic correctness.

**Create**:
- `tests/compile/forward/test_forward_adder.py`

**Note**: `adder.py` uses module-level `max_digits = 3`. Tests monkeypatch `adder.max_digits` before calling `create_network_parts()` for 1-digit tests.

**Tests**:
1. `test_1digit_adder` — compile with max_digits=1, verify "1+1=" -> "2", "4+5=" -> "9", etc.
2. `test_1digit_layer_count` — verify layer count is reasonable (expect ~5-8 layers)
3. `test_3digit_adder` — compile with max_digits=3, verify "1+1="→"2", "123+456="→"579"
4. `test_3digit_autoregressive` — run autoregressively, verify complete output sequences
5. `test_3digit_resource_usage` — log layers used, peak column utilization

**Sizing**: `d=1024, d_head=16` gives 64 heads per layer. The 200-entry digit addition tables (from `map_to_table`) each need 200 internal FFN slots, so ~5 per FFN sublayer at d=1024.

**Gate**: Both adders compile and produce correct arithmetic.

---

## Files Summary

| File | Phase |
|------|-------|
| `modelscriptor/compiler/forward/__init__.py` | 0 |
| `modelscriptor/compiler/forward/graph_analysis.py` | 0 |
| `modelscriptor/compiler/forward/residual_map.py` | 1 |
| `modelscriptor/compiler/forward/weight_writer.py` | 2 |
| `modelscriptor/compiler/forward/scheduler.py` | 3 |
| `modelscriptor/compiler/forward/compile.py` | 4 |
| `tests/compile/forward/__init__.py` | 0 |
| `tests/compile/forward/test_graph_analysis.py` | 0 |
| `tests/compile/forward/test_residual_map.py` | 1 |
| `tests/compile/forward/test_weight_writer.py` | 2 |
| `tests/compile/forward/test_scheduler.py` | 3 |
| `tests/compile/forward/test_forward_compile.py` | 4 |
| `tests/compile/forward/test_forward_adder.py` | 5 |

**No existing files modified.**

## Verification

After each phase: `pytest tests/compile/forward/ -v`
After Phase 4+: `pytest tests/ -v` (ensure no regressions)
After Phase 5: Full adder arithmetic verification + resource usage logging
