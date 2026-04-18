# Affine bound propagation

Design of the affine-bound replacement for `Range` inside
`NodeValueType`. This document describes the completed system —
not the steps to get there.

---

## Why this exists

The scheduler compiles a computation graph into transformer weights.
Several graph-level ops derive numerical-stability constants from
their inputs' declared bounds:

- `cond_gate` picks its cancellation offset `M` from `max |input|`.
- `floor_int` builds its piecewise-linear breakpoint grid from the
  input's range.
- `assert_in_range` emits a runtime predicate and a static claim on
  the wrapped node.
- `attend_mean` sets `atol` from the value input's range.

Today, `NodeValueType.value_range` is a single `Range(lo, hi)` that
every component of a node's output shares. Each op's
`compute_value_type()` propagates it by scalar interval
arithmetic — if `u.value_range` is `[a, b]` and `v.value_range` is
`[c, d]`, then `(u + v).value_range` is `[a + c, b + d]`. This is
sound, but it treats every pair of inputs to every op as
independent. In a torchwright graph, inputs are rarely independent.
The residual stream carries the same node through many downstream
consumers; `cond_gate` adds `M` and a later `Add` subtracts it;
sum chains are frequently dominated by correlated or identical
operands.

The primary workaround already in the codebase lives in
`linear_output_range` (`graph/value_type.py:257-275`), the helper
called by `Linear.compute_value_type()`. It performs per-column
interval arithmetic over the weight matrix (each output
component's contribution computed independently, then the
componentwise min/max is aggregated into a single `Range`). This
recovers some tightness inside each `Linear`, but the per-component
information is lost at the aggregate `Range` boundary — downstream
`Linear`s consuming that output see only the scalar union, so the
recovered tightness doesn't compose through the graph.

The motivating cancellation pattern is `cond_gate`: add a large
offset `M`, do gated work, subtract `M`. The two `M` constants
should cancel exactly, but under scalar interval arithmetic each
side contributes `±M` independently, widening the output range by
`2M`. The compiler picks `M = 2 · max|input|` (a safety factor
baked into `_max_abs_or_raise`) — so an IBP-loose bound forces a
wider `M`, which in turn forces more numerical precision loss in
the gate's MLP. Tightening the bound tightens the constant.

Worked-example sketch. Say an input node `x` has declared range
`[-10, 10]`. The compiler picks `M = 20`. Inside `cond_gate` a
subgraph computes `y = ReLU(x + M) = ReLU(x + 20)` (always
non-negative since `x + 20 ∈ [10, 30]`), then `z = y - M = y - 20`.
The true value of `z` equals `x`. Under scalar interval arithmetic:
`y.value_range = [10, 30]`; `z.value_range = [10 - 20, 30 - 20] =
[-10, 10]` — correct in this simple case because the straddling
case is avoided, but the `±M` contributes in other `cond_gate`
patterns that do cross zero. Under affine bounds: `y.bounds` is the
affine expression `x + 20`; `z.bounds` subtracts the constant `20`
from the offset, yielding the affine expression `x`. Downstream
`Linear`s that consume `z` see the same `A` matrix as they would
for `x` directly — no accumulated looseness from the offset round
trip. This is the structural pattern that extends to every `Add`
whose operands share upstream structure.

This is the failure mode: scalar interval arithmetic can't see
cancellation between correlated operands.
Affine bound propagation fixes that at the abstraction layer —
`Linear`, `Add`, `ReLU`, and `Concatenate` all become exact (modulo
`ReLU`'s linear envelope — a pair of affine functions that
sandwich the true `ReLU` output when the input straddles zero; see
the `ReLU` rule below), and downstream constants come out tighter
as a consequence. The per-component min/max logic inside
`linear_output_range` becomes redundant.

### Why forward propagation

The bound-propagation literature distinguishes forward-mode (each
node's affine bound is computed from its parents' affine bounds,
cached, and reused by descendants) from backward-mode (given a
specific query node, walk back through its ancestors composing
relaxations to derive that node's bound). Backward-mode is tighter
per query — because it picks a relaxation optimized for the
specific output — but it re-walks the graph per query.

Torchwright's per-node `compute_value_type()` contract fits
forward-mode natively: bounds are computed once, at finalization,
and stored on each node where every bound-consumer op picks them up
by reading `node.value_type` at the materialization step. Backward
queries would pay re-propagation cost per consumer, which there
are many of (one per `cond_gate`, `floor_int`, `assert_in_range`,
`attend_mean` call in the compiled graph). Tighter-per-query
backward-mode is a real option to consider if forward bounds prove
insufficient in practice, but it would require restructuring the
finalization pass around query nodes and is deliberately not part
of v1.

---

## The basis

Every affine expression in the system is written over the same set
of variables, called **the basis**. The term is local to this
system; it is unrelated to the "residual-stream column basis" used
elsewhere in the codebase. The basis of affine bounds is the set
of every component of every `InputNode` in the graph.

Concretely, if the graph has `InputNode`s with widths
`d_1, d_2, ..., d_k`, the basis has `n = d_1 + d_2 + ... + d_k`
variables. Basis variable `j` corresponds to one specific component
of one specific `InputNode`, and it takes values in the range that
`InputNode`'s declared bounds give it (from `assert_in_range` on
the input, or an equivalent declaration at `InputNode`
construction).

The basis is fixed for the lifetime of the graph. It does not grow
as the graph gets deeper. It does not change when we encounter
`Attn` or any other op. Every `AffineBound` in the graph is
expressed over the same `Basis` object, so affine expressions
compose directly: row-wise addition for `Add`, matrix multiply for
`Linear`, and so on.

```python
@dataclass(frozen=True)
class Basis:
    x_lo: torch.Tensor    # shape [n]; per-variable lower bound
    x_hi: torch.Tensor    # shape [n]; per-variable upper bound
    # metadata mapping variable index → (InputNode, component)
```

### Position semantics

Torchwright's `compute_value_type()` is position-agnostic: it
produces one `NodeValueType` per node that describes every
component of the node's output at every token position. The affine
bound system inherits this. A basis variable represents the
per-position value of its corresponding `InputNode` component, with
an interval wide enough to hold every possible value at every
position. An affine expression `A[i, :] · x + b[i]` describes the
node's value at component `i` at *whatever position it is evaluated
at*, in terms of that same position's `InputNode`-component values.

Soundness follows pointwise: for any concrete position `p` and any
concrete sample of `InputNode` values at that position drawn from
the basis box, the node's per-position value at component `i` lies
in the interval derived from the affine expression at those input
values. Because every position draws from the same basis box, the
derived component interval covers every position simultaneously.

This is why `Attn` is the only op that cannot produce a non-zero
`A` matrix. The attention output at position `p` is a function of
`InputNode` values at positions `0, 1, ..., p` — not just position
`p` — and the affine framework has no way to express that
dependency in one position-local basis.

### Sizing

For a typical compiled program, `n` is on the order of tens to low
hundreds — game-state inputs, cursor positions, flags — not
thousands. Per-node storage is `[d_output × n]` floats per bound,
so typical cost per node stays modest. Bound coefficients are held
in `torch.float64` regardless of the transformer's forward-pass
dtype; scalar-interval error compounds over many `Linear` / `Add`
steps and the extra precision buys headroom for free. This
assumes weight matrices with bounded spectral norm (condition number
roughly O(1) per `Linear`), which is the regime compiled torchwright
graphs are built to stay in. Graphs that stack many `Linear`s with
large weight norms would see bound error accumulate geometrically
regardless of dtype; the compiler's numerical-stability discipline
— which is part of why this tightening effort matters — keeps
that regime out of scope.

---

## `AffineBound`

```python
@dataclass(frozen=True)
class AffineBound:
    """Per-component affine lower and upper bounds on a node's output.

    For a node with width ``d_output``, bounded over the graph's
    basis (see ``Basis``):

        lower(x)[i] = A_lo[i, :] · x + b_lo[i]
        upper(x)[i] = A_hi[i, :] · x + b_hi[i]

    with the invariant, for every x in the basis's declared box:

        lower(x)[i] ≤ node_output[i] ≤ upper(x)[i]
    """
    basis: Basis                # the graph's single basis
    A_lo: torch.Tensor          # shape [d_output, n]
    b_lo: torch.Tensor          # shape [d_output]
    A_hi: torch.Tensor          # shape [d_output, n]
    b_hi: torch.Tensor          # shape [d_output]
```

### Deriving per-component intervals

```python
# for each component i
lo[i] = b_lo[i] + Σ_j term_lo(A_lo[i, j], x_lo[j], x_hi[j])
hi[i] = b_hi[i] + Σ_j term_hi(A_hi[i, j], x_lo[j], x_hi[j])

# where
term_lo(a, lo, hi) = 0            if a == 0
                     a * lo       if a > 0
                     a * hi       if a < 0

term_hi(a, lo, hi) = 0            if a == 0
                     a * hi       if a > 0
                     a * lo       if a < 0
```

Two sign-split reductions. `AffineBound.to_interval()` returns
per-component `lo`, `hi` tensors (each of shape `[d_output]`).

The explicit `a == 0` branch is not cosmetic: the basis may contain
unbounded variables (`x_lo[j] = -inf` or `x_hi[j] = +inf`) for
inputs without declared ranges, and IEEE `0 * inf = NaN` would
corrupt every downstream bound. The short-circuit keeps zeroed
coefficients out of the sum. Vectorised form:

```python
term_lo = torch.where(A_lo > 0, A_lo * x_lo,
          torch.where(A_lo < 0, A_lo * x_hi, 0))
```

### API surface

`AffineBound` is a value object; instances are frozen. The public
API is intentionally small:

- **Factory methods.**
  `AffineBound.identity(basis, input_node)` — one-hot rows picking
  out the basis slice belonging to `input_node`; offsets zero.
  `AffineBound.constant(basis, values)` — zero `A`; `b_lo = b_hi =
  values`. `AffineBound.degenerate(basis, lo, hi)` — zero `A`;
  scalar `lo` and `hi` broadcast to `[d_output]`, used as the
  escape hatch when an op cannot compute affine bounds.
- **`to_interval()`** — per-component `(lo, hi)` tensors of shape
  `[d_output]`, computed via the sign-split formula above.
- **`__repr__`** — summary only (basis id, `d_output`, coarse
  stats of `A_lo` / `A_hi` magnitudes). The full coefficient
  tensors are not dumped; use direct attribute access when
  debugging.
- **Serialization.** `AffineBound`s are not persisted to the
  compiled artifact — ONNX export strips them, as they are
  compile-time metadata. Bounds can be recomputed from the graph
  by calling `finalize(root)` on a rebuilt graph.

Arithmetic (addition, matrix multiply, scalar scale) is not
exposed as operator overloads; the per-op rules in
*Per-op propagation rules* do the required math inline.

---

## The new `NodeValueType`

```python
@dataclass(frozen=True)
class NodeValueType:
    bounds: AffineBound

    # Structural claims about the output (retained from the prior design).
    is_integer: bool = False
    is_binary: bool = False   # implies is_integer and 0 ≤ value ≤ 1
    is_sign: bool = False     # implies is_integer and -1 ≤ value ≤ 1
    is_one_hot: bool = False  # vector-level; implies is_binary

    @property
    def value_range(self) -> Range:
        """Aggregate scalar interval covering every component.

        Derived from ``bounds`` as
        ``Range(bounds.to_interval().lo.min(),
                bounds.to_interval().hi.max())``.

        Preserves the pre-existing semantics: "every component of
        the node's output lies in this range" holds exactly as
        before, just derived now from ``bounds`` rather than stored
        independently.
        """
        ...
```

- `bounds` is the source of truth for numeric bounds. Every
  `compute_value_type()` produces an `AffineBound`; every rule that
  reads numeric bounds reads through `bounds`.
- `value_range` becomes a derived property. Every existing consumer
  that reads `node.value_type.value_range` keeps working; the
  returned `Range` is strictly inside (or equal to) the old one.
- The structural flags (`is_integer`, `is_binary`, `is_sign`,
  `is_one_hot`) are kept as plain booleans. They carry semantics
  that interval bounds alone cannot — e.g. `floor_int` skips
  transition handling on integer inputs; lookup ops require
  binary / one-hot inputs.
- The `Guarantee` enum is removed. All structural flags are
  treated as unconditional claims (the former `Guarantee.ALWAYS`
  semantics). The `Guarantee.APPROXIMATE` level and its
  transition-zone handling are dropped; ops that previously
  declared `APPROXIMATE` either declare no structural flag (if the
  claim was not load-bearing) or declare an explicit claim along
  with an `assert_in_range` that documents the tolerance.

Nodes whose rules can't (or won't) compute affine bounds carry a
degenerate `AffineBound` — zero `A_lo` and `A_hi`, and `b_lo` /
`b_hi` set to whatever scalar interval the rule decides on (or
±∞). This preserves the "unknown range" escape hatch and keeps
every op total.

### Removal of the `Guarantee` machinery

The `Guarantee` enum removal is a complete deletion, not a shim. In
the finished system:

- **`Guarantee`, `GuaranteeLevel`, `_min_guarantee`, `_max_guarantee`
  are gone.** Nothing in the codebase imports or mentions them.
- **Factory helpers** — `NodeValueType.integer(...)`,
  `.binary(...)`, `.sign(...)`, `.one_hot(...)`, and any others
  that previously accepted a `guarantee=` parameter — drop the
  parameter. Their behavior is the former `Guarantee.ALWAYS` path.
- **`tightened_with`** (the helper that merges an `Assert`'s
  `claimed_type` onto a wrapped node) simplifies: structural flags
  compose with plain boolean OR (either side may claim, and the
  claim holds); ranges intersect; no guarantee-level arithmetic.
- **`intersect_element_props`** is no longer called by
  `Concatenate.compute_value_type()`, which uses the exact
  row-stacking rule (see *Per-op propagation rules*). If the helper
  is still useful for other call sites it reduces to the same
  boolean-AND-of-flags composition; otherwise it is deleted.
- **`assert_matches_value_type`** (the runtime-verifier predicate)
  drops its `APPROXIMATE`-branch warnings. Every declared flag that
  the observed tensor violates is a hard `AssertionError`.
- **Op call sites that declared `Guarantee.APPROXIMATE`** are
  audited individually. The existing consumers of each
  `APPROXIMATE` flag are already conservative — e.g. `floor_int`'s
  use of `is_integer` is a correctness optimization that must not
  be invoked on near-integer-but-not-exactly-integer values. The
  audit picks, per op, between: (a) drop the flag entirely (flag
  was informational, no consumer depended on strictness); (b)
  promote to unconditional `True` because the op's construction
  actually does guarantee it; or (c) keep the claim but wrap the
  output in an explicit `assert_in_range` that carries the
  tolerance the old `APPROXIMATE` level implied.

### How `assert_in_range` interacts with bounds

Assertions enter the system at two different places depending on
what they wrap:

- **Asserts on an `InputNode`** tighten the basis itself. The
  asserted `[lo, hi]` replaces the corresponding `x_lo[j]` /
  `x_hi[j]` entries for each of that `InputNode`'s components. Every
  downstream `AffineBound.to_interval()` that consults the basis
  box automatically sees the tighter bounds. No `AffineBound`
  coefficients change.

- **Asserts on a non-`InputNode`** clamp the wrapped node's derived
  interval. The `Assert` node's `AffineBound` *copies the wrapped
  node's `A_lo` / `A_hi` / `b_lo` / `b_hi` coefficients
  unchanged*, so downstream ops that read affine coefficients
  (`Linear`, `Add`, `Concatenate`) retain every cancellation
  opportunity from the wrapped node. The assertion is applied
  only inside `to_interval()` — which intersects the derived
  per-component interval with the asserted range.

  Downstream ops that read an interval (`ReLU` for its
  straddling-case analysis, `Attn` for V's interval) therefore see
  the tightened interval; downstream ops that read coefficients
  see the unmodified ones. If the inferred interval was already
  tighter than what the user asserted, the assert is a no-op
  statically (the runtime predicate still runs).

---

## Graph lifecycle

The previous `NodeValueType` model was eager: `Node.__init__`
computed the node's value type on the spot, from already-constructed
parents. That no longer works. A non-`InputNode`'s `AffineBound` is
expressed over a basis whose variables are *every* `InputNode` in
the graph — a fact only known once graph construction is complete.

Under the new model, graph construction and bound computation are
separate phases. Users writing the happy path (build a graph, run
or compile it) do not need to think about the phase boundary —
compile and compute trigger finalization automatically. This section
describes the lifecycle for completeness; users writing consumer
ops, diagnostic tools, or tests that want to inspect bounds
directly need to know it.

### Construction phase

- `InputNode(d_output, ...)` registers a new basis contribution.
  `compute_value_type` does not run.
- `Linear`, `Add`, `ReLU`, `Concatenate`, `Attn`, `Assert`,
  `Embedding`, `PosEncoding`, `LiteralValue` record their inputs
  and structural parameters. `compute_value_type` does not run.
- Bound-consumer ops (`cond_gate`, `floor_int`, `assert_in_range`,
  `attend_mean`) return a **placeholder node** with a known
  `d_output` and a recorded reference to the wrapped input and
  op-specific configuration. The subgraph that materializes the
  consumer's internal constants is not yet built.

During construction, access to `node.value_type` or
`node.value_type.bounds` raises `ValueTypeNotFinalized` with a
message naming `finalize(root)`, `compile_graph(root)`, and
`node.compute(...)` as ways to trigger finalization.

### `finalize(root: Node)`

A single public entry point that runs, in order:

1. Walk the ancestors of `root` topologically. Collect every
   `InputNode` reachable. Construct the `Basis`: `n = sum of
   d_output` over `InputNode`s; `x_lo`, `x_hi` populated from each
   `InputNode`'s declared range (or ±∞ if none).
2. For each non-placeholder node in topo order, run
   `compute_value_type()` and cache the resulting `NodeValueType`
   on the node.
3. For each consumer-op placeholder in topo order, call its
   **materialize hook**. The hook sees the wrapped input's now-known
   `AffineBound` (or its `to_interval()`), derives the scalar
   constants it needs (`M` for `cond_gate`; breakpoint grid for
   `floor_int`; etc.), and constructs the actual subgraph in place
   of the placeholder. Downstream references to the placeholder are
   rebound to the subgraph's output node.
4. Run `compute_value_type()` on the primitives created during
   step 3.
5. Mark the graph frozen.

`finalize` is **idempotent** — calling it on an already-frozen
graph is a no-op.

### Triggering

- `compile_graph(root)` calls `finalize(root)` as its first step if
  the graph is not already frozen.
- `Node.compute(n_pos, input_values)` does the same, using the node
  it's called on as the root.
- Users who only build and compile never need to call `finalize`.
- Users who want to inspect bounds before compile (tests, debugging)
  or who author consumer ops or diagnostic tooling call
  `finalize(root)` explicitly.

### Frozen graph invariants

After finalization:

- Every reachable node has a valid `node.value_type`.
- No consumer placeholders remain anywhere under `root`.
- Adding a new `InputNode` to the same session raises.
- Structural modification (re-parenting, rewriting) to frozen
  nodes is an error.

For tests that build multiple graphs in one process, a scoped
helper (`with fresh_graph_session():` or equivalent) resets the
per-session state.

---

## Per-op propagation rules

Each `compute_value_type()` below consumes its inputs' `AffineBound`
and produces this node's `AffineBound`. All inputs and outputs
share the graph's single basis.

### `Linear` — `y = W · v + c`

Exact. Sign-split `W`:

```
W_plus  = clamp(W, min=0)
W_minus = clamp(W, max=0)

A_lo(y) = W_plus · A_lo(v) + W_minus · A_hi(v)
b_lo(y) = W_plus · b_lo(v) + W_minus · b_hi(v) + c

A_hi(y) = W_plus · A_hi(v) + W_minus · A_lo(v)
b_hi(y) = W_plus · b_hi(v) + W_minus · b_lo(v) + c
```

Four GEMMs (two per bound), `[d_out × d_in] · [d_in × n]`. This is
the dominant compile-time cost of the bound-propagation pass
itself; it is separate from the compiled transformer's runtime
forward-pass compute.

The per-component min/max aggregation inside
`linear_output_range` is no longer needed — per-component
tightness is already present in the input's `A`.

### `Add` — `z = u + v`

Preserves affine structure exactly. Addition of affine expressions:

```
A_lo(z) = A_lo(u) + A_lo(v)
b_lo(z) = b_lo(u) + b_lo(v)

A_hi(z) = A_hi(u) + A_hi(v)
b_hi(z) = b_hi(u) + b_hi(v)
```

No sign split — addition preserves monotonicity in both
directions. This is the rule that makes correlation cancellation
automatic. `cond_gate`'s `add M → ... → sub M` pattern cancels
exactly: the `M` constants have zero `A` coefficients and their
`b` offsets are equal-and-opposite, so the sum's `b_lo` / `b_hi`
contributions from the two `M`s cancel cleanly.

Two important caveats on "exactly":

1. *Degenerate operands do not cancel against each other.* If
   either operand has a degenerate `AffineBound` (zero `A`
   matrices — e.g. the output of `Attn`), `Add` correctly sums
   their intervals but no coefficient cancellation is possible
   because there are no coefficients. A cancellation intended
   between two nodes that both passed through `Attn` will not be
   recovered.

2. *Straddling `ReLU` breaks exact cancellation through the ReLU.*
   When a node passes through `ReLU` in the straddling case, its
   output carries the linear-envelope coefficients (chord on the
   upper side, `α · v` on the lower), not the original function's
   coefficients. Two operands whose *true* values cancel exactly
   can re-emerge from `ReLU` with envelopes that differ and no
   longer cancel. Cancellation via `Add` is exact only when both
   operands reach `Add` via paths through exact rules (`Linear`,
   `Add`, `Concatenate`, non-straddling `ReLU`); a straddling
   `ReLU` on the path inserts envelope gaps that survive the sum.
   In practice the scheduler does not construct patterns that
   depend on cancellation through a straddling `ReLU`.

### `ReLU` — element-wise

Element-wise case analysis on each component's derived interval
`[l[i], h[i]]`. Three cases, using non-strict inequalities on the
boundary so `l[i] == 0` or `h[i] == 0` exactly are deterministic:

- `l[i] >= 0` (component always non-negative, including `l[i] == 0`):
  identity. Copy the input's row of `A_lo`, `A_hi`, `b_lo`, `b_hi`
  unchanged.
- `h[i] <= 0` (component always non-positive, including
  `h[i] == 0`): zero the row — `A_lo[i, :] = A_hi[i, :] = 0`,
  `b_lo[i] = b_hi[i] = 0`.
- `l[i] < 0 < h[i]` (strict straddling): apply linear envelope.

The boundary cases (`l[i] == 0` alone, `h[i] == 0` alone) are
handled by the first two branches above; the all-zero interval
`l[i] == h[i] == 0` is handled by the second branch (it matches
`h[i] <= 0`) and produces the zero row — consistent with `ReLU(0)
= 0`. This matters because `slope = h / (h - l)` in the straddling
branch is only defined when `h - l > 0`, which strict straddling
guarantees.

Straddling components:

```
# Upper bound: chord from (l, 0) to (h, h).
slope = h[i] / (h[i] - l[i])
A_hi(z)[i, :] = slope * A_hi(v)[i, :]
b_hi(z)[i]    = slope * (b_hi(v)[i] - l[i])

# Lower bound: α · v, α ∈ [0, 1].  Heuristic: α = 1 if h >= -l else 0.
alpha = 1.0 if h[i] >= -l[i] else 0.0
A_lo(z)[i, :] = alpha * A_lo(v)[i, :]
b_lo(z)[i]    = alpha * b_lo(v)[i]
```

Soundness sketch for the lower bound. Pointwise, `ReLU(v) ≥ α · v`
holds for every `v` and every `α ∈ [0, 1]` (at `v ≥ 0`,
`ReLU(v) = v ≥ α · v`; at `v < 0`, `ReLU(v) = 0 ≥ α · v`).
Substituting `v`'s lower affine envelope `v_lower(x)` and using
`α ≥ 0`: `ReLU(v(x)) ≥ α · v(x) ≥ α · v_lower(x)`. When
`α · v_lower(x)` happens to be negative on part of the box, this
is trivially true (`ReLU` outputs are non-negative); the bound is
loose there but still sound. `α = 0` reduces to the trivial
`ReLU(v) ≥ 0` bound; `α = 1` is tight at `v = h` and loose at
`v = l`. The heuristic picks whichever gives the smaller expected
gap.

The `α ∈ {0, 1}` choice flips at the boundary `h == -l` (the
symmetric-straddling case). The chosen `α` value is therefore
discontinuous in the input range: a small perturbation of the
input's bounds across that boundary changes the component's
lower-bound coefficients from `0` to the full `A_lo(v)` row or
vice versa. Downstream bounds are sound in both regimes, but the
specific lower bound values can jump. Tests that compare bounds
across small input-range perturbations should not assume
continuity.

`ReLU` is the **only** primitive rule that introduces looseness.
Every other primitive rule is exact; any slack in a downstream
interval traces back to a straddling-`ReLU` envelope or to an
`Attn` (see below).

The α heuristic above is sufficient for the first-cut system.
A tighter choice — learning α per component via gradient descent
against a specific output bound — is known in the bound-propagation
literature as α-CROWN. It is *not* a drop-in swap of this rule:
α-CROWN is formulated around a specific query node (the thing
whose bound you're trying to tighten) and requires
backward-propagation of gradients from that query through every
intermediate `AffineBound`. Adopting it would mean adding autograd
through the bound computation and a per-query optimization pass —
a meaningful architectural addition, not a local rule upgrade. If
the heuristic bounds prove too loose in practice, that work is
well-understood in the literature; it just isn't small.

### `Concatenate` — row-stacking

Exact. Stack the inputs' `A` matrices row-wise and their offset
vectors element-wise:

```
A_lo(z) = vstack([A_lo(inp) for inp in inputs])
b_lo(z) = concat([b_lo(inp) for inp in inputs])
# same for A_hi, b_hi
```

The existing `intersect_element_props`-based implementation and its
"union of child ranges" aggregate are replaced by this exact
concatenation. Downstream `Linear`s read the stacked `A` directly,
which is where per-component tightness now comes from — no
`linear_output_range`-style aggregation needed.

### `LiteralValue`

```
A_lo(z) = A_hi(z) = zeros([d_output, n])
b_lo(z) = b_hi(z) = value
```

### `InputNode`

Row `i` is the one-hot vector picking out the basis variable that
*is* component `i` of this `InputNode`:

```
A_lo(z)[i, j] = A_hi(z)[i, j] = 1 if j == basis_index_of(input_node, i)
                                else 0
b_lo(z) = b_hi(z) = 0
```

### `Assert`

The `Assert` node wraps another node and attaches an asserted range
(and an optional structural claim). Its `AffineBound` copies the
wrapped node's coefficients unchanged; its `to_interval()`
intersects with the asserted range. See
*How `assert_in_range` interacts with bounds* above for the full
story.

Structural flags (`is_integer`, etc.) on the `Assert`'s
`NodeValueType` come from the user's declaration, OR-ed with
whatever the wrapped node already claimed.

### `Embedding`

An `Embedding` node is an integer-indexed lookup into a constant
table whose values are known at construction time. Its
`AffineBound` is a degenerate (zero-coefficient) bound whose
offsets are the per-component min and max over the full embedding
table:

```
A_lo = A_hi = zeros([d_output, n])
b_lo[k] = min over rows r of table[r, k]
b_hi[k] = max over rows r of table[r, k]
```

Tightening the min/max by restricting to rows the index input can
actually reach (via its own bound) is a future optimization; the
full-table rule is always sound.

### `PosEncoding`

Position encodings are position-dependent constants whose component
range is a property of the encoding function itself, not of the
graph's sequence length. For the standard sin/cos encoding, every
component lies in `[-1, 1]` regardless of position; the
`AffineBound` is degenerate with those offsets:

```
A_lo = A_hi = zeros([d_output, n])
b_lo[k] = -1.0    # for sin/cos; different encodings give different ranges
b_hi[k] = +1.0
```

Other encoding functions substitute their own per-component range.

### `Placeholder`

`Placeholder` is a zero-width sentinel used where a real node is
not yet available. Its `AffineBound` has shape `[0, n]` for `A`
and `[0]` for `b` — technically valid, contributes nothing to
downstream consumers. A `Placeholder` should never end up
reachable from a finalized graph's root; if one does, finalization
raises.

### `ValueLogger`

`ValueLogger` is a pass-through wrapper used for debugging. Its
`AffineBound` is a direct copy of the wrapped input's — same
coefficients, same offsets. Structural flags copy through
unchanged. `ValueLogger` must not appear in any graph that is
compiled; `compile_graph` rejects it.

### Structural flag composition

The structural flags (`is_integer`, `is_binary`, `is_sign`,
`is_one_hot`) carry information orthogonal to the `AffineBound`.
Each primitive op composes them alongside the affine propagation,
per the rules below. Unlisted flag/op combinations default to
`False` (conservative).

**`Linear` — `y = W · v + c`:**
- `is_integer(y)` = `v.is_integer AND W` is an integer tensor
  AND `c` is an integer tensor
- `is_binary`, `is_sign`, `is_one_hot`: `False` (Linear does not
  preserve these in general; a permutation matrix is a special
  case not worth detecting)

**`Add` — `z = u + v`:**
- `is_integer(z)` = `u.is_integer AND v.is_integer`
- `is_binary`, `is_sign`, `is_one_hot`: `False`. Sums of binary
  values land in `{0, 1, 2}`; sums of sign values in
  `{-2, 0, 2}`; sums of two one-hots carry two 1s.

**`ReLU` — `z = relu(v)`:**
- `is_integer(z)` = `v.is_integer`
- `is_binary(z)` = `v.is_binary OR v.is_sign`. Binary inputs pass
  through (`ReLU` is identity for non-negative values); sign
  inputs (`{-1, +1}`) become binary (`{0, 1}`) under `ReLU`.
- `is_sign(z)`: `False`. Sign becomes binary under `ReLU`, never
  stays sign.
- `is_one_hot(z)` = `v.is_one_hot`. `is_one_hot` already implies
  `is_binary`, which implies non-negative inputs, so `ReLU` is the
  identity and preserves one-hotness.

**`Concatenate` — `z = [u_1, u_2, ...]`:**
- `is_integer(z)` = AND over all inputs
- `is_binary(z)` = AND over all inputs
- `is_sign(z)` = AND over all inputs
- `is_one_hot(z)`: `False` when there are two or more inputs (the
  stacked vector carries one `1` per input, so sum > 1).
  Pass-through for a single-input `Concatenate`.

**`LiteralValue`:**
Flags are inferred from the literal tensor: `is_integer` from its
dtype and values; `is_binary`, `is_sign`, `is_one_hot` from
element-range and sum checks. Same as today.

**`InputNode`:**
Flags come from the user's declaration at construction, or from
`assert_integer` / `assert_onehot` / similar applied to the input
before the basis is frozen.

**`Assert`:**
Flags compose by OR: either the wrapped node's inferred flag holds
*or* the user asserted it explicitly. The runtime predicate
enforces the claim either way.

**`Attn`, `Embedding`, `PosEncoding`:**
No structural flags claimed by default. These outputs pass through
softmax, a lookup table, or sin/cos respectively — none of which
preserve binary / sign / one-hot / integer in general. Call sites
that need a flag on one of these outputs should wrap in
`assert_integer` / `assert_onehot` / etc.

---

### Composite ops

Composite ops are Python functions, not `Node` subclasses. They
split into two groups depending on whether they read their inputs'
bounds.

**Pure subgraph constructors** — `sum_nodes`, `thermometer_*`, and
similar ops that never read `.value_type` — continue to build their
subgraphs immediately from primitive nodes and return one of those
primitives. They need no affine rule; their return node is a
`Linear` / `Add` / `ReLU` / `Concatenate`, each of which has a rule.

**Bound-consumer ops** — `cond_gate`, `floor_int`,
`assert_in_range`, `attend_mean`, and any other op that reads its
input's `value_type.value_range` or
`value_type.bounds.to_interval()` to pick a constant — return a
**placeholder node** at construction time. See the
*Graph lifecycle* section above: the placeholder carries enough
information (wrapped input, op configuration) for `finalize(root)`
to materialize the subgraph later, once bounds are known.

From the caller's perspective this is invisible: `cond_gate(x,
cond, val)` still returns "a node representing the cond-gate's
output" with the right `d_output` — just backed by a placeholder
instead of a fully-built subgraph until finalization. Downstream
code that treats it as a `Node` input to further ops works
unchanged.

### Ops with custom `compute_value_type` overrides

Any `Node` subclass that currently defines its own
`compute_value_type` (other than the primitives listed above) needs
to be ported: either (a) write an affine rule specific to the op,
or (b) fall back to a degenerate `AffineBound` (zero `A`; `b_lo`,
`b_hi` set to whatever scalar interval the current rule produces).
Option (b) is sound and matches today's tightness; option (a) is
the upgrade path that unlocks the affine benefits for that op.

---

## `Attn`

Torchwright's `Attn` computes
`softmax(Q · K^T, causal-masked) · V · output_matrix`, where
`Q = query_in · query_matrix`, `K = key_in · key_matrix`, and
`V = value_in · value_matrix` — each projection is a `Linear`-shape
transform on one of the three input nodes (see
`torchwright/graph/attn.py:117-150`).

Sound linear relaxations for softmax and for the bilinear `Q·K^T`
do exist in the bound-propagation literature (Shi et al., ICLR
2020, "Robustness Verification for Transformers," and subsequent
work). We do not adopt them here. The reasons are design-local: the
published relaxations introduce coupling between token positions
that a single-position basis cannot represent, and the compile-time
benefit at torchwright's specific bound-consumer sites is
unmeasured. Treating `Attn` as an affine-terminating op is a
deliberate scope choice, not a mathematical impossibility.

`Attn.compute_value_type()` therefore produces a degenerate
`AffineBound` for the whole op, via three steps that mirror the
actual forward pass:

1. **Propagate `value_in`'s `AffineBound` through `value_matrix`**
   using the same sign-split rule as `Linear` (`value_matrix` plays
   the role of `W`; bias is zero). This gives an `AffineBound` for
   the projected V in the graph's basis. Call `to_interval()` on
   it to get per-component intervals `[vproj_lo[k], vproj_hi[k]]`
   for `k = 0 .. d_v - 1`. Because `NodeValueType` is
   position-agnostic, these bounds hold at every token position by
   construction.
2. **Apply the "any probability vector" relaxation.** The attention
   output at component `k` (before the output projection) is a
   convex combination over token positions of the projected V at
   that position, each of which lies in
   `[vproj_lo[k], vproj_hi[k]]`. A convex combination of values all
   lying in one interval lies in the same interval. (Q and K are
   not consulted; the probability-vector assumption is a strict
   over-approximation of what softmax actually produces.)
3. **Propagate through `output_matrix`** using a sign-split on
   scalar intervals:

   ```
   out_matrix_plus  = clamp(output_matrix, min=0)   # shape [d_v, d_output]
   out_matrix_minus = clamp(output_matrix, max=0)

   b_lo = vproj_lo @ out_matrix_plus + vproj_hi @ out_matrix_minus
   b_hi = vproj_hi @ out_matrix_plus + vproj_lo @ out_matrix_minus
   ```

   Emit the final `AffineBound`:

   ```
   A_lo = A_hi = zeros([d_output, n])
   b_lo, b_hi = the interval computed above
   ```

   The attention output has no dependence on the basis. Its
   interval is carried in the offset vectors. Note that query and
   key matrices never appear in the bound — the probability-vector
   relaxation makes them irrelevant to output range.

Downstream ops consume this `AffineBound` normally. Their own `A`
matrices keep non-zero coefficients for *their other inputs*, so
any cancellation that flows through a skip connection around the
attention (e.g. an `InputNode` component that passes the attention
sublayer unchanged and is later subtracted) is preserved exactly.

### What is lost

Any cancellation whose two sides both pass *through* an `Attn`
cannot be recovered, because both sides have zero coefficients in
the basis after attention. In practice the scheduler does not
build graphs that rely on such patterns; when they occur, an
explicit `assert_in_range` on a post-attention quantity gives the
following `Add` something tight to cancel against.

### Related prior work

Forward-mode linear-relaxation bounds for transformer verification
are developed in Shi, Zhang, Chang, Huang, and Hsieh, "Robustness
Verification for Transformers" (ICLR 2020), which includes
relaxations for softmax and the bilinear `Q·K^T`. Bonaert et al.,
"Fast and Precise Certification of Transformers" (PLDI 2021), and
subsequent work refine the softmax rules. The auto_LiRPA library
(Xu et al., NeurIPS 2020) provides both forward- and backward-mode
LiRPA/CROWN implementations with these relaxations. This design
deliberately stops short of adopting those rules — see the
rationale at the top of the `Attn` section — but the literature is
the natural reference if we decide to push further on attention
tightness later.

---

## Effect on existing bound consumers

Every op that today reads `node.value_type.value_range` keeps
reading it. The aggregate returned by the derived property is
strictly inside (or equal to) the old one, so the only observable
change at these call sites is tighter constants:

| Consumer | Where it reads bounds | Effect of tighter bounds |
|---|---|---|
| `cond_gate` | `_max_abs_or_raise(input)` to pick `M` | Smaller `M` offset; less precision loss in the gate's MLP |
| `floor_int` | span of input range to pick breakpoints | Fewer breakpoints; smaller MLP width |
| `assert_in_range` | declared `[lo, hi]` vs inferred | Runtime predicate unchanged; inferred static claim often meets or exceeds what the user asserted |
| `attend_mean` | V input's range for `atol` | Smaller tolerance on the downstream assert |

Consumers that want per-component precision (instead of the
node-wide aggregate) call `node.value_type.bounds.to_interval()`
and index into `lo[i]` / `hi[i]`. In most cases they don't need
that granularity anymore, because downstream `Linear`s already see
per-component tightness through the input's `A`.

---

## Runtime verification

`TW_VERIFY_VALUE_TYPES` continues to wrap each `Node.compute` and
check the observed tensor against the declared `value_type`. Two
changes:

- The range check is per-component: each `observed[i]` is checked
  against `bounds.to_interval().lo[i]` / `.hi[i]`, not the aggregate.
  The old aggregate check was strictly coarser; violations that the
  old check missed now surface.
- Structural-flag checks (`is_integer`, `is_binary`, `is_sign`,
  `is_one_hot`) still run, identically to today, but without the
  `APPROXIMATE` tolerance tier — every violation is a hard
  assertion failure.

An optional debug check (`TW_VERIFY_AFFINE=1`, off by default)
samples a small number of points from the basis box, evaluates the
affine expressions at those points, and asserts
`A_lo · x + b_lo ≤ observed(x) ≤ A_hi · x + b_hi`. Used while
developing new rules; never on in production runs.

---

## Deliberately out of scope

- **Affine-preserving attention rules.** Published linear
  relaxations for softmax and bilinear `Q·K^T` exist but are not
  adopted here (see the `Attn` section for the rationale). `Attn`
  emits a degenerate `AffineBound` whose `A` matrices are zero.
- **Cancellation through attention.** Not recovered; see the
  `Attn` section above.
- **Learnable α on `ReLU`.** The `α ∈ {0, 1}` heuristic is used.
  Per-component learnable α (α-CROWN) is a real tightness upgrade,
  but adopting it requires backward-mode infrastructure (autograd
  through bounds, per-query optimization) — not a local change to
  the `ReLU` rule alone. See the `ReLU` section for details.
- **Piecewise-affine joint reasoning.** Each component's bound is
  a single pair of affine expressions, not a union over cases.
- **`Guarantee.APPROXIMATE`.** Removed. Structural flags are
  treated as unconditional claims; ops that previously declared
  `APPROXIMATE` either drop the claim or pair it with an explicit
  `assert_in_range`.

---

## Data flow summary

```
InputNodes ──► Linear ─► Add ─► ReLU ─► Linear ─► Add ─► ... ─► Attn ─► Linear ─► ...
           │                                                         │
           │      all AffineBounds over the graph's single basis     │
           │           (InputNode components, declared ranges)       │
           │                                                         │
           │ exact propagation for Linear/Add/Concatenate            │
           │ linear-envelope relaxation for straddling ReLU          │
           │ zero-coefficient AffineBound emitted by Attn            │
```

Every bit of bound tightness comes from exact affine algebra on
`Linear` / `Add` / `Concatenate`. Every bit of looseness comes
from one of two places: straddling-`ReLU` envelopes, or `Attn`
emitting a zero-coefficient `AffineBound`.

---

## Appendix: context for plan-mode and future maintainers

This appendix holds context that shaped the design but does not
belong in the design proper — rejected alternatives, validation
steps that were discussed but deferred, claims that should be
verified during implementation, and open questions that plan mode
needs to resolve. None of it changes the design; all of it is
useful when deciding *how* to implement or *whether* to revisit a
choice.

### A. Design alternatives considered and rejected

- **Forward-mode vs backward-mode LiRPA / CROWN.** Backward-mode is
  tighter per query (relaxations picked per output), but requires
  re-walking ancestors per bound-consumer site; this design has
  many such sites across the graph. Forward-mode was chosen for fit
  with the per-node `compute_value_type()` contract. If forward
  bounds prove too loose in practice, backward-mode is the
  documented upgrade path.
- **Basis construction timing.** Earlier options: (a) convention —
  require all `InputNode`s to be declared first and auto-freeze the
  basis on the first non-`InputNode` construction; (c) dynamic
  basis that widens as new `InputNode`s arrive. Both were rejected
  in favor of (b) — explicit `finalize(root)` with compile / compute
  auto-trigger — because it cleanly separates graph construction
  from bound computation without imposing construction order and
  without the complexity of growing-basis bookkeeping.
- **`Guarantee` removal as a gradual migration.** A shim where
  `Guarantee.ALWAYS` aliases `True` and `APPROXIMATE` silently
  promotes (or warns) was rejected in favor of all-at-once
  deletion. The scope is small and the shim would leave two
  spellings of the same concept alive in the code indefinitely.
- **auto_LiRPA as a runtime / compile-time dependency.** Rejected.
  auto_LiRPA wraps `nn.Module`s, expects a query-driven workflow,
  and carries its own softmax / bilinear relaxations that this
  design does not use. Adopting it would mean writing an IR-to-
  Module adapter per primitive and running the library in a mode
  it is not designed for. A from-scratch implementation of
  forward-mode LiRPA inside `compute_value_type()` is both smaller
  and a better fit.
- **Affine-preserving attention.** Published softmax and bilinear
  `Q·K^T` relaxations (see *Related prior work* in the `Attn`
  section) could tighten attention bounds but introduce
  cross-position coupling that the single-position basis cannot
  represent. Rejected for v1; not for reasons of mathematical
  impossibility.
- **α-CROWN for the `ReLU` lower bound.** Rejected for v1 because
  it is not a local rule change — it requires autograd through
  bounds and a per-query optimization pass. Treated in the doc as
  a future upgrade with non-trivial architectural cost.

### B. Validation step discussed but not performed

Before committing to implementation, a diagnostic measurement was
proposed: wrap the compiled `HeadlessTransformer` (walkthrough or
similar real graph) in `auto_LiRPA`'s `BoundedModule`, run CROWN,
and compare the resulting interval widths at the four bound-
consumer sites (`cond_gate._max_abs_or_raise`,
`floor_int`'s breakpoint span, `assert_in_range`'s inferred claim,
`attend_mean`'s `atol`) against the current scalar-IBP widths.
The expected ratios are what justify the implementation cost.

This step was not performed. The decision to proceed rested on
qualitative reasoning: the codebase already has hand-rolled
workarounds for correlation-blindness (`linear_output_range`'s
per-component arithmetic, the cond_gate offset pattern), which
suggests the gap is meaningful. Plan mode should decide whether to
run the measurement as a first milestone to set an empirical
expectation, or proceed directly.

### C. Claims to verify during implementation

The design makes several empirical claims that are plausible but
uncorroborated. Each should be checked when the relevant code is
touched:

- **"The scheduler does not build graphs that rely on
  cancellation through a straddling `ReLU`."** Stated in the `Add`
  caveat. Before trusting it, grep for patterns like
  `ReLU(x) - ReLU(x)` or structural equivalents and confirm no
  compiled op depends on exact cancellation downstream of a
  straddling ReLU.
- **"Typical `n` is on the order of tens to low hundreds."**
  Stated in the sizing subsection. Verify by computing
  `sum_of_input_widths` on the actual compiled DOOM graph and any
  other non-trivial example.
- **"`float64` is sufficient under bounded weight norms."**
  Stated in the sizing subsection. Verify by running a test
  pipeline that compares `to_interval()` output between
  `float32` and `float64` coefficient storage on a 50-70 layer
  graph. If divergence is larger than bound-consumer tolerance,
  the claim fails.
- **Structural flag composition rules match existing compiled
  behavior.** The tables in *Structural flag composition*
  were derived from first principles. Compare against the
  current `_min_guarantee`-based composition on a representative
  set of real nodes to confirm no semantic drift.

### D. Open implementation questions

The design says *what* happens but not always *how*. Plan mode
needs to pin these down:

- **Placeholder Python class.** Is there a single
  `ConsumerPlaceholder(Node)` class parameterized by op type and
  config, or one subclass per consumer op? What attributes does
  it carry for materialize? How are downstream references to it
  rebound when materialization replaces it with a subgraph —
  mutation of `inputs` lists on descendants, or a separate indirection?
- **`fresh_graph_session` scope and semantics.** Context manager
  vs explicit reset? What state is per-session vs global? What
  happens if nested?
- **`AffineBound` factory signatures.** The appendix in the doc
  sketches `identity`, `constant`, `degenerate`. Exact signatures
  (keyword args, dtype handling, what accepts `None`) are
  undecided.
- **Migration ordering.** Which primitive ports first? Option (b)
  in *Ops with custom `compute_value_type` overrides* allows a
  partial migration (degenerate bounds for un-ported ops). What is
  the minimal set of ports needed to unlock measurable tightening
  at the four bound-consumer sites?
- **Test strategy.** Unit tests per rule (hand-computed bounds
  for small graphs)? Property-based comparison against `auto_LiRPA`
  on simple nets? End-to-end regression on compiled DOOM
  constants?
- **`compile_graph` integration.** Where exactly does the auto-
  `finalize` call land — before scheduling, during, or after? How
  is the frozen-graph invariant enforced against scheduler mutations?

### E. Adjacent concerns the plan should account for

- **`make measure-noise` re-run per ported op.** Tighter bound-
  driven constants change compiled op behavior. The numerical-
  noise data in `docs/op_noise_data.json` and the commentary in
  `docs/numerical_noise_findings.md` need to be regenerated /
  updated as ops migrate. See the *Numerical noise* section of
  `CLAUDE.md` for the exact workflow.
- **ONNX export.** `AffineBound` is compile-time metadata and is
  explicitly excluded from export. Verify the existing ONNX
  export tests still pass after migration — no bound-related
  state should leak into exported weights.
- **Compiled-artifact parity.** Migration should be neutral or
  tightening for compiled artifact behavior. A regression gate
  that compares compiled behavior on a reference example
  (walkthrough, adder, calculator) before and after migration is
  a good guardrail.
