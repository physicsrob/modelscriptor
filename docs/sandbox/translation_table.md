# Sandbox → torchwright/doom/ Translation Table

This is the porting bible. When porting a sandbox phase to the real
graph, every sandbox primitive has a corresponding construct in
`torchwright/`. This table records the mapping.

**Discipline rule.** Adding or changing a sandbox primitive in
`doom_sandbox/api/` updates the corresponding entry here in the same
commit. Out-of-date entries here are how port-time discovers
"primitive X doesn't actually exist on the real side" the hard way.

## Vec / data flow

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `Vec` | A node in `torchwright/graph/` plus its allocated residual columns | `.shape` is the node's `d_output`; `.depth` is the longest op chain producing it |
| `Vec.depth` propagation | The compiler's CP-SAT scheduler computes layer assignments from read-after-write constraints | Sandbox depth is a *floor* on compiled layer count |
| `concat(*vecs)` | `Concatenate(*nodes)` in the graph DSL | Free in both worlds (just renames residual column ranges) |
| `split(vec, sizes)` | Manual indexing into `Concatenate` outputs, or pass-through with sliced consumers | Free |

## PWL primitive

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `pwl_def(fn, breakpoints, input_range)` at module level | A `PiecewiseLinear` op in `torchwright/ops/` constructed from the same `fn` and breakpoint configuration | Sandbox stores the function; real-graph builds the breakpoint table at compile time |
| `PWLDef.__call__(vec)` | `piecewise_linear(node, op)` in graph construction | Adds 1 MLP sublayer in the compiled transformer |
| Affine PWLs (interp-residual = 0) | `Linear` node in the graph | The compiler can recognize affine PWLs and lower them to `Linear` if useful. The sandbox's per-call gaussian (`σ = 1e-6 * |value|`) still applies — it represents FP32 / matmul noise, not the PWL approximation residual |

## Token system

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `TokenType` declaration | An entry in `torchwright/doom/embedding._CATEGORY_INDEX` plus rows in `W_EMBED` | Type identity → E8 category code in the real embedding |
| `IntSlot(lo, hi)` | A residual column carrying the integer literally, K-column-style, for small ranges; or a one-hot for large | Real graph has the K column (`cols[25:26]`) for `k ≤ 255` |
| `FloatSlot(lo, hi, levels=65536)` | The Gray-code + raw-slot mechanism in `W_EMBED[8:25]` | Same precision contract — quantize via `quantize_to_range`, recover via argmax against `W_EMBED.T` |
| `make_token(type, **slots)` | A `compose_switch` over the `next_token_embedding` build, plus the per-slot encoding ops | The "build a structured embedding" path |
| `extract_int_slot(input_vec, name)` | A `Linear` node reading the K-column or relevant residual range | One MLP-equivalent sublayer |
| `extract_float_slot(input_vec, name)` | `Linear` reading the raw slot, followed by `dequantize_from_range` | The decoded float, not the integer |
| `is_type(input_vec, T)` | An `equals_vector` (E8 dot-product) against the type's category code | Sharp because E8 codes have wide pairwise margin (1600 self-dot, ≤800 cross-dot) |

## past.* — cross-position primitives

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `past.pick_argmax(query, key, value)` | `attend_argmax_dot(query=Q, key=K, value=V)` in `torchwright/graph/attention/` | Direct match |
| `past.pick_argmin(query, key, value)` | `attend_argmax_dot` with negated key (or negated query) | The graph doesn't have `argmin` directly; you negate |
| `past.pick_above_argmin(query, key, value, threshold)` | A quadratic-equality `attend_argmax_dot` parameterized by the threshold; or a chain that masks-then-argmins | The doom_graph doc notes the old `attend_argmin_above_integer` was replaced by quadratic-equality patterns |
| `past.lookup(query, key, value)` | `attend_argmax_dot` with key-design that gives a clean margin (≥1.0 score gap) | Same as `pick_argmax` mechanically; the "raise on no clear winner" is a sandbox-only check |
| `past.pick_most_recent(query, key, value)` | `attend_most_recent_matching` | Direct match — recency-biased equality lookup |
| `past.mean(value_name)` | `attend_mean_where(value, where)` where `where` is non-zero on contributors | Direct match. Single-contributor case = broadcast; multi-contributor case = mean |
| `past.pick_argmax_by(score, value)` | `attend_argmax_dot` with `query=constant_one` and `key=score` | The score becomes the K |
| `past.pick_argmin_by(score, value)` | `attend_argmax_dot` with `query=constant_one` and `key=-score` | Negate the score |
| `past.pick_above_argmin_by(score, value, threshold)` | Quadratic-equality formulation of "smallest above threshold" | Compose with the threshold |

## Forward output / lifecycle

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `ForwardOutput.next_token` | The 49-wide `next_token_embedding` overflow output | Argmaxed against `W_EMBED.T` by the host |
| `ForwardOutput.pixels` | The `pixels`/`col`/`start`/`length` overflow region | Read by the host, blitted, never re-embedded |
| `past.publish(name, vec)` | Writing a column (or column range) to the residual stream after an MLP/attention layer | Each published name maps to a designated column range. Subsequent attention layers (this position's later layers, or any later position's layers) can read it. Re-publishing under the same name = a residual-column rewrite by a later layer |
| `input.type` / `input.<slot>` auto-publish | The token's E8 / Gray-coded slots already on the residual stream from the embedding layer | Free in the real graph — the embedding deposits the slot info; sandbox materializes it as queryable Vecs at every position |
| `Config.terminal_token_types` | The `done` overflow signal | Host stops when `done > 0` |
| `Config.max_positions` | The host loop's safety cap | No real-graph counterpart needed (host concern) |
| `run(config, prefill, forward)` | The host loop in `torchwright/doom/play.py` / `step_frame` | The autoregressive driver |

## stdlib (api/std.py)

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `type_switch(*branches)` | `compose_switch` in `torchwright/graph/` | Mutually-exclusive branch selection |
| `relu(input_range)` | `relu` op in `torchwright/ops/` | Piecewise-linear with 2 breakpoints, exact for affine inputs |
| `clamp(lo, hi)` | `clamp` / `clip` op in `torchwright/ops/arithmetic_ops.py` | Identity PWL with runtime clamping at the input boundaries |
| `compare_const(c, input_range)` | `compare_const(c)` op in `torchwright/ops/` | Steep ramp around `c` |
| `piecewise_linear(fn, breakpoints, input_range)` | `piecewise_linear` op | Generic 1D PWL, equivalent to `pwl_def` |
| `multiply(a, b)` | `multiply_2d` op in `torchwright/ops/` | Bilinear PWL |
| `piecewise_linear_2d(fn, breakpoints, input_range)` | `piecewise_linear_2d` op | Generic 2D PWL |

## Debug primitives

| Sandbox | Real-graph counterpart | Notes |
|---------|------------------------|-------|
| `print_vec(vec, label)` | `debug_watch(node, lambda v: True, label)` (always-on observation) | The real graph's `DebugWatch` always takes a predicate |
| `debug_watch(vec, predicate, label)` | `debug_watch(node, predicate, label)` | Direct match |
| `assert_in_range(vec, lo, hi)` | `assert_in_range` from `torchwright/graph/asserts.py` | Direct match |
| `assert_close(vec, expected, atol)` | No direct match — usually a chain of `compare` + `assert_bool` | The sandbox version is a single primitive for ergonomics |
| `assert_bool(vec)` | `assert_bool(node)` | Direct match |
| `assert_integer(vec)` | `assert_integer(node)` | Direct match |
| `assert_(vec, predicate, message)` | Custom assertion via `Assert(node, predicate)` node | Generic fallback |

## What does NOT have a clean translation

These exist as conveniences in the sandbox but require composition or
care when porting:

- **`past.lookup`'s "raise on ambiguous match"** — the real graph doesn't
  validate softmax concentration at runtime. The agent must design key
  schemes with sufficient margin (E8 codes, quadratic-equality, etc.);
  the sandbox's loud failure becomes "softmax silently produces a
  blend" in the real graph if the key scheme is bad. Port-time
  discipline: when porting a `past.lookup`, verify the chosen keys
  give a real margin.

- **`past.pick_argmin` and `pick_above_argmin`** — no direct primitives.
  Express via `attend_argmax_dot` with negated keys or
  quadratic-equality formulations. Translation table entries above
  show the recipe.

- **The `extra="ignore"` pydantic config on type schemas** — sandbox-only
  ergonomics for fixture loading. The real-graph side reads
  `MapSubset` directly from its source-of-truth dataclass; no
  serialization layer.
