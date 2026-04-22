# angle=210 investigation: overlay overlap, FP nondeterminism, value-range mismatch

Three independent bugs surfaced while debugging
`test_sort_order_matches_bsp[210]` and the full-suite `[210]` failures.
Each has its own fix; they share only the test that surfaced them.

## 1. Allocator overlap at the delta-transfer layer

**Root cause (one sentence).** Layer 54's `advance_wall` delta head
writes to `residual[160]` while `Linear 2870` (the 60-wide `pixels`
output) still claims col 160 as its slot 25 — two live nodes on the
same residual column.

**Mechanism.**

- `Linear 2870` (pixels output) is allocated `[41, 66, 118, …, 160, …, 225]`;
  col 160 is its slot 25.
- `advance_wall` overlay declares `target_cols = [160]`, `source_cols = [53]`.
- Layer 54 is the appended delta-transfer layer. Head 0 emits the
  combined-form delta for the overlay: slot 0 writes `+residual[53]`,
  slot 1 writes `-residual[160]`, both into O-col 160 — the net is
  `residual[160] := Linear 2900` (the `advance_wall` select output,
  `-1.0` at non-render positions).
- Pre-sublayer reads all land before writes, so each overlay's declared
  target_cols receive their intended value. What's destroyed is
  `Linear 2870`'s in-place slot-25 value at col 160.

**Why the test still failed anyway.** The wrong-output path from this
overwrite alone was not traced — every delta-layer read completes
pre-sublayer, and downstream of layer 54 nothing reads col 160 assuming
it still holds `Linear 2870[25]`. The col-160 destruction is a real
invariant violation but doesn't by itself explain the runtime drift at
step 91/92. See §2 for the actual mechanism behind the `[210]`
regressions.

**Detection.** `module.step(debug=True, debug_atol=1e-7)` fires the
residual-stream self-consistency check:

```
node: output (id=2870, Linear, width=60)
first: layer_39_mlp_out_state      first val: [0.0, …, 0.0]
later: layer_54_mlp_out_state      later val: [-1.0, …, -1.0]
worst col: residual[160] (node index 25)   max_abs_diff: 1.0
```

**Fix (commit `e404eec`).** Reserve overlay target cols `[160..229]`
from the allocator's free pool at compile start. `Linear 2870` lands on
disjoint cols, and the invariant "no live node's residual cols may
overlap an overlay's target_cols" is enforced by the reserve machinery.

**Open.** `_get_effective_consumers` treats a leaf of the terminal
Concatenate as dead once the Concatenate is computed — that's what
lets the overlap form. A cleaner long-term fix would extend "live
until delta" to every output leaf, dropping the reserve band.

## 2. FP nondeterminism at the wall/visibility cond tolerance

**Root cause (one sentence).** `select()` and `cond_gate()` in the
visibility chain ran at `c_tol=0.005` with actual conds landing near
`|cond|=0.995` — a margin smaller than the GPU's own FP nondeterminism,
so the same code on the same inputs intermittently fires borderline
conds.

**Evidence of the nondeterminism.** Two distinct prefill sort outputs
observed on identical code + inputs + GPU:

- `[…53,53,…,81,81,81,81,81]` — step 89 fires with `cond = −0.9949`.
- `[…73,73,…,81,81,81,81,81]` — step 89 passes (`|cond| > 0.995`).

The source is cuBLAS algorithm selection / TF32 / atomics ordering —
below the `c_tol=0.005` absolute budget, but enough to flip a
borderline cond across the boundary.

**Connection to the `[210]` full-suite regressions.** The step 91/92
runtime drift that caused `test_eos_state_matches_reference_oblique[210]`,
`test_sort_order_matches_bsp[210]`, and
`test_frame_matches_reference_oblique[210]` to fail when the full
`tests/doom/` suite ran (but pass under `-k sort_order`) was the same
class of issue: full-suite state (prior tests' cuBLAS warmup biasing
algorithm selection) pushed intermediates onto the noisy side of their
budget. This is a hypothesis the fix validates rather than a traced
causal chain — but widening `_VIS_C_TOL` cleared all three `[210]`
regressions across two consecutive full-suite runs.

**Fix (commit `f569987`).** `_VIS_C_TOL = 0.01` applied to every
`select()` and `cond_gate()` in `_compute_visibility_columns`,
`_plane_clip_contribs`, and `_endpoint_to_column`. Twice the default
budget; downstream consumers absorb the widened cond.

**Open: graph-wide hardening.** Per-op noise measurements in
`docs/op_noise_data.json` don't account for GPU FP nondeterminism when
setting tolerance budgets. Principled fixes: (a) force
`torch.use_deterministic_algorithms` inside `_run_debug_checks`, or
(b) tighten per-op noise measurements until no op lives at its
budget.

## 3. `texture_id_e8` declared value_range vs actual E8 code range

**Root cause (one sentence).** `texture_id_e8` was declared
`value_range=(-1.0, 1.0)` at `torchwright/doom/game_graph.py:361`, but
`torchwright/graph/spherical_codes.py:18` multiplies the raw E8 codebook
by 10×, so actual input values have components in `{-30, -10, 10, 30}`.

**How it reaches an assert.** `cond_gate` computes its M-constant as
`M = 2·max_abs_range`. With the mis-declared range, `M = 20`. At key
positions where `inp[j] = -30`, the on-path `ReLU(M·c + inp[j])`
saturates to `0`, giving output `0 + ReLU(-M) - M = -20`. That violates
the `assert_matches_value_type(Range(-1, 10), atol=0.1)` postcondition
at `render/tex_attention` under `debug=True`.

**Why the test passed without debug.** The saturated `-20` at key slots
doesn't derail `attend_argmax_dot(match_gain=1000.0)`: the `-20` only
appears on slots whose correct value would have been `≤ -1`, and the
large `match_gain` dominates the `≤ 19` dot-product error.

**Fix (commit `d0d981b`).** Widen declared range to `(-30.0, 30.0)` at
`game_graph.py:361`. `M` becomes 60; the cond_gate output type widens
to `Range(-30, 30)`; the assert passes.

**`token_type` had the identical mis-declaration** (line 338,
`value_range=(-1.0, 1.0)` for the same 10×-scaled E8 codes) and was
widened to `(-30.0, 30.0)` preemptively while the context was loaded,
even though it was latent — `token_type` is only consumed by
`equals_vector(...)` (dot-product producing `±1` regardless of input
range) and overlay write-back (which doesn't pass through `cond_gate`).

## Debug infrastructure added while investigating

Now part of `torchwright/` — these are the canonical way to reproduce
this class of bug:

- `CompiledHeadless.__call__` / `.step` accept a `debug_atol`
  parameter for tuning the residual-stream self-consistency
  tolerance. `1e-7` matches legitimate fp-rounding noise; set higher
  to suppress noise-floor diffs when hunting a larger aliasing.
- The self-consistency check collects every violating node
  (sorted by max_abs_diff) instead of aborting on the first.
  First-failure-only mode had previously masked multi-node aliasings.
- `assert_picked_from` and `assert_softmax_hardness` skip query
  positions with no valid key in the causal window and queries whose
  per-row type boolean is 0 (the documented "type isolation by zero
  query/key" pattern). Both patterns leave the softmax uniform over
  the causal window, but the downstream consumer gates the output
  away, so asserting on it fires on a value nothing reads.
