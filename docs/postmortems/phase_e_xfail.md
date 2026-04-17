# Phase E xfail: post-mortem

**Test:** `tests/doom/test_game_graph.py::TestGameGraph::test_renders_off_center_oblique[3.0-2.0-20]`
**Xfail landed:** commit `c2d5a7a` (Phase E)
**Xfail removed:** commit `8367413`
**Root-cause fix:** commit `2e6f5da` (tightened `piecewise_linear_2d` output range)

## Status

**Resolved.** The original xfail hypothesis — residual-column aliasing
in the compiled SORTED attention layer — was wrong. The actual cause
was fp32 precision compounding driven by an overly-loose declared
value range on `piecewise_linear_2d` outputs, which in turn propagated
through the approximate-path `cond_gate`/`select` formula and amplified
small `compare`-op noise by a factor of 1000× or more. Tightening
`piecewise_linear_2d`'s declared output to the function's exact grid
min/max bound collapsed the amplification chain and the test passes
cleanly.

This document is preserved for its history of the diagnostic dead-end
and the lessons it surfaced about noise measurement and value-type
discipline.

## What happened at scene (px=3, py=2, angle=20)

At sort[0], the SORTED stage's `attend_argmin_above_integer` softmax
concentrated on SORTED[0] itself (weight 1.0, logit +800) rather than
any WALL position (logits +555..+637 instead of design +1000). The
raw `sel_bsp_rank` residual carried -1171.875 instead of a clean
integer in `[0, max_walls-1]`. `sort_done` correctly detected the
nonsense and replaced the bogus value with a 99-sentinel, but
downstream THINKING/RENDER produced incorrect pixels from the 99-marked
cached position.

Only this specific scene (and a small neighborhood around it) triggered
the failure; other off-center / oblique scenes worked correctly.

## How the original hypothesis went wrong

The xfail's original reason claimed the bug was "column aliasing in
the compiled SORTED attention layer — specific pair not yet
identified." The reasoning path was:

1. Observation: non-WALL residual columns carry ~100-magnitude values
   where the graph says they should be exactly zero.
2. Magnitude argument: the per-op noise docs show `compare_near_thresh_05`
   with max abs error 0.005 and `piecewise_linear_2d doom_diff_trig`
   with max abs error 7.78. Neither can produce a 100-unit error alone.
3. Inference: the only remaining explanation is residual-column
   aliasing — some other live node's value landed in `sort_value`'s
   bsp_rank slot.
4. Claim: "most naturally explained by residual-column aliasing."

Three flaws in this chain:

- **Step 2 compared the observed error to per-op bounds in isolation.**
  A chain of N piecewise ops compounds fp32 rounding-order drift and
  accumulation error. More importantly, the approximate-path
  `cond_gate`/`select` formula `M·cond + inp - M` has a built-in
  Lipschitz constant `M` w.r.t. its `cond` input. When `M` is in the
  millions (because a downstream `multiply_2d` declared its output
  range to be millions wide), even the tiny `compare`-op noise of
  0.005 becomes 5000-unit error at the gate output.
- **Step 3 named the wrong mechanism.** I1 (allocator
  self-consistency) already forbids two simultaneously-live nodes from
  sharing a residual column; it's asserted on every mutation. "Column
  aliasing" was the one hypothesis that was mechanically ruled out by
  the existing invariants. The author either didn't know I1 covered
  exactly this case, or did and didn't cross-check against it.
- **Step 4 stated confidence without evidence.** No allocator-state
  dump existed; no compiled-vs-oracle per-node walk had been run; the
  investigation tooling available at the time (`scripts/investigate_phase_e.py`)
  only probed residual *values*, not ownership or per-node divergence.

## What actually happened

Evidence from the follow-up investigation (commit `640d523` and the
`scripts/dump_phase_e_allocator.py` runs):

1. **I1 holds.** 212 live columns across 47 nodes at the SORTED
   attention's read layer, all pairwise-disjoint. Allocator bookkeeping
   is sound.
2. **No aliasing at key_in columns.** `pos_encoding`, `score`, and
   `indicators_above`'s leaves own exclusively the columns they're
   supposed to own. The aliasing hypothesis is directly refuted.
3. **Weights are wired correctly.** Moving the compiled module to
   CPU + fp64 collapses the entire compiled-vs-oracle error to ~3e-6.
   The graph's semantic math and the compiler's weight-matrix
   construction agree to double-precision accuracy.
4. **In fp32 on GPU the error is ~16,000 at select_linear2 outputs.**
   First divergence at `c_day_ex_linear2` is a trivial 0.0007 (within
   per-op bounds). Through the chain `piecewise_linear_2d` →
   `cond_gate` → `multiply_const` → `select_linear2`, the error balloons
   to four-digit magnitude.
5. **The amplification lever is the approximate gate's `M` constant.**
   After per-call `M` was introduced (commit `f0e6f86`), `side_P_vec`
   collapsed to clean values (M≈1 for one-hot gating, so M·ε_cond ≈
   0.005). But the `select` chains downstream of `multiply_2d` outputs
   had `M ≈ 8.6 million` — because `piecewise_linear_2d` never declared
   a tight output range and the Linear's default `linear_output_range`
   worst-case bound claimed ranges in the millions for functions whose
   actual output is bounded by ~100.
6. **Tightening `piecewise_linear_2d`'s declared range** (commit
   `2e6f5da`) to the mathematically exact `[min over grid, max over
   grid]` bound collapsed `max(M)` from 8.6M to 16.8K — a 513× drop —
   and brought the worst `select_linear2` fp32 error from 16,223 to 57.
   The Phase E test then passed on its own.

## Doctrine lessons this incident teaches

### Noise docs measure op behavior on clean inputs; the amplification story needs input-Lipschitz constants too.

`cond_gate`'s per-op noise-footer measurement (3e-5 abs, 4e-3 rel)
was taken with `cond ∈ {-1, +1}` to machine precision. The op's
*sensitivity to input noise* is governed by its Lipschitz constant
w.r.t. `cond` (= `M`, which with the old `big_offset = 1000` was
1000). Three orders of magnitude of amplification weren't
documented anywhere because the measurement harness never fed a
noisy `cond`. Any op that routes a signal via a "large offset
cancellation" trick has this property, and its noise envelope
should be stated as a *function of input noise*, not a single
number.

### Declared value ranges are load-bearing, not decorative.

`piecewise_linear_2d` never declared a tight output range because
"nobody reads it downstream" was assumed. That assumption was false
— every downstream `cond_gate`/`select` uses `max|declared_range|` as
the `M` constant in its approximate formula. A 4-order-of-magnitude
over-estimate on one op's declared range became a 4-order-of-magnitude
amplification on every downstream approximate gate. The cheapest
fix was a 5-line addition that declares the range the op's own
docstring already promised.

### Rule out stated invariants before hypothesizing violations of them.

The original xfail hypothesized "residual-column aliasing" — which
is precisely what I1 (allocator self-consistency) forbids, with an
assertion on every allocator mutation. The one-minute check of "does
I1 fire on this test?" would have immediately ruled out the
hypothesis and pointed the investigator elsewhere. The CLAUDE.md
entry for I1 has since been tightened (commit `640d523`) to state
explicitly: "this is the invariant that forbids residual-column
aliasing among simultaneously-live nodes; if I1 doesn't fire, the
bug is not aliasing."

### Investigation tooling should probe structure, not just values.

`scripts/investigate_phase_e.py` was purpose-built for this scene but
only surfaced residual values. Walking the allocator's per-layer
column ownership, or running `probe_compiled` to get per-node
compiled-vs-oracle divergence, would have located the actual failing
node in minutes. `scripts/dump_phase_e_allocator.py` (landed in
commit `640d523`) adds exactly that dimension, organised into six
blocks: I1 sanity, key_in ownership, compiled-vs-oracle values at
the read layer, `probe_compiled`'s first-divergent walk, CPU-fp64
replay, and a per-approximate-gate M-audit with back-traced upstream
sources.

## Tooling introduced during the investigation

- `scripts/dump_phase_e_allocator.py` — six-block diagnostic dump
  (see section above).
- `modal_dump_phase_e.py` — Modal entrypoint for the dump.
- CLAUDE.md §I1 — tightened doc entry naming the aliasing hypothesis
  as I1's explicit domain.
- `piecewise_linear_2d` — now declares its output range to the
  mathematically exact grid bound.
