# Phase A, Part 5 — Validation & carryover

Closure, not new behavior. Part 5 diagnoses and fixes the M4
regressions carried forward, runs the numerical-noise pipeline per
CLAUDE.md, measures compiled depth, validates rendered output on
canonical scenes, and enumerates what Phase A explicitly leaves for
future phases.

Some of Part 5's work is continuous across Parts 2–4 rather than
strictly sequential. Measurement tools like `make graph-stats` and
`make measure-noise` are run as each part lands, not only at the end.
Part 5 is where the *acceptance* checking formalizes, but the checks
themselves shouldn't wait.

## Strong opinions this part commits to

### Part 5 ships closure, not behavior

No new stage, no new attention pattern, no new token. The thinking
architecture is fully landed after Part 4; Part 5 validates that it
behaves and cleans up the carryovers.

If an issue surfaces that requires a new mechanism, it belongs in
Part 1–4, not here. Part 5's scope is bounded to fixes for already-
known regressions, measurement against already-defined budgets, and
documentation of already-visible gaps.

### M4 regressions are diagnosed first, mechanism second

Two known regressions from M4:

- **SORTED softmax dilution** — `test_pipeline.py::test_frame_matches_reference_off_center[3.0-2.0-20]`
  fails on Modal GPU. 56 extra thinking positions (M4) dilute the
  `attend_argmin_above_integer` softmax in exhaustion detection.
  With Parts 2–4 landed, the thinking phase grows to ~222 positions,
  so the symptom likely worsens.
- **test_affine_bounds looseness** — geometry-attention outputs
  (sel_ax, sel_bx, ...) in thinking_wall have looser bounds than
  raw WALL fields. Pure tightness, not runtime-correctness.

Part 5 diagnoses each to a concrete root cause before selecting a
fix. The fix approach is deliberately not pre-specified. Candidates:

- SORTED: increase match_gain, explicit masking of thinking
  positions, or restructure the exhaustion signal. Pick after root-
  cause diagnosis.
- Affine bounds: tighten the propagator itself, or tighten per
  call-site. Pick based on fanout observed after Parts 2–4 land.

Each regression gets a regression test under `tests/` that fails
pre-fix and passes post-fix. Fixes without regression tests don't
count.

### The SORTED dilution fix may need to land inside Part 4

If the dilution is severe enough in Part 4 that the Part 4
integration test (collision scenario rendered correctly) can't pass
without the fix, the fix lands in Part 4. Part 5 then just records
"already done in Part 4" for this item and proceeds with the rest.

This is a pragmatic allowance: Part 4 needs to ship atomically and
pass acceptance, so if a blocker surfaces, fix in place. Don't hold
Part 4 open waiting for Part 5 formality.

### The noise pipeline is re-run per CLAUDE.md

After Part 3 (when new op call-sites land) and after Part 4 (when
RESOLVED aggregation adds another distribution), run
`make measure-noise`. Diff `docs/op_noise_data.json` against the
pre-Phase-A commit. For each material change, update
`docs/numerical_noise_findings.md`.

"Material change" is a judgment call:
- A measured number grows by more than 2× its prior value → update.
- A new `doom_*` distribution appears → add an entry.
- Rank order of distributions changes → update.
- A previously-flagged bound is now within expectations → remove
  the stale finding.

If nothing materially changed after Parts 3–4, the file is
untouched and no commit is needed.

### Compiled depth is measured, not chased

Phase A does not ship a depth reduction. The design-doc target of
~25–30 compiled layers requires the prefill WALL stage to be
deleted, which is post-Phase-A. In Phase A, prefill WALL stays, so
its ~30-layer visibility-columns stage + RENDER precompute + new
thinking identifier steps combine to produce a compiled depth in the
70–100 layer range.

Acceptance for Part 5:
- Record the post-Phase-A compiled depth via `make graph-stats`.
- Confirm no regression from pre-Phase-A baseline (same scene,
  compiled with main at the merge base).
- Document the gap explicitly — where the current depth comes from,
  what reducing it requires.

### Canonical scene coverage is sufficient

Part 5 validates:
- `make test` passes the full suite on Modal.
- `make walkthrough ARGS="--scene box --frames 10"` matches
  reference.
- `make walkthrough ARGS="--scene multi --frames 10"` matches
  reference.
- At least one collision-scenario test from Part 4 passes.

Not expanding the test matrix beyond this unless Parts 1–4 surface
edge cases warranting specific regression scenes.

### What Phase A does not deliver

Enumerated explicitly so Phase A's scope is unambiguous:

- **Prefill WALL stage is not deleted.** It still computes
  collision flags, bsp_rank, is_renderable, vis_lo/hi — consumed by
  SORTED, RESOLVED aggregation, and RENDER.
- **Compiled depth is not reduced to the design-doc target.** Same
  ballpark as pre-Phase-A.
- **RENDER's overlaid state (col, chunk_k, wall_counter) remains.**
  Per CLAUDE.md the per-pixel-output phase eliminates this, post-
  Phase-A.
- **WALL-as-identifier-value-pairs prefill refactor is not done.**
  WALL stays as a single-token-per-wall prompt encoding.
- **The "aggregate values across walls" helper is not built.**
  RESOLVED in Phase A uses prefill WALL; the thinking-token-only
  path waits on prefill WALL deletion.
- **Seven thinking values (BSP_RANK, IS_RENDERABLE, VIS_LO,
  VIS_HI, HIT_FULL, HIT_X, HIT_Y) compute but are never consumed
  downstream in Phase A.** Their correctness is validated via dual-
  path tests in Part 3; their use waits on prefill WALL deletion.

## Scope

- Diagnose root cause of SORTED softmax dilution; implement fix;
  add regression test.
- Diagnose affine-bound looseness in thinking-wall's geometry
  attentions; tighten propagator or per-site as appropriate;
  add regression test.
- Run `make measure-noise` after Parts 3–4 land; update
  `docs/op_noise_data.json` + `docs/numerical_noise_findings.md`
  per material change.
- Run `make graph-stats`; record compiled depth; compare against
  pre-Phase-A baseline.
- Run `make test` on Modal; confirm full suite passes.
- Run `make walkthrough` on box + multi scenes; confirm reference
  match.
- Write a "Phase A completion" section in the master plan (or a
  dedicated brief) listing (a) what shipped, (b) what was explicitly
  deferred, and (c) the measured numbers (depth, noise, test pass
  rates).

## Not in scope

- Any new thinking-token mechanism.
- Deletion of the prefill WALL stage.
- Reduction of the compiled-layer count.
- Improvements to stages that aren't directly regressed.
- Master plan / index doc (`phase_a_plan.md`). That's a separate
  follow-on after Part 5 ships.

## Acceptance criteria

1. Both M4 regressions (SORTED dilution, affine bounds) are
   resolved. Each has a regression test that fails on a pre-fix
   checkout and passes after the fix.
2. `make measure-noise` output is in sync with
   `docs/numerical_noise_findings.md`. `docs/op_noise_data.json`
   matches the current code via the existing drift test.
3. `make graph-stats` is run on the post-Phase-A compiled module;
   the number is recorded in the part-5 doc (or a successor brief).
   No regression in compiled depth vs. the pre-Phase-A baseline on
   the same scene.
4. `make test` passes the full suite on Modal.
5. `make walkthrough ARGS="--scene box --frames 10"` and
   `make walkthrough ARGS="--scene multi --frames 10"` both match
   reference within existing tolerances.
6. Part 4's collision-scenario integration test passes.
7. A "Phase A completion" summary exists — listing what shipped,
   what was deferred, and the measured numbers.

## Open questions

### Exact fix for the SORTED softmax dilution

The root-cause diagnosis determines the fix. Candidate mechanisms
listed above, but the right one depends on what the measurements
show after Parts 2–4 land.

### Affine-bound fix strategy

Tighten the propagator (one compiler change, wide impact) or
per-site (many small changes, local impact). Decide after observing
the post-Part-4 violation profile. If only a few call-sites violate,
per-site is cheaper; if it's fanning out wide, fix the propagator.

### Material-change threshold for noise pipeline

CLAUDE.md delegates this to judgment. Part 5 exercises that
judgment once, based on the actual diff.

### Follow-up work ordering post-Phase-A

Part 5 enumerates what Phase A doesn't do. Prioritizing the
follow-on phases (prefill WALL deletion, RENDER per-pixel output,
WALL-as-identifier-pairs) is a separate planning conversation, not
Part 5's job.

## High-level task list (not exhaustive)

1. Diagnose SORTED softmax dilution root cause. Likely requires
   running the failing test on GPU, capturing softmax weight
   distributions at SORTED's attention with `probe_attention`.
2. Implement SORTED fix; add regression test.
3. Diagnose affine-bound looseness in thinking-wall. Compare
   propagator output on geometry-attention outputs vs. raw WALL
   fields. Identify whether the propagator is losing tightness or
   whether the attention itself is inherently loose.
4. Implement affine-bound fix; add regression test.
5. Run `make measure-noise`. Diff `docs/op_noise_data.json` against
   pre-Phase-A baseline. Update
   `docs/numerical_noise_findings.md` for each material change.
6. Run `make graph-stats`. Record compiled depth. Compare against
   baseline.
7. Run `make test` on Modal. Confirm full-suite pass.
8. Run `make walkthrough` on box + multi scenes. Confirm reference
   match.
9. Write Phase A completion summary — what shipped, what deferred,
   measured numbers.
