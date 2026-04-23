# Phase B, Part 4 — Validation tail

Lightweight validation after Part 3 lands: walkthrough reference
matches, noise pipeline re-run, compiled depth recorded, brief
completion note appended to `phase_b.md`.

## Prerequisites

- Part 3 landed. WALL prefill is a data carrier only; RESOLVED
  reads from thinking tokens; SORTED emits wall_index as a VALUE
  token; RENDER reads wall identity and `vis_lo` / `vis_hi` via
  attention.

## Development speed

Part 4 is deliberately small. No formal completion ceremony, no
comparison tables, no dilution diagnostics (the SORTED softmax
dilution from M4 is moot after Part 2's attention rewrite —
verify empirically via walkthrough only). If any check fails,
fix in the responsible part and re-run; do not build out
elaborate diagnostic infrastructure here.

Total work budget: a few hours of running commands and updating
two files.

## Scope

### 1. Walkthrough reference matches

Run:

```
make walkthrough ARGS="--scene box --frames 10"
make walkthrough ARGS="--scene multi --frames 10"
```

Both must produce reference-matching GIFs within existing
tolerances. If a mismatch surfaces, diagnose back to the
responsible part (Part 1, 2, or 3) and fix there.

### 2. Noise pipeline re-run

Phase B modifies several op call-sites:

- Part 1 restructures the `_compute_clip_and_project` math (same
  ops, different positions; call-site distribution changes).
- Part 1 restructures `factor_q_to_embedding` (hi/lo split;
  `thermometer_floor_div` distribution changes).
- Part 1 adds running-accumulator emits for `HIT_*` (introduces
  new call-sites for the saturate + Linear path).
- Part 2 adds scalar-channel emits at BSP_RANK.
- Part 3 deletes the prefill WALL visibility + collision chains
  (many call-sites disappear).

Per CLAUDE.md's numerical-noise doctrine (D7):

- Run `make measure-noise`. This regenerates
  `docs/op_noise_data.json` and the auto-generated sections of
  `docs/numerical_noise.md` plus op docstring footers.
- Diff `docs/op_noise_data.json` against the pre-Phase-B version.
  For each material change, update
  `docs/numerical_noise_findings.md`:
  - New `doom_*` distribution appeared → add an entry.
  - A measured number grew by more than 2× → update the entry.
  - Rank order of distributions shifted → update.
  - A previously-flagged bound is now within expectations →
    remove the stale entry.
  - A removed call-site's distribution disappeared → remove
    the stale entry.

If nothing materially changed, the findings file is untouched and
no commit is needed for it.

### 3. Compiled depth recorded

Run `make graph-stats`. Record:

- Final compiled layer count.
- The new critical path (which ops, which stage).
- Which cross-position bottleneck now drives depth (if any).

The pre-Phase-B baseline is 115 layers with the critical path
entirely inside one thinking step (`t_and_vis` 39 ops +
`select_and_factor` 19 ops). Phase B expects the critical path to
shift — either to a shallower thinking step, to RENDER's internal
chain, or to a new chain involving the quadratic-equality
attention. The actual number depends on how RENDER parallelizes
against the shortened thinking chain.

### 4. Completion note

Append a short "Phase B complete" section to `docs/phase_b.md`
with:

- Five or fewer bullets, one per shipped change (thinking-slot
  split, running accumulators, factor cascade, SORTED/RENDER
  restructure, prefill WALL gutting).
- Final compiled depth number.
- Two or three bullets on what remains for Phase C.

Not a formal summary — a paragraph-and-a-half. Enough that
someone landing on `docs/phase_b.md` six months later can see at
a glance what Phase B achieved and what it left for the next
phase.

## Not in scope

- SORTED softmax dilution diagnostic from M4 / Phase A. The old
  `attend_argmin_above_integer` primitive (the dilution source)
  is gone after Part 2. If dilution-like symptoms appear in the
  walkthroughs, diagnose then.
- Affine-bounds looseness fix from M4 / Phase A. The geometry-
  attention outputs in thinking_wall that were the looseness
  source now read raw geometry directly after Part 3. If
  `test_affine_bounds` still regresses post-Phase-B, diagnose
  then.
- Test coverage expansion beyond smoke walkthroughs.
- Any Phase C work.
- Full rewrite of `docs/doom_graph.md` to reflect Phase B state.
  That's a bigger doc-writing task and belongs separately (the
  doc was written for pre-thinking-token architecture and needs
  full reauthoring, not patching).

## Open questions

- **M4 regression triage.** Two M4 issues were expected to
  surface during Phase A Part 5's validation: SORTED softmax
  dilution and `test_affine_bounds` looseness. Both have
  architectural roots that Phase B removes. Expectation: neither
  surfaces in the walkthroughs. If either does, the fix belongs
  in whichever part introduced the new mechanism — not here.

## High-level task list

1. Run box + multi walkthroughs; confirm reference match.
2. Run `make measure-noise`; diff the JSON; update
   `docs/numerical_noise_findings.md` for material changes only.
3. Run `make graph-stats`; record final compiled depth, critical
   path, and bottleneck.
4. Append a 5-bullet "Phase B complete" note to
   `docs/phase_b.md`.
