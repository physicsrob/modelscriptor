# Communication

## Use plain English; reintroduce terms on every use

When explaining technical concepts, describe the mechanics in plain
English rather than introducing named abstractions. Say "the input
range crosses zero" not "straddling." Say "the slope of the line
connecting the endpoints" not "the chord relaxation." If a term
doesn't already exist in the codebase, prefer the description — the
user will name it if it needs a name.

When a named term genuinely earns its keep (you'll reference the
concept many times and the name saves confusion), define it inline on
every use until the user starts using it themselves — that's the
signal they've adopted it. "The PL-drift (the gap between the
piecewise-linear approximation and the exact function) compounds
through..." not just "The PL-drift compounds through..."

Never stack coined terms. "The chord relaxation of the straddling ReLU
in the forward-mode LiRPA" is four layers of undefined vocabulary.
Each layer of jargon you build on top of another layer compounds
confusion. If you need multiple concepts, introduce them one at a time
with plain-English definitions between them.

The user manages multiple projects and does not have your earlier
definitions loaded. Write every explanation so it can be understood
cold.

## Admit uncertainty; don't fill gaps with plausible stories

When you aren't sure whether two things are really the same, whether a
mechanism works the way you think, or whether a number is right — say
so. "I think these might be the same thing but I'm not sure" is always
better than treating them as interchangeable and building an
explanation on top. Check the code before building an explanation on
any factual claim (a constant's value, what a function reads, how a
data structure is used). Never construct a narrative that "sounds
right" without tracing the actual code path — the most dangerous
explanations are the ones that are internally consistent but don't
match reality.

## Flag complexity before building it

Before introducing a new abstraction, indirection layer, or
deferred-execution pattern, flag it to the user: what it is, why you
think you need it, and what the simpler alternative would be. "I'm
about to add a placeholder system because the basis needs to be fully
known before computing bounds — the simpler alternative is making the
basis mutable. The placeholder approach is more complex but avoids
changing Basis. Which do you prefer?" Don't commit to elaborate
machinery without explicit agreement. If the user says "that feels
gross," trust that instinct and look for the direct path.

# Testing

## Running Tests

ALWAYS use `make test` to run tests. NEVER invoke pytest directly.

    # Run all tests (auto-sharded across A100 GPUs on Modal)
    make test

    # Run a specific test file (single container, no sharding)
    make test FILE=tests/graph/test_embedding.py

    # Run tests matching a keyword
    make test ARGS="-k test_foo"

    # Combine file and args
    make test FILE=tests/compile/ ARGS="-k forward"

    # Run on CPU only
    make test ARGS="--device cpu"

## Running Tests Locally

`make test-local` runs pytest on the **local machine** (no Modal), for
fast iteration on a single file without the Modal round-trip.

    # REQUIRED: FILE must point at a single test file
    make test-local FILE=tests/graph/test_embedding.py

    # Pass extra pytest args
    make test-local FILE=tests/graph/test_embedding.py ARGS="-k foo -v"

**FILE is mandatory.** The target refuses to run without it — this is
intentional to prevent accidentally running the full suite (or a whole
directory) locally, which can saturate the local GPU, take far longer
than the Modal sharded run, and produce misleading results. Full-suite
and directory-level runs belong on Modal via `make test`.

## Critical Rules

- NEVER run tests in the background. Always foreground, always wait for completion.
- NEVER run pytest directly. `make test` includes a cross-session mutex lock.
- NEVER run tests in parallel (no pytest-xdist, no &, no background execution).
- NEVER re-run tests just because you lost output (e.g. you piped through
  `| tail -10` and now want to grep). If the code hasn't changed, the previous
  run's full output is in the log file — `make test` prints its path at the
  start and end, and `/tmp/torchwright-test.log` symlinks to the latest run.
  Grep that file instead of spending another ~90s re-running the suite.

## How Test Sharding Works

`make test` runs the full suite across 4 independent A100 GPU containers
on Modal.  Each container runs a subset of tests with exclusive GPU access.

**Why sharding?** The DOOM renderer tests call `compile_game(d=2048)` which
builds a ~8GB transformer model on GPU.  Sharding gives each compiled-test
file its own GPU so compilations run in parallel.

**How it's configured** (in `modal_test.py`):

- `_HEAVY_FILES` — test files with large `compile_game()` calls get their
  own container.
- `_MEDIUM_FILES` — other GPU test files grouped into one container.
- Everything else goes into a catch-all shard automatically.

**When you add new tests:**

- New test files anywhere under `tests/` are picked up by the catch-all
  shard automatically.  No config changes needed.
- If a new file calls `compile_game(d=2048)` and is slow enough to warrant
  its own container, add it to `_HEAVY_FILES` or `_MEDIUM_FILES`.

**When using `FILE=`**, sharding is bypassed — the file runs in a single
container.  `-k` filters passed via `ARGS=` are applied to every shard.

## Performance Expectations

Full suite (`make test`): ~90s wall time (4 shards of ~30-60s each, plus
Modal container orchestration overhead).

Single file (`make test FILE=...`): depends on the file.
- Fast tests (ops, compile/forward, graph): 10-30s
- Compiled DOOM tests (test_game_graph.py): ~35s
- Compiled wall selection (test_wall_selection.py): ~55s

## Writing Tests That Use compile_game()

Tests calling `compile_game()` are expensive (~17s to compile, ~2s per
`step_frame` inference on A100).  Follow these patterns:

1. **Use class-scoped fixtures** to share the compiled module across tests
   in the same class.  See `TestGameGraph` and `TestCompiledStructure` for
   examples.

2. **Don't pass `device="cpu"`** to `compile_headless()` or `compile_game()`.
   The default is `"auto"` which uses GPU when available.  Forcing CPU will
   make inference ~8x slower.

# Numerical noise

Every approximate op in `torchwright/ops/` is measured against its exact-math
reference and the numbers are committed alongside the code. `docs/op_noise_data.json`
is the canonical source; `docs/numerical_noise.md` and the per-op docstring
footers are **generated** from that JSON. Never edit them by hand.

Commentary on the measurements — which numbers are by design, which deserve
investigation, and which DOOM call-sites each distribution covers — lives in
`docs/numerical_noise_findings.md`. That file is **hand-written and is
Claude's responsibility to keep current**. The measurement pipeline does not
regenerate it and the consistency test does not enforce it; synthesizing
findings from the raw numbers is an interpretive task that belongs to the
agent running the workflow, not to the script.

Specifically, whenever you run `make measure-noise` — whether because you
added an op, edited one, or widened a distribution — you must:

1. Diff the new `docs/op_noise_data.json` against the previous commit.
2. For each material change (number grew, rank order of distributions
   changed, a finding-worthy input emerged), either update the corresponding
   entry in `docs/numerical_noise_findings.md` or add a new one.
3. Remove findings that no longer apply (e.g., if a bound is tightened and
   the previously-flagged number is now within expectations).
4. Keep the call-site cross-reference table in sync when you add, rename, or
   remove a `doom_*` distribution.

Regenerate the auto-artefacts with:

    make measure-noise

**Any change to a piecewise op's implementation, breakpoint grid, or
`step_sharpness` requires a re-measurement before you commit.** The
consistency test only verifies that JSON, markdown, and footers agree with
each other — it cannot tell that the numbers were measured against old code.
Skipping `make measure-noise` after editing an op leaves stale numbers in the
docstring and markdown with a commit SHA that no longer matches HEAD.

Workflow when you change a piecewise op's implementation or breakpoint grid:

1. Run `make measure-noise` to regenerate the JSON, markdown, and docstring
   footers at the new commit.
2. Review the diff in `docs/op_noise_data.json` and update
   `docs/numerical_noise_findings.md` to reflect anything newly surprising
   or newly resolved.
3. `git diff docs/ torchwright/ops/` — the only auto-generated changes
   should be noise numbers and commit SHAs; the findings-doc changes are
   yours.
4. Commit.

The consistency test `tests/docs/test_numerical_noise_consistency.py` fails in
CI if the three artefacts drift out of sync. The error message points back to
`make measure-noise`.

To add a new op, append a `TargetOp(...)` to `_target_ops()` in
`scripts/measure_op_noise.py`. See the "Adding a new op" section at the end
of `docs/numerical_noise.md` for the full pattern.

# Walkthrough

## Rendering a Walkthrough

Use `make walkthrough` to compile the DOOM game graph on a Modal A100 and
generate a GIF walkthrough.  This produces both a transformer-rendered
`walkthrough.gif` and a reference-rendered `reference.gif`, then opens
the walkthrough GIF.

    # Render 10 frames (default)
    make walkthrough

    # Render a specific number of frames
    make walkthrough ARGS="--frames 5"

    # Use the multi-room scene instead of the default box room
    make walkthrough ARGS="--scene multi"

    # Combine options
    make walkthrough ARGS="--frames 20 --scene multi"

All flags (`--width`, `--height`, `--fps`, `--scale`, `--d`, `--d-head`,
`--rows-per-patch`, `--tex-size`) are passed through via `ARGS=`.

## Critical Rules

- NEVER pipe `make walkthrough` through `tail`, `head`, or any other
  output-truncating filter.  The user always wants to see the full
  output.
- NEVER re-run `make walkthrough` just because you lost output (e.g.
  you piped through `| tail -10` and now want to grep).  If the code
  hasn't changed, the previous run's full output is in the log file —
  `make walkthrough` prints its path at the start and end, and
  `/tmp/torchwright-walkthrough.log` symlinks to the latest run.
  Grep that file instead of spending another render cycle.

# Running scripts on GPU

**If a script needs a GPU, run it on Modal via `make modal-run`.**
Never write a new `modal_*.py` file just to run a script remotely —
that is the Modal equivalent of the ad-hoc `/tmp/` scripts D8 warns
against.

    # Run a committed module (preferred)
    make modal-run MODULE=scripts.investigate_phase_e

    # Pass args through
    make modal-run MODULE=scripts.foo ARGS="--input bar"

    # Run an arbitrary file
    make modal-run SCRIPT=path/to/one_shot.py

    # CPU-only shard (no GPU reservation)
    make modal-run MODULE=scripts.some_cpu_job CPU_ONLY=1

## When NOT to use modal-run

- **Tests** — use `make test`.  Its sharding + mutex + log-file
  plumbing is not reproduced by `modal-run`.
- **Walkthrough renders** — use `make walkthrough`.  It syncs GIF
  bytes back to the local working tree, which `modal-run` does not
  do.
- **Scripts that produce local artifacts** (GIFs, JSON files under
  `docs/`, etc.).  `modal-run` captures stdout/stderr only; anything
  the script writes to disk stays on the Modal worker.  If your
  script needs artifact sync-back, that is the *only* acceptable
  reason to add a new purpose-built `modal_*.py` entrypoint — and
  when you do, import the image from `modal_image.py` rather than
  duplicating it.

## Critical rules

- NEVER write a new `modal_*.py` at the repo root just to run a
  one-off investigation.  Put the script under `scripts/` (or
  `tests/` if it's really a test) and run it via `make modal-run`.
- NEVER duplicate the Modal image definition.  Import `IMAGE` from
  `modal_image.py`.
- NEVER re-run `make modal-run` just because you lost output (e.g.
  you piped through `| tail -10` and now want to grep).  If the
  script hasn't changed, the previous run's full output is in the
  log file — `make modal-run` prints its path at the start and end,
  and `/tmp/torchwright-modal-run.log` symlinks to the latest run.
  Grep that file instead of spending another Modal round-trip.

# Doctrine

The DOOM renderer project has a recurring failure mode: ship a
99%-working thing, build on top, and have the 1% bite later. The
rules below exist to defeat that pattern. They constrain how to
investigate, what to ship, what to defer, and what to write in
xfail reasons.

**Pending tooling links.** Some doctrines below reference tooling
or reference docs that have not yet been built (Plans 1–3 in
`/tmp/DERISK_PLAN.md`).  Those are flagged inline as
`[TBD: link to <path> when Plan N lands]`.  If you encounter a
`[TBD:` marker and the named artifact now exists, replace the
marker with the actual link.

## D1 — Suspected-compiler-bug protocol

**A suspected compiler bug stops all other work.** Don't reshape
user code to route around it.

**Triggers** (any one): a reproducible value mismatch that no
per-op error budget can explain; residual corruption after a
topology-only change to the user graph; output that violates a
stated compiler invariant.

**Why.** Routing around a compiler bug leaves a landmine for the
next user. Every "I packed it differently and the bug went away"
fix is one we'll re-encounter, harder to debug, somewhere else.

**Worked example.** Phase E (commit `c2d5a7a`) attempted three
rounds of routing around suspected residual-column overlap by
reshaping user code (separate output → packed payload →
packed-but-narrow-extract).  None resolved the root cause; the bug
surfaced again as the (3, 2, 20) xfail at
`tests/doom/test_game_graph.py:146`.  See `/tmp/plan-e.md` for the
full attempt log.

**Escalation.** Inform the user immediately with the specific
trigger that fired and ask for guidance.  Do not proceed with
workarounds unilaterally.

**Tooling.** See the *Compiler Invariants* section below — the four
runtime-asserted invariants are the canonical list of "stated
compiler invariants" for this trigger.

## D2 — Never defer numerical problems

**Off-by-an-unexpected-amount has exactly one acceptable answer:
the bit-level reason for the divergence.** "I don't know yet,
investigating" is also acceptable — it admits ignorance honestly.
A plausible-sounding guess is *not* acceptable, because guesses
look like understanding.

**Why.** Numerical bugs compound.  A guess shipped today becomes
tomorrow's assumed explanation, and the real bug grows another
layer of camouflage.

**Worked example.** The Phase E xfail at
`tests/doom/test_game_graph.py:146` was shipped with the reason
*"likely due to compile-side precision loss in the per-wall
is_renderable gate at geometry that lands near the
attention-edge-of-view."* The actual smoking gun (a deterministic
`-1000 == -_ABOVE_BONUS` output from the SORTED stage) was never
investigated.  The reason is a guess wearing the costume of an
explanation.

**Tooling.** `torchwright/debug/probe.py` runs a compiled module
side-by-side with the recursive graph oracle and reports the first
node whose compiled value diverges.  Use it as the first step on
any unexplained divergence.  [TBD: link to the per-op
numerical-noise reference once Plan 3 lands.]

## D3 — Understanding rule

**If you can't explain a behavior's root cause in one sentence
without hand-waving, you don't understand it.** The one-sentence
test applies whenever you describe a bug, write an xfail reason,
fill in a postmortem, or tell the user "this code does X because
Y."  Research until the sentence compresses without hedges.  If
the doc that would have let someone else write the sentence is
missing, add it.

**Why.** Compressing the cause to one sentence is the diagnostic.
If the sentence won't compress, the cause isn't known yet.

**Worked example.** The Phase E xfail reason wraps three hedged
conjunctions into one sentence ("likely … near … under …").  That
*and-of-maybes* structure is the warning sign — when only an
and-of-maybes will fit, the sentence is hiding ignorance.

## D4 — Foundation rule

**Never move on if the foundation isn't 100% solid.** An
un-investigated anomaly in phase N is the first task of phase
N+1, not a footnote.

**Why.** Every layer added on top of an anomaly multiplies the
cost of going back.  The cost of fixing Phase E grows with every
downstream change to the SORTED stage.

**Worked example.** The Phase E xfail was shipped to unblock
downstream phases.  By the time the (3, 2, 20) regression is
investigated, the xfailed test will no longer be the only code
touching the SORTED above-threshold primitive — the cost of
reverting or re-architecting has gone up.

## D5 — xfail hygiene

**No `xfail` without a precisely documented root cause.** Two
acceptable forms:

1. `xfail(reason="precise root cause: X; will be fixed by Y",
   strict=True)` — root cause known, fix deferred for a stated
   reason.
2. `xfail(reason="unknown, investigating, linked to issue N",
   strict=True)` — root cause not yet known, but a tracked
   follow-up exists.

Unacceptable: `xfail(reason="likely due to <guess>")` with no
evidence and no follow-up.

**Why.** An xfail with a guessed reason isn't a TODO — it's a
trap.  The next contributor reads the reason, takes it as an
explanation, and stops looking.

**Worked example.** `tests/doom/test_game_graph.py:146` shipped
with the unacceptable form.  It must be replaced by form 1 or
form 2 before the test can be shipped again as a known
limitation.

**Tooling.** `torchwright/debug/probe.py` is what you use to
convert form 2 into form 1.

## D6 — Reproducer-before-fix

**Every bug becomes a permanent unit test at the smallest
reproducing layer** — not the integration test that surfaced it,
the smallest layer that still reproduces it.  A render mismatch
caused by an op error becomes an op test, not a render test.
(Trivial fixes — typos, comment changes — have no reproducing
layer; this rule applies to behavior bugs.)

**Why.** Integration tests that catch bugs are slow, indirect,
and easily broken by unrelated changes.  Smallest repros are
fast, direct, and survive refactors.

**Worked example.** Phase E's (3, 2, 20) regression surfaced as a
render test (slow, system-level).  The smallest repro is a
SORTED-stage call to `attend_argmin_above_integer` with the
specific `indicators_above` input that fails to concentrate.  The
stage-level test belongs in `tests/doom/stages/test_sorted.py` or
below.

**Tooling.** `torchwright/debug/probe.py` to identify which
layer / which node / which inputs reproduce the bug; `make
test-local FILE=...` to iterate fast against the resulting unit
test.

## D7 — Per-op noise sync

**Modifying a piecewise op's implementation or breakpoint grid
requires re-measuring its noise bound and updating its docstring
in the same commit.** When a consolidated noise reference exists,
update it too.

**Why.** Stale noise bounds are the supply chain for stale
assumptions.  If `compare`'s bound moves from 1e-3 to 5e-3
silently, every downstream stage that budgeted against 1e-3 is
now over-budget without anyone knowing.

**Worked example.** Phase E raised `_ABOVE_BONUS` from 100 to
1000 to give the SORTED softmax more headroom.  That changes the
piecewise softmax's effective working range and may change its
measured error bound.  Neither was re-measured at the time.

**Tooling.** Op docstrings under `torchwright/ops/` are today's
source of truth — keep them current.  [TBD: link to the
consolidated numerical-noise reference and the proposed
docstring/reference-doc consistency check once Plan 3 lands.]

## D8 — Tooling sources of truth

**Use the established tooling; do not reinvent.** Ad-hoc debug
scripts in `/tmp/` are write-once, never indexed, never
re-runnable, and don't accumulate institutional knowledge.

- **Probing residual values / divergence:**
  `torchwright/debug/probe.py`.  [TBD: extend the description
  once Plan 1's generalized harness — residual inspection,
  attention inspection, layer-wise diff — lands.]
- **Compiler invariants:** assertions in
  `torchwright/compiler/`.  See the *Compiler Invariants* section
  below for the canonical list.
- **Per-op precision budgets:** op docstrings under
  `torchwright/ops/`.  [TBD: link to the consolidated noise
  reference once Plan 3 lands.]
- **Adversarial integration coverage:**
  `tests/doom/test_game_graph.py`.  [TBD: link to the parametric
  sweep section once Plan 4 lands.]
- **Running a committed script on a GPU:** `make modal-run
  MODULE=<dotted.name>` (see *Running scripts on GPU* above).
  Writing a new `modal_*.py` at the repo root to run a
  one-off is banned by the same rule that bans `/tmp/` probes.

**Why.** `tests/doom/test_mode_c_probe.py` started life as an
ad-hoc probe and grew to 1000+ lines hard-coded to `angle=192`.
The cost of generalizing it later (Plan 1) is exactly the cost
of having let the ad-hoc form persist.  Don't repeat the
pattern.

# Compiler Invariants

The forward-compile pipeline guarantees the four invariants below.
Each is enforced at runtime by an `AssertionError` inside the
compiler — negative unit tests in
`tests/compile/forward/test_compiler_assertions.py` pin the error
shape.  These are the canonical "stated compiler invariants"
referenced by doctrine D1.

**Absolute rule.** If one of these assertions fires on the existing
test suite, **STOP**.  That is a real compiler bug.  Do NOT weaken
the assertion, do NOT `try/except` around it, do NOT xfail the
affected test.  Follow D1: report the firing assertion, the test,
and `git rev-parse HEAD` to the user and wait for guidance.

Also: **do not add new assertions here without a matching negative
test** in `tests/compile/forward/test_compiler_assertions.py`.  The
pair (assertion + negative test) is what keeps the invariant honest
across refactors.

## I1 — Allocator self-consistency

`ResidualStreamMap`'s internal state is consistent after every
mutation:

1. Pairwise disjoint — no column appears in two nodes' index lists.
2. `_free ∩ allocated == ∅`.
3. `_free ∪ allocated == {0 .. d-1}`.

**Enforced in** `torchwright/compiler/forward/residual_map.py`:
`ResidualStreamMap._check_invariants`, called at the end of
`allocate`, `free`, and `reassign`.  `allocate` also runs a
pre-commit uniqueness check so a firing assertion names the
conflicting node.

**What a fire means.** Either (a) allocator code was edited and no
longer preserves the invariant, or (b) an external caller reached
in and mutated `_free` / `_node_to_indices`.  Both are bugs.

**Equivalently: this is the invariant that forbids *residual-column
aliasing* among simultaneously-live nodes.** If you suspect a bug
where two live nodes share a column and their values contaminate
each other, that hypothesis is "I1 is firing."  Since `_check_invariants`
runs after every allocate/free/reassign, if the test suite compiles
without I1 firing then no such aliasing exists — the bug is elsewhere
(candidates: stale residual-stream values at reassigned columns
despite clean allocator bookkeeping; compound numerical drift through
a long op chain; schedule ordering across sublayers).  Do not restate
this hypothesis without first checking that I1 is not fired on the
repro.

## I2 — Literal stability

Every `LiteralValue` scheduled via `compute_literal_value`, and
every `Linear` bias scheduled via `compute_bias`, has
`len(op.target_cols) == node.value.numel()` (or
`node.output_bias.numel()`).  Writes never silently truncate.

**Enforced in**
`torchwright/compiler/forward/scheduler.py` at emission time
(around `compute_literal_value`), and
`torchwright/compiler/forward/weight_writer.py` at
`_write_compute_literal_value` and `_write_compute_bias`.

**What a fire means.** Allocation width drifted from the source
tensor's numel — either scheduler logic changed without updating
the invariant, or a `LiteralValue` / `Linear` was constructed with
inconsistent width.  Removing the assertion would reintroduce the
pre-invariant silent `[: len(target_cols)]` slice that masked the
bug.

## I3 — Attention Q/K/V/O row-width correctness

At `_write_compute_attn` entry, the captured column indices match
the `Attn` node's declared input / output widths:

- `len(q_source_cols) == len(query_in)`
- `len(k_source_cols) == len(key_in)`
- `len(source_cols)   == len(value_in)`   (V)
- `len(target_cols)   == node.d_output`   (O)

**Enforced in**
`torchwright/compiler/forward/weight_writer.py:_write_compute_attn`.

**What a fire means.** The scheduler captured the wrong columns
for this attention op.  Most likely a `Concatenate`-resolution
bug dropped or duplicated a leaf.  The attention head's Q/K/V
rows would otherwise be scattered to the wrong positions —
silent value corruption.

## I4 — Column liveness

A node's residual columns stay allocated until every effective
consumer (transparent through `Concatenate`) has been computed.
Two enforcement points:

- **Schedule-time (always on):** every source-column capture in
  `LayerScheduler` — `compute_linear`, `compute_attn` Q/K/V,
  `compute_add` a0/a1, `add_into` dead/live addends,
  `compute_relu` L1 input, `compute_standalone_relu` input —
  first calls `LayerScheduler._require_live(node, rmap, op_label)`
  which walks through `Concatenate` leaves.
- **End-of-layer (gated behind `TW_COMPILER_VERIFY=1`):**
  `compile._verify_end_of_layer_liveness` walks every computed
  non-`Concatenate` node after `write_mlp_sublayer`; if any has
  uncomputed effective consumers and is no longer allocated, it
  raises.

**Enforced in**
`torchwright/compiler/forward/scheduler.py:_require_live` and
`torchwright/compiler/forward/compile.py:_verify_end_of_layer_liveness`.

**What a fire means.** Something freed a node's columns before
all its consumers ran.  Without the assertion the symptom would
be a `KeyError` deep inside `get_indices` with no op context, or
(worse) silently reading stale residual values from the reclaimed
columns.  Always-on for schedule-time because every read needs a
live source; gated for end-of-layer because the walk is
`O(|nodes| · fanout)`.
