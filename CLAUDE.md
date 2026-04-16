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

**Tooling.** [TBD: link to the compiler-invariants reference once
Plan 2 lands.]

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
  `torchwright/compiler/`.  [TBD: link to the compiler-invariants
  reference once Plan 2 lands.]
- **Per-op precision budgets:** op docstrings under
  `torchwright/ops/`.  [TBD: link to the consolidated noise
  reference once Plan 3 lands.]
- **Adversarial integration coverage:**
  `tests/doom/test_game_graph.py`.  [TBD: link to the parametric
  sweep section once Plan 4 lands.]

**Why.** `tests/doom/test_mode_c_probe.py` started life as an
ad-hoc probe and grew to 1000+ lines hard-coded to `angle=192`.
The cost of generalizing it later (Plan 1) is exactly the cost
of having let the ad-hoc form persist.  Don't repeat the
pattern.
