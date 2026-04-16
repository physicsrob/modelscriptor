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
