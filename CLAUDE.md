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

## Critical Rules

- NEVER run tests in the background. Always foreground, always wait for completion.
- NEVER run pytest directly. `make test` includes a cross-session mutex lock.
- NEVER run tests in parallel (no pytest-xdist, no &, no background execution).

## How Test Sharding Works

`make test` runs the full suite across ~12 independent A100 GPU containers
on Modal.  Each container runs a subset of tests with exclusive GPU access.

**Why sharding?** The DOOM renderer tests call `compile_game(d=2048)` which
builds a ~15GB transformer model.  Multiple models cannot share a single GPU
without thrashing.  Sharding gives each compilation its own GPU.

**How it's configured** (in `modal_test.py`):

- `HEAVY_CLASSES` — test classes that call `compile_game()`.  These are
  auto-collected and split into sub-shards of `max_per_shard` tests each.
- `CATCH_ALL_SPLITS` — directories that partition the remaining tests.

**When you add new tests:**

- Tests in an existing heavy class (e.g. `TestCompiledStructure`,
  `TestGameGraph`) are picked up automatically.  No config changes needed.
- New test files anywhere under `tests/` are picked up by the catch-all
  shard automatically.
- Only update `HEAVY_CLASSES` when you create a **new class** that calls
  `compile_game()` or `compile_headless(d=2048)`.

**When using `FILE=`**, sharding is bypassed — the file runs in a single
container.  `-k` filters passed via `ARGS=` are applied to every shard.

## Performance Expectations

Full suite (`make test`): ~3 minutes wall time (tests execute in <90s per
shard; the rest is Modal container orchestration overhead).

Single file (`make test FILE=...`): depends on the file.
- Fast tests (ops, compile/forward, graph): 10-30s
- Compiled DOOM tests (test_game_graph.py): ~90s (includes ~15s compilation)
- Compiled wall selection (test_wall_selection.py): ~90s per 2-test shard

## Writing Tests That Use compile_game()

Tests calling `compile_game()` are expensive (~15s to compile, ~20s per
`step_frame` inference).  Follow these patterns:

1. **Use class-scoped fixtures** to share the compiled module across tests
   in the same class.  See `TestGameGraph` and `TestCompiledStructure` for
   examples.

2. **Keep compiled test classes in `HEAVY_CLASSES`** in `modal_test.py` so
   they get auto-split across containers.

3. **Parametrize over angles/configs** instead of looping inside a test —
   this lets the sharding system distribute individual cases to separate
   containers.
