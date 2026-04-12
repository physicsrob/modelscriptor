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
