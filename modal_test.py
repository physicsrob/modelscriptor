"""Run pytest on Modal with GPU access.

In sharded mode (default for full suite), tests are distributed across
independent Modal containers, each with its own A100 GPU.

Usage (via Makefile):
    make test                    # sharded across GPUs
    make test FILE=tests/foo.py  # single container, no sharding
    make test ARGS="-k test_foo" # filter applied to all shards
"""

import shlex
import subprocess
import sys
import time

import modal

from modal_image import IMAGE

app = modal.App("torchwright-test", image=IMAGE)

# ── Shard definitions ─────────────────────────────────────────────
# Simple file-level sharding.  Heavy compiled-test files get their
# own container; everything else is batched together.
# New test files are caught by the catch-all shard automatically.

_HEAVY_FILES = [
    "tests/doom/test_game_graph.py",
    "tests/doom/test_wall_selection.py",
    "tests/doom/test_bsp_rank_integration.py",
    "tests/doom/test_angle_192_column_drift.py",
    "tests/doom/test_tex_col.py",
    "tests/doom/test_render_graph_precision.py",
    "tests/debug/test_probe_phase_e_trace.py",
]

_MEDIUM_FILES = [
    "tests/doom/test_combined.py",
    "tests/debug/test_probe.py",
    "tests/doom/test_parametric_render.py",
    "tests/doom/test_parametric_intersection.py",
]

SHARDS = [
    *_HEAVY_FILES,
    " ".join(_MEDIUM_FILES),
    "tests " + " ".join(f"--ignore={f}" for f in _HEAVY_FILES + _MEDIUM_FILES),
]


# ── Remote function ───────────────────────────────────────────────


@app.function(gpu="a100-80gb", cpu=8, memory=32768, timeout=1800)
def run_pytest(pytest_args: str, extra_args: str = "") -> int:
    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *shlex.split(pytest_args),
        "-v",
        "--tb=short",
        "--no-header",
        "--durations=0",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f"[shard] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"\n[shard] finished in {elapsed:.0f}s (exit {result.returncode})")
    return result.returncode


# ── Entrypoint ────────────────────────────────────────────────────


@app.local_entrypoint()
def main(file: str = "tests", args: str = ""):
    if file != "tests":
        rc = run_pytest.remote(pytest_args=file, extra_args=args)
        sys.exit(rc)

    shards = SHARDS
    print(f"Running {len(shards)} shards in parallel:")
    for i, s in enumerate(shards):
        label = s[:90] + "…" if len(s) > 90 else s
        print(f"  shard {i}: {label}")

    t0 = time.time()
    results = list(run_pytest.map(shards, kwargs={"extra_args": args}))
    elapsed = time.time() - t0

    failed = sum(1 for rc in results if rc != 0)
    print(f"\n{'=' * 60}")
    print(f"All shards finished in {elapsed:.0f}s")
    for i, rc in enumerate(results):
        status = "PASS" if rc == 0 else "FAIL"
        print(f"  shard {i}: {status} (rc={rc})")
    if failed:
        print(f"{failed}/{len(results)} shards failed")
    print(f"{'=' * 60}")

    sys.exit(1 if failed else 0)
