"""Run pytest on Modal with GPU access.

In sharded mode (default for full suite), tests are distributed across
independent Modal containers, each with its own A100 GPU.

Heavy test classes (those calling compile_game) are auto-collected and
split into sub-shards.  New tests are picked up automatically.

Usage (via Makefile):
    make test                    # auto-sharded across GPUs
    make test FILE=tests/foo.py  # single container, no sharding
    make test ARGS="-k test_foo" # filter applied to all shards
"""

import shlex
import subprocess
import sys
import time

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_file("doom1.wad", "/root/doom1.wad")
    .add_local_python_source("torchwright", "examples", "tests")
)

app = modal.App("torchwright-test", image=image)

# ── Shard configuration ──────────────────────────────────────────
# Heavy test classes that call compile_game() are auto-collected and
# split into sub-shards.  Only update this config when you create a
# NEW heavy class — adding tests to existing classes is automatic.

HEAVY_CLASSES = [
    {
        "file": "tests/doom/test_wall_selection.py",
        "class": "TestCompiledStructure",
        "max_per_shard": 2,
    },
    {
        "file": "tests/doom/test_game_graph.py",
        "class": "TestGameGraph",
        "max_per_shard": 3,
    },
]

# The catch-all is split across these directories for balance.
CATCH_ALL_SPLITS = [
    "tests/compile",
    "tests/doom tests/debug",
    "tests --ignore=tests/compile --ignore=tests/doom --ignore=tests/debug",
]


# ── Remote function ───────────────────────────────────────────────

@app.function(gpu="A100", cpu=8, memory=32768, timeout=1800)
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


# ── Local helpers ─────────────────────────────────────────────────

def _collect_local(file_path: str, class_name: str) -> list[str]:
    """Collect test node IDs locally (fast, no GPU needed)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", file_path,
         "-k", class_name, "--collect-only", "-q"],
        capture_output=True, text=True,
    )
    return [
        line for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    ]


def _build_shards() -> list[str]:
    shards: list[str] = []
    all_heavy_ids: list[str] = []

    for spec in HEAVY_CLASSES:
        tests = _collect_local(spec["file"], spec["class"])
        max_n = spec["max_per_shard"]
        n_shards = -(-len(tests) // max_n)
        print(f"  {spec['class']}: {len(tests)} tests → {n_shards} sub-shards")
        all_heavy_ids.extend(tests)

        for i in range(0, len(tests), max_n):
            chunk = tests[i : i + max_n]
            shards.append(" ".join(chunk))

    # Catch-all: ALL tests minus the heavy ones (via --deselect).
    # Split across directory groups for balance.
    deselects = " ".join(f"--deselect={t}" for t in all_heavy_ids)
    for path in CATCH_ALL_SPLITS:
        shards.append(f"{path} {deselects}")

    return shards


# ── Entrypoint ────────────────────────────────────────────────────

@app.local_entrypoint()
def main(file: str = "tests", args: str = ""):
    if file != "tests":
        rc = run_pytest.remote(pytest_args=file, extra_args=args)
        sys.exit(rc)

    # Build shards (collection happens locally — instant, no GPU)
    print("Building shards...")
    shards = _build_shards()
    print(f"\nRunning {len(shards)} shards in parallel:")
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
