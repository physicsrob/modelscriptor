"""Run pytest on Modal with GPU access.

Usage (via Makefile):
    make test
    make test FILE=tests/ops/test_arithmetic_ops.py
    make test ARGS="-k test_foo"

Direct usage:
    modal run modal_test.py
    modal run modal_test.py --file tests/graph/test_embedding.py
    modal run modal_test.py --args "-k test_foo --device cpu"
"""

import subprocess
import sys

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_file("doom1.wad", "/root/doom1.wad")
    .add_local_python_source("torchwright", "examples", "tests")
)

app = modal.App("torchwright-test", image=image)


@app.function(gpu="A100", cpu=8, timeout=600)
def run_pytest(file: str = "tests", extra_args: str = "") -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        file,
        "-v",
        "--tb=short",
        "--no-header",
        "--durations=0",
        "-n",
        "4",
    ]
    if extra_args:
        cmd.extend(extra_args.split())

    result = subprocess.run(cmd)
    return result.returncode


@app.local_entrypoint()
def main(file: str = "tests", args: str = ""):
    rc = run_pytest.remote(file=file, extra_args=args)
    sys.exit(rc)
