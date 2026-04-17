"""Run scripts/dump_phase_e_allocator.py on a Modal A100.

Usage:
    modal run modal_dump_phase_e.py

Prints the three diagnostic blocks (I1 sanity, ownership, compiled-vs-oracle)
for the SORTED attention's read layer at scene (px=3, py=2, angle=20).
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_python_source("torchwright", "scripts")
)

app = modal.App("torchwright-dump-phase-e", image=image)


@app.function(gpu="a100-80gb", cpu=8, timeout=1800)
def run_dump(tf32: bool = True) -> str:
    import io
    from contextlib import redirect_stdout

    import torch

    if not tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("TF32 DISABLED — matmul and cuDNN forced to full fp32.")

    from scripts import dump_phase_e_allocator

    buf = io.StringIO()
    with redirect_stdout(buf):
        dump_phase_e_allocator.main()
    return buf.getvalue()


@app.local_entrypoint()
def main(tf32: bool = True):
    output = run_dump.remote(tf32=tf32)
    print(output)
