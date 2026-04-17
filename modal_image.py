"""Shared Modal image used by modal_run / modal_test / modal_walkthrough."""

import modal

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_file("doom1.wad", "/root/doom1.wad")
    .add_local_python_source(
        "torchwright", "examples", "tests", "scripts", "modal_image"
    )
)
