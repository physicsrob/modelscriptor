"""Compile the DOOM game graph and generate a walkthrough GIF on Modal.

Usage (via Makefile):
    make walkthrough
    make walkthrough ARGS="--frames 20 --scene multi"

Direct usage:
    modal run modal_walkthrough.py
    modal run modal_walkthrough.py --frames 20 --scene multi
"""

import sys

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["dev"], extra_options="--no-install-project")
    .uv_pip_install(
        "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
    )
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_file("doom1.wad", "/root/doom1.wad")
    .add_local_python_source("torchwright", "examples", "tests")
)

app = modal.App("torchwright-walkthrough", image=image)


@app.function(gpu="A100", cpu=8, timeout=1800)
def generate_walkthrough(
    scene: str = "box",
    width: int = 120,
    height: int = 100,
    rows_per_patch: int = 10,
    tex_size: int = 64,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
    d: int = 2048,
) -> bytes:
    import glob
    import os

    cuda12_dirs = glob.glob("/.uv/.venv/lib/python3.12/site-packages/nvidia/*/lib")
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda12_dirs)

    from torchwright.compiler.export import compile_headless_to_onnx
    from torchwright.compiler.onnx_load import OnnxHeadlessModule
    from torchwright.doom.compile import step_frame_compiled
    from torchwright.doom.game_graph import build_game_graph
    from torchwright.doom.walkthrough import generate_walkthrough, save_gif
    from torchwright.reference_renderer.scenes import (
        box_room_textured,
        multi_room_textured,
    )
    from torchwright.reference_renderer.trig import generate_trig_table
    from torchwright.reference_renderer.types import RenderConfig

    trig_table = generate_trig_table()
    config = RenderConfig(
        screen_width=width,
        screen_height=height,
        fov_columns=32,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    if scene == "box":
        segments, textures = box_room_textured(
            wad_path="doom1.wad", tex_size=tex_size,
        )
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path="doom1.wad", tex_size=tex_size,
        )
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0

    # Step 1: Compile the ONNX model
    print(f"Building game graph (d={d})...")
    output_node, pos_encoding = build_game_graph(
        segments, config, max_coord,
        move_speed=0.3, turn_speed=4,
        textures=textures,
        rows_per_patch=rows_per_patch,
    )

    max_seq_len = width * (height // rows_per_patch)
    onnx_path = "/tmp/doom_game.onnx"

    compile_headless_to_onnx(
        output_node=output_node,
        pos_encoding=pos_encoding,
        output_path=onnx_path,
        d=d,
        d_head=16,
        max_seq_len=max_seq_len,
        max_layers=200,
        verbose=True,
        extra_metadata={"rows_per_patch": rows_per_patch},
    )

    # Step 2: Generate the walkthrough
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"Loading {onnx_path} (providers={providers})...")
    module = OnnxHeadlessModule(onnx_path, providers=providers)

    def frame_fn(state, inputs):
        return step_frame_compiled(
            module, state, inputs, config, cache_lookback=10,
        )

    print(f"Generating {frames} frames at {width}x{height}...")
    frame_list = generate_walkthrough(
        segments, config, frame_fn, start_x, start_y, start_angle,
        total_frames=frames, wall_threshold=1.5,
    )

    gif_path = "/tmp/walkthrough.gif"
    print(f"Saving GIF (scale={scale}x, fps={fps})...")
    save_gif(frame_list, gif_path, fps=fps, scale=scale)

    with open(gif_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    scene: str = "box",
    width: int = 120,
    height: int = 100,
    rows_per_patch: int = 10,
    tex_size: int = 64,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
    d: int = 2048,
    output: str = "walkthrough.gif",
):
    gif_bytes = generate_walkthrough.remote(
        scene=scene,
        width=width,
        height=height,
        rows_per_patch=rows_per_patch,
        tex_size=tex_size,
        frames=frames,
        fps=fps,
        scale=scale,
        d=d,
    )

    with open(output, "wb") as f:
        f.write(gif_bytes)
    print(f"Saved {output} ({len(gif_bytes)} bytes)")
