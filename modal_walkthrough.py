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
    .add_local_file("E8.8.1024.txt", "/root/E8.8.1024.txt")
    .add_local_file("doom1.wad", "/root/doom1.wad")
    .add_local_python_source("torchwright", "examples", "tests")
)

app = modal.App("torchwright-walkthrough", image=image)


def _scene_data(scene, tex_size):
    from torchwright.reference_renderer.scenes import (
        box_room_textured,
        multi_room_textured,
    )

    if scene == "box":
        segments, textures = box_room_textured(
            wad_path="doom1.wad", tex_size=tex_size,
        )
        return segments, textures, 0.0, 0.0, 0, 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path="doom1.wad", tex_size=tex_size,
        )
        return segments, textures, -8.0, 0.0, 0, 15.0


def _config(width, height):
    from torchwright.reference_renderer.trig import generate_trig_table
    from torchwright.reference_renderer.types import RenderConfig

    return RenderConfig(
        screen_width=width,
        screen_height=height,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


@app.function(gpu="A100", cpu=8, timeout=1800)
def generate_transformer(
    scene: str = "box",
    width: int = 120,
    height: int = 100,
    rows_per_patch: int = 10,
    tex_size: int = 64,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
    d: int = 2048,
    d_head: int = 32,
) -> bytes:
    from torchwright.doom.compile import compile_game, step_frame, segments_to_walls
    from torchwright.doom.walkthrough import generate_walkthrough, save_gif

    config = _config(width, height)
    segments, textures, start_x, start_y, start_angle, max_coord = _scene_data(scene, tex_size)

    walls = segments_to_walls(segments)
    print(f"Compiling game graph (walls-as-tokens, {len(walls)} walls)...")
    module = compile_game(
        config, textures,
        max_walls=max(8, len(walls)),
        max_coord=max_coord,
        d=d, d_head=d_head,
        rows_per_patch=rows_per_patch,
        device="cuda",
    )

    def frame_fn(state, inputs):
        return step_frame(module, state, inputs, walls, config)

    print(f"Generating {frames} transformer frames at {width}x{height}...")
    frame_list = generate_walkthrough(
        segments, config, frame_fn, start_x, start_y, start_angle,
        total_frames=frames, wall_threshold=1.5,
    )

    gif_path = "/tmp/walkthrough.gif"
    save_gif(frame_list, gif_path, fps=fps, scale=scale)

    with open(gif_path, "rb") as f:
        return f.read()


@app.function(cpu=4, timeout=1800)
def generate_reference(
    scene: str = "box",
    width: int = 120,
    height: int = 100,
    tex_size: int = 64,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
) -> bytes:
    from torchwright.doom.game import update_state
    from torchwright.doom.walkthrough import generate_walkthrough, save_gif
    from torchwright.reference_renderer.render import render_frame

    config = _config(width, height)
    segments, textures, start_x, start_y, start_angle, _ = _scene_data(scene, tex_size)

    def frame_fn(state, inputs):
        new_state = update_state(
            state, inputs, segments, config.trig_table,
        )
        frame = render_frame(
            new_state.x, new_state.y, new_state.angle, segments, config,
            textures=textures,
        )
        return frame, new_state

    print(f"Generating {frames} reference frames at {width}x{height}...")
    frame_list = generate_walkthrough(
        segments, config, frame_fn, start_x, start_y, start_angle,
        total_frames=frames, wall_threshold=1.5,
    )

    gif_path = "/tmp/reference.gif"
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
    d_head: int = 32,
):
    # Launch both in parallel
    transformer_call = generate_transformer.spawn(
        scene=scene, width=width, height=height,
        rows_per_patch=rows_per_patch, tex_size=tex_size,
        frames=frames, fps=fps, scale=scale, d=d, d_head=d_head,
    )
    reference_call = generate_reference.spawn(
        scene=scene, width=width, height=height,
        tex_size=tex_size, frames=frames, fps=fps, scale=scale,
    )

    transformer_bytes = transformer_call.get()
    reference_bytes = reference_call.get()

    with open("walkthrough.gif", "wb") as f:
        f.write(transformer_bytes)
    print(f"Saved walkthrough.gif ({len(transformer_bytes)} bytes)")

    with open("reference.gif", "wb") as f:
        f.write(reference_bytes)
    print(f"Saved reference.gif ({len(reference_bytes)} bytes)")
