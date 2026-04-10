"""Generate a DOOM walkthrough as an animated GIF.

Uses a wall-following algorithm: walk forward until close to a wall,
turn right 90 degrees, repeat.  By default the game logic and rendering
run inside a compiled transformer.

Usage:
    python -m torchwright.doom.walkthrough [output.gif] [--scene box|multi] ...
"""

import argparse
import time
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from torchwright.doom.game import GameState, update_state
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer.render import intersect_ray_segment, render_frame
from torchwright.reference_renderer.scenes import box_room_textured, multi_room_textured
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


# ---------------------------------------------------------------------------
# Wall distance sensing
# ---------------------------------------------------------------------------


def forward_wall_distance(
    x: float,
    y: float,
    angle: int,
    segments: List[Segment],
    trig_table: np.ndarray,
) -> float:
    """Cast a ray in the player's facing direction and return distance to nearest wall."""
    cos_a = float(trig_table[angle, 0])
    sin_a = float(trig_table[angle, 1])
    best_t = float("inf")
    for seg in segments:
        hit = intersect_ray_segment(x, y, cos_a, sin_a, seg)
        if hit is not None:
            t, _u = hit
            if t > 0 and t < best_t:
                best_t = t
    return best_t


# ---------------------------------------------------------------------------
# Wall-following controller
# ---------------------------------------------------------------------------


class _Phase(Enum):
    WALKING = auto()
    TURNING = auto()


class WalkthroughController:
    """Generates PlayerInput commands using a wall-following strategy."""

    def __init__(
        self,
        segments: List[Segment],
        trig_table: np.ndarray,
        wall_threshold: float = 1.5,
        turn_frames: int = 16,
    ):
        self.segments = segments
        self.trig_table = trig_table
        self.wall_threshold = wall_threshold
        self.turn_frames = turn_frames
        self._phase = _Phase.WALKING
        self._turn_counter = 0
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

    def get_input(self, state: GameState) -> PlayerInput:
        if self._phase is _Phase.TURNING:
            self._turn_counter += 1
            if self._turn_counter >= self.turn_frames:
                self._phase = _Phase.WALKING
                self._turn_counter = 0
                # Reset so first walking frame doesn't trigger stuck detection
                self._prev_x = None
                self._prev_y = None
            return PlayerInput(turn_right=True)

        # WALKING -- check if we should turn
        dist = forward_wall_distance(
            state.x, state.y, state.angle, self.segments, self.trig_table,
        )

        stuck = False
        if self._prev_x is not None:
            if abs(state.x - self._prev_x) < 0.01 and abs(state.y - self._prev_y) < 0.01:
                stuck = True

        self._prev_x = state.x
        self._prev_y = state.y

        if dist < self.wall_threshold or stuck:
            self._phase = _Phase.TURNING
            self._turn_counter = 1
            return PlayerInput(turn_right=True)

        return PlayerInput(forward=True)


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------


def generate_walkthrough(
    segments: List[Segment],
    config: RenderConfig,
    frame_fn: Callable[[GameState, PlayerInput], Tuple[np.ndarray, GameState]],
    start_x: float,
    start_y: float,
    start_angle: int,
    total_frames: int = 300,
    wall_threshold: float = 1.5,
) -> List[np.ndarray]:
    """Render a walkthrough sequence, returning a list of uint8 RGB frames.

    Args:
        frame_fn: Callable(state, inputs) -> (frame, new_state).
    """
    state = GameState(x=start_x, y=start_y, angle=start_angle)
    controller = WalkthroughController(
        segments, config.trig_table, wall_threshold=wall_threshold,
    )

    frames: List[np.ndarray] = []
    frame_times: List[float] = []
    t_start = time.perf_counter()
    for i in range(total_frames):
        inputs = controller.get_input(state)

        t0 = time.perf_counter()
        frame, state = frame_fn(state, inputs)
        frame_times.append(time.perf_counter() - t0)

        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frames.append(pixels)

        n = i + 1
        if n <= 5 or n % 10 == 0:
            avg_ms = sum(frame_times) / len(frame_times) * 1000
            elapsed = time.perf_counter() - t_start
            print(
                f"  frame {n}/{total_frames}  "
                f"{frame_times[-1]*1000:.0f}ms (avg {avg_ms:.0f}ms)  "
                f"elapsed {elapsed:.1f}s  "
                f"pos=({state.x:.1f}, {state.y:.1f}) angle={state.angle}"
            )

    total_time = time.perf_counter() - t_start
    avg_ms = sum(frame_times) / len(frame_times) * 1000
    print(f"  {total_frames} frames in {total_time:.1f}s "
          f"(avg {avg_ms:.0f}ms, {1000/avg_ms:.1f} fps)")

    return frames


# ---------------------------------------------------------------------------
# GIF output
# ---------------------------------------------------------------------------


def save_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    scale: int = 1,
) -> None:
    """Save a list of uint8 RGB frames as an animated GIF."""
    pil_frames: List[Image.Image] = []
    for f in frames:
        img = Image.fromarray(f, mode="RGB")
        if scale > 1:
            w, h = img.size
            img = img.resize((w * scale, h * scale), Image.NEAREST)
        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate a DOOM walkthrough GIF")
    parser.add_argument(
        "output", nargs="?", default="walkthrough.gif", help="Output GIF path",
    )
    parser.add_argument("--scene", choices=["box", "multi"], default="box")
    parser.add_argument(
        "--mode", choices=["transformer", "reference"], default="transformer",
        help="transformer: compiled transformer (default). "
             "reference: pure Python implementation.",
    )
    parser.add_argument("--wad", type=str, default="doom1.wad",
                        help="Path to doom1.wad for DOOM textures")
    parser.add_argument("--tex-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=int, default=4,
                        help="Nearest-neighbor upscale factor for output")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for transformer mode (default: cuda)")
    parser.add_argument("--d", type=int, default=None,
                        help="Transformer residual stream width "
                             "(default: auto from height)")
    parser.add_argument("--wall-threshold", type=float, default=1.5,
                        help="Distance to wall that triggers a turn")
    parser.add_argument(
        "--from-onnx", type=str, default=None,
        help="Path to a pre-saved game .onnx (from torchwright.doom.to_onnx). "
             "Skips the in-process compile and runs via onnxruntime. "
             "Scene/width/height/fov must match the config used at export.",
    )
    args = parser.parse_args()

    trig_table = generate_trig_table()
    config = RenderConfig(
        screen_width=args.width,
        screen_height=args.height,
        fov_columns=args.fov,
        trig_table=trig_table,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )

    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path=args.wad, tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    else:
        segments, textures = multi_room_textured(
            wad_path=args.wad, tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0

    if args.mode == "transformer":
        from torchwright.doom.compile import step_frame_compiled

        if args.from_onnx is not None:
            from torchwright.compiler.onnx_load import OnnxHeadlessModule

            print(f"Loading {args.from_onnx}...")
            module = OnnxHeadlessModule(args.from_onnx)
        else:
            from torchwright.doom.compile import compile_game

            # Auto-size d to fit per-column output (H*3 pixels) plus working
            # stream (~3500 used for H=80 with d=4096). Round up to next pow2.
            if args.d is None:
                needed = max(4096, args.height * 3 + 3500)
                d = 1024
                while d < needed:
                    d *= 2
            else:
                d = args.d

            print(f"Compiling game graph (d={d})...")
            module = compile_game(
                segments, config, max_coord,
                textures=textures,
                d=d, d_head=16,
                device=args.device,
            )

        def frame_fn(state, inputs):
            return step_frame_compiled(module, state, inputs, config)
    else:
        def frame_fn(state, inputs):
            new_state = update_state(state, inputs, segments, trig_table)
            frame = render_frame(
                new_state.x, new_state.y, new_state.angle, segments, config,
                textures=textures,
            )
            return frame, new_state

    print(f"Generating {args.frames} frames at {args.width}x{args.height} "
          f"({args.mode})...")
    frames = generate_walkthrough(
        segments, config, frame_fn, start_x, start_y, start_angle,
        total_frames=args.frames, wall_threshold=args.wall_threshold,
    )

    print(f"Saving {args.output} (scale={args.scale}x, fps={args.fps})...")
    save_gif(frames, args.output, fps=args.fps, scale=args.scale)
    print(f"Done! {args.output}")


if __name__ == "__main__":
    main()
