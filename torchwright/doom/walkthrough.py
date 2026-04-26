"""Generate a DOOM walkthrough as an animated GIF.

Uses a wall-following algorithm: walk forward until close to a wall,
turn right 90 degrees, repeat.  By default the game logic and rendering
run inside a compiled transformer.

Usage:
    python -m torchwright.doom.walkthrough [output.gif] [--scene box|multi|e1m1] ...
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
            state.x,
            state.y,
            state.angle,
            self.segments,
            self.trig_table,
        )

        stuck = False
        if self._prev_x is not None:
            if (
                abs(state.x - self._prev_x) < 0.01
                and self._prev_y is not None
                and abs(state.y - self._prev_y) < 0.01
            ):
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
    still: bool = False,
) -> List[np.ndarray]:
    """Render a walkthrough sequence, returning a list of uint8 RGB frames.

    Args:
        frame_fn: Callable(state, inputs) -> (frame, new_state).
        still: If True, send empty PlayerInput each frame (no wall-following).
    """
    state = GameState(x=start_x, y=start_y, angle=start_angle)
    controller = (
        None
        if still
        else WalkthroughController(
            segments,
            config.trig_table,
            wall_threshold=wall_threshold,
        )
    )

    frames: List[np.ndarray] = []
    frame_times: List[float] = []
    t_start = time.perf_counter()
    for i in range(total_frames):
        inputs = PlayerInput() if controller is None else controller.get_input(state)

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
    print(
        f"  {total_frames} frames in {total_time:.1f}s "
        f"(avg {avg_ms:.0f}ms, {1000/avg_ms:.1f} fps)"
    )

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
            img = img.resize((w * scale, h * scale), Image.Resampling.NEAREST)
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
        "output",
        nargs="?",
        default="walkthrough.gif",
        help="Output GIF path",
    )
    parser.add_argument("--scene", choices=["box", "multi", "e1m1"], default="box")
    parser.add_argument(
        "--mode",
        choices=["transformer", "reference"],
        default="transformer",
        help="transformer: compiled transformer (default). "
        "reference: pure Python implementation.",
    )
    parser.add_argument(
        "--wad",
        type=str,
        default="doom1.wad",
        help="Path to doom1.wad for DOOM textures",
    )
    parser.add_argument("--tex-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--fov", type=int, default=32)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Nearest-neighbor upscale factor for output",
    )
    parser.add_argument(
        "--wall-threshold",
        type=float,
        default=1.5,
        help="Distance to wall that triggers a turn",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Render chunk height (pixels per render token).",
    )
    parser.add_argument(
        "--d", type=int, default=2048, help="Residual stream width (d_model)."
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

    subset = None
    still = False
    if args.scene == "box":
        segments, textures = box_room_textured(
            wad_path=args.wad,
            tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = 0.0, 0.0, 0
        max_coord = 10.0
    elif args.scene == "multi":
        segments, textures = multi_room_textured(
            wad_path=args.wad,
            tex_size=args.tex_size,
        )
        start_x, start_y, start_angle = -8.0, 0.0, 0
        max_coord = 15.0
    else:  # e1m1
        from torchwright.doom.map_subset import load_map_subset
        from torchwright.doom.wad import WADReader

        # Read the player-1 start (thing type 1) from the WAD's THINGS
        # lump.  The angle is in degrees (0=east, 90=north); convert to
        # the renderer's 0-255 scale.
        md = WADReader(args.wad).get_map("E1M1")
        spawn = next(t for t in md.things if t.type == 1)
        spawn_x = float(spawn.x)
        spawn_y = float(spawn.y)
        start_angle = round(spawn.angle / 360 * 256) % 256
        print(
            f"E1M1 player-1 spawn: pos=({spawn_x}, {spawn_y}) "
            f"angle_deg={spawn.angle} → renderer_angle={start_angle}"
        )

        # max_walls > 4 saturates the transformer's ±40 CROSS_A/DOT_A
        # clamp on far walls, but the reference renderer ray-casts in
        # world coords with no such limit; pick a count that gives the
        # spawn alcove + the room and corridor visible from it.
        subset = load_map_subset(
            wad_path=args.wad,
            map_name="E1M1",
            px=spawn_x,
            py=spawn_y,
            max_walls=32,
            tex_size=args.tex_size,
        )
        segments = subset.segments
        textures = subset.textures
        start_x, start_y = spawn_x, spawn_y
        max_coord = 100.0
        still = True

    if args.mode == "transformer":
        from torchwright.doom.compile import compile_game, step_frame
        from torchwright.doom.map_subset import build_scene_subset

        print(f"Compiling game graph (walls-as-tokens, {len(segments)} walls)...")
        module = compile_game(
            config,
            textures,
            max_walls=max(8, len(segments)),
            max_coord=max_coord,
            d=args.d,
            chunk_size=args.chunk_size,
        )
        if subset is None:
            subset = build_scene_subset(segments, textures)

        def frame_fn(state, inputs):
            return step_frame(module, state, inputs, subset, config, textures=textures)

    else:

        def frame_fn(state, inputs):
            new_state = update_state(state, inputs, segments, trig_table)
            frame = render_frame(
                new_state.x,
                new_state.y,
                new_state.angle,
                segments,
                config,
                textures=textures,
            )
            return frame, new_state

    print(
        f"Generating {args.frames} frames at {args.width}x{args.height} "
        f"({args.mode})..."
    )
    frames = generate_walkthrough(
        segments,
        config,
        frame_fn,
        start_x,
        start_y,
        start_angle,
        total_frames=args.frames,
        wall_threshold=args.wall_threshold,
        still=still,
    )

    print(f"Saving {args.output} (scale={args.scale}x, fps={args.fps})...")
    save_gif(frames, args.output, fps=args.fps, scale=args.scale)
    print(f"Done! {args.output}")


if __name__ == "__main__":
    main()
