"""Probe SORT stage divergence at angle=210.

Compiles DOOM at d=2048 and runs a single frame at (0, 0, 210).
With ``TW_DEBUG_SORT=1``, step_frame prints sort_done + wall_index
at every position so we can see exactly where the SORT attention
resolves the "wrong" way.

Usage:
    make modal-run MODULE=scripts.probe_sort_divergence
"""

import os

os.environ["TW_DEBUG_SORT"] = "1"

from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.doom.trace import FrameTrace
from torchwright.reference_renderer.render import Segment
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig

TRIG = generate_trig_table()


def _box_room_config():
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room_segments(half=5.0):
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


def main():
    config = _box_room_config()
    textures = default_texture_atlas()
    segs = _box_room_segments()
    subset = build_scene_subset(segs, textures)

    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )

    state = GameState(x=0.0, y=0.0, angle=210, move_speed=0.3, turn_speed=4)
    inp = PlayerInput()
    trace = FrameTrace()
    try:
        step_frame(module, state, inp, subset, config, textures=textures, trace=trace)
    except IndexError as e:
        print(f"\n[probe] IndexError: {e}")
    print("\n[probe] SORT step count:", len(trace.sort_steps))
    for s in trace.sort_steps:
        print(
            f"  sort_step {s.position_index}: wall={s.selected_wall_index}  "
            f"vis_lo={s.vis_lo:.2f}  vis_hi={s.vis_hi:.2f}  sort_done={s.sort_done}"
        )


if __name__ == "__main__":
    main()
