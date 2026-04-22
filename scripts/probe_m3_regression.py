"""Diagnose the Phase A M3 regression at scene (px=3, py=2, angle=20°).

Runs step_frame at the failing scene and inspects the full per-token
trajectory (requires compile.py's ``if True`` probe patch that records
length=0 render steps too).  Reports what the compiled state machine
did vs what the reference renderer expects.

Runs on GPU via ``make modal-run MODULE=scripts.probe_m3_regression``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.doom.trace import FrameTrace
from torchwright.reference_renderer.render import (
    project_wall,
    render_frame,
    render_wall_column,
)
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_TRIG = generate_trig_table()


def _config() -> RenderConfig:
    return RenderConfig(
        screen_width=64,
        screen_height=80,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room(half: float = 5.0) -> List[Segment]:
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


def _reference_column_distances(
    px: float, py: float, angle: int, segs: List[Segment], config: RenderConfig
) -> List[Tuple[int, List[Tuple[int, float, int, int]]]]:
    """For each column, list (wall_index, distance, vis_lo, vis_hi) in
    front-to-back order for walls visible at that column."""
    W = config.screen_width
    plans = []
    projections = []
    for wall_i, seg in enumerate(segs):
        proj = project_wall(px, py, angle, seg, config)
        projections.append((wall_i, seg, proj))
    for col in range(W):
        here = []
        for wall_i, seg, proj in projections:
            if proj is None:
                continue
            if proj.vis_lo <= col <= proj.vis_hi:
                rc = render_wall_column(col, proj, px, py, angle, config)
                if rc is None:
                    continue
                here.append((wall_i, rc.distance, proj.vis_lo, proj.vis_hi))
        here.sort(key=lambda t: t[1])
        plans.append((col, here))
    return plans


def main() -> int:
    PX, PY, ANGLE = 3.0, 2.0, 20
    print(f"=== M3 regression probe: scene (px={PX}, py={PY}, angle={ANGLE}) ===")

    config = _config()
    segs = _box_room()
    textures = default_texture_atlas()
    subset = build_scene_subset(segs, textures)

    print(f"\nCompiling (d=2048, d_head=32)...")
    module = compile_game(
        config, textures, max_walls=8, d=2048, d_head=32, verbose=False
    )

    print(f"\nRunning step_frame with full trace...")
    state = GameState(x=PX, y=PY, angle=ANGLE)
    trace = FrameTrace()
    frame, _ = step_frame(
        module, state, PlayerInput(), subset, config, textures=textures, trace=trace
    )
    ref = render_frame(PX, PY, ANGLE, segs, config, textures=textures)

    # Quick pixel stats
    diff = np.abs(frame - ref).max(axis=-1)
    matched = (diff <= 0.05).mean()
    print(f"\n=== FRAME COMPARISON ===")
    print(f"  matched_fraction = {matched:.4f}")
    print(f"  unmatched pixels = {int((diff > 0.05).sum())} / {diff.size}")

    # Reference column distances — what should be at each col
    ref_cols = _reference_column_distances(PX, PY, ANGLE, segs, config)
    print(f"\n=== REFERENCE COLUMN PLAN ===")
    for col, walls in ref_cols[:6]:
        print(f"  col {col:2d}: {[(w, f'{d:.2f}', lo, hi) for w, d, lo, hi in walls]}")
    print("  ...")
    for col, walls in ref_cols[-3:]:
        print(f"  col {col:2d}: {[(w, f'{d:.2f}', lo, hi) for w, d, lo, hi in walls]}")

    # Sort trajectory
    print(f"\n=== COMPILED SORT TRAJECTORY ===")
    for i, s in enumerate(trace.sort_steps):
        print(
            f"  sort[{i}] wall={s.selected_wall_index} "
            f"vis_lo={s.vis_lo:.2f} vis_hi={s.vis_hi:.2f} "
            f"sort_done={s.sort_done}"
        )
    print(f"  n_renderable = {trace.n_renderable}")

    # Render trajectory — with compile.py probe patch this includes length=0 rows.
    print(f"\n=== COMPILED RENDER TRAJECTORY (first 40) ===")
    rs = trace.render_steps
    print(f"  total render steps: {len(rs)}")
    for i, r in enumerate(rs[:40]):
        print(
            f"  render[{i:3d}] col={r.col:2d} wall={r.wall_index} "
            f"start={r.start:2d} length={r.length:2d} done={r.done}"
        )
    if len(rs) > 50:
        print(f"  ... {len(rs) - 50} rows omitted ...")
        for i, r in enumerate(rs[-10:], start=len(rs) - 10):
            print(
                f"  render[{i:3d}] col={r.col:2d} wall={r.wall_index} "
                f"start={r.start:2d} length={r.length:2d} done={r.done}"
            )

    # Count length=0 runs — indicate stuck state
    zero_runs = 0
    cur = 0
    max_run = 0
    for r in rs:
        if r.length == 0:
            cur += 1
            max_run = max(max_run, cur)
        else:
            if cur > 0:
                zero_runs += 1
            cur = 0
    print(f"\n  length=0 runs: {zero_runs}, max consecutive length=0: {max_run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
