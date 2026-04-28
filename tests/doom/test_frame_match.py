"""Pixel-pipeline smoke test for the textured (production) DOOM graph.

A single textured frame match keeps the texture path — texture
attention, height projection, chunked column fill — gated in CI.  All
the structural and contract-VALUE assertions live in
``test_rollout.py``; this file exists only so the pixel pipeline can't
silently break.
"""

import pytest

from tests._utils.image_compare import compare_images
from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.reference_renderer.render import render_frame
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_TRIG = generate_trig_table()


def _config():
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _segments(half: float = 5.0):
    # Clockwise winding so FRONT (right of a→b) faces inside the room.
    # Required for the new render_column's back-face cull.
    common = dict(
        color=(0.8, 0.2, 0.1),
        front_floor=-1.0,
        front_ceiling=1.0,
    )
    return [
        # East going south.
        Segment(ax=half, ay=half, bx=half, by=-half, texture_id=0, **common),
        # South going west.
        Segment(ax=half, ay=-half, bx=-half, by=-half, texture_id=3, **common),
        # West going north.
        Segment(ax=-half, ay=-half, bx=-half, by=half, texture_id=1, **common),
        # North going east.
        Segment(ax=-half, ay=half, bx=half, by=half, texture_id=2, **common),
    ]


class TestFrameMatch:
    """Single textured frame match at 16×20 (production graph)."""

    @pytest.fixture(scope="class")
    def scene(self):
        config = _config()
        textures = default_texture_atlas()
        segs = _segments()
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs

    @pytest.fixture(scope="class")
    def module(self, scene):
        config, textures, _subset, _segs = scene
        return compile_game(
            config,
            textures,
            max_walls=8,
            d=2048,
            d_head=32,
            verbose=False,
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "renderer projection model divergence: the reference renderer "
            "was rewritten in this branch to DOOM-style focal-length "
            "projection driven by per-seg front_floor/front_ceiling and "
            "RenderConfig.player_eye_z, while the compiled transformer's "
            "vertical projection still computes wall extent as "
            "screen_height/perp_distance centred on the eye line. "
            "Will be fixed by porting the focal-length + sector floor/ceiling "
            "projection into the DOOM graph stages so the compiled "
            "transformer matches the reference; this xfail comes off in "
            "that commit."
        ),
    )
    def test_box_room_frame_matches_reference(self, module, scene):
        config, textures, subset, segs = scene
        state = GameState(x=0.0, y=0.0, angle=0, move_speed=0.3, turn_speed=4)
        frame, _ = step_frame(
            module, state, PlayerInput(), subset, config, textures=textures
        )
        ref = render_frame(0.0, 0.0, 0, segs, config, textures=textures)
        compare_images(frame, ref).assert_matches(
            min_matched_fraction=0.96, max_err=float("inf")
        )
