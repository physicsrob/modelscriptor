"""End-to-end tests: compile the renderer graph and compare against reference."""

import numpy as np
import pytest
import torch

from torchwright.doom.compile import compile_renderer, render_frame_compiled
from torchwright.reference_renderer import (
    RenderConfig,
    Segment,
    generate_trig_table,
    render_frame,
)


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def config(trig_table):
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=trig_table,
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


@pytest.fixture
def single_wall():
    """A single wall directly ahead at x=5."""
    return [Segment(ax=5.0, ay=-10.0, bx=5.0, by=10.0, color=(1.0, 0.0, 0.0))]


@pytest.fixture
def compiled_single_wall(single_wall, config):
    """Compile the renderer for a single wall — cached across tests."""
    return compile_renderer(
        single_wall, config, max_coord=15.0, d=512, d_head=16, verbose=False,
    )


def test_compiled_single_wall(compiled_single_wall, single_wall, config):
    """Compiled renderer matches reference for a single wall, 4 viewpoints."""
    module = compiled_single_wall

    for angle in [0, 32, 224]:
        compiled_frame = render_frame_compiled(module, 0.0, 0.0, angle, config)
        ref_frame = render_frame(0.0, 0.0, angle, single_wall, config)

        np.testing.assert_allclose(
            compiled_frame, ref_frame, atol=0.15,
            err_msg=f"Mismatch at angle={angle}",
        )


def test_compiled_two_walls(config):
    """Two walls at different distances — nearer one occludes."""
    segments = [
        Segment(ax=3.0, ay=-10.0, bx=3.0, by=10.0, color=(1.0, 0.0, 0.0)),
        Segment(ax=8.0, ay=-10.0, bx=8.0, by=10.0, color=(0.0, 1.0, 0.0)),
    ]
    module = compile_renderer(
        segments, config, max_coord=15.0, d=512, d_head=16, verbose=False,
    )

    compiled_frame = render_frame_compiled(module, 0.0, 0.0, 0, config)
    ref_frame = render_frame(0.0, 0.0, 0, segments, config)

    # Center column should show the nearer (red) wall
    center_col = config.screen_width // 2
    center_row = config.screen_height // 2
    np.testing.assert_allclose(
        compiled_frame[center_row, center_col],
        ref_frame[center_row, center_col],
        atol=0.15,
        err_msg="Center pixel mismatch (should be nearer red wall)",
    )


def test_compiled_box_room(config):
    """Box room from center, looking in all 4 directions."""
    from torchwright.reference_renderer.scenes import box_room
    segments = box_room()

    module = compile_renderer(
        segments, config, max_coord=10.0, d=512, d_head=16, verbose=False,
    )

    for angle in [0, 64, 128, 192]:
        compiled_frame = render_frame_compiled(module, 0.0, 0.0, angle, config)
        ref_frame = render_frame(0.0, 0.0, angle, segments, config)

        # Check center pixel matches
        center_col = config.screen_width // 2
        center_row = config.screen_height // 2
        np.testing.assert_allclose(
            compiled_frame[center_row, center_col],
            ref_frame[center_row, center_col],
            atol=0.15,
            err_msg=f"Center pixel mismatch at angle={angle}",
        )
