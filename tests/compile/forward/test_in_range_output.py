"""Test that textured column fill with many bands compiles correctly.

When _textured_column_fill uses 16+ bands, the chained broadcast_select
calls cause big_offset to accumulate, producing values ~1000 instead of
the expected texture color (~0.5) in the wall region.
"""

import torch

from torchwright.compiler.export import compile_headless
from torchwright.doom.renderer import _textured_column_fill
from torchwright.graph.misc import LiteralValue
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


def test_textured_column_fill_16_bands():
    """16-band textured column fill should produce values near the texture color."""
    pos_encoding = create_pos_encoding()
    wall_top = create_input("wt", 1)
    wall_bottom = create_input("wb", 1)
    wall_height = create_input("wh", 1)

    config = RenderConfig(
        screen_width=8, screen_height=12, fov_columns=4,
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )

    # 16 texture rows, all the same red color
    tex_colors = LiteralValue(
        torch.tensor([0.9, 0.1, 0.1] * 16), name="tc",
    )

    output = _textured_column_fill(
        wall_top, wall_bottom, wall_height, tex_colors, 16, config,
    )

    module = compile_headless(
        output, pos_encoding, d=2048, d_head=16, max_layers=200, verbose=False,
    )

    # Wall from row 2 to row 10 (height=8)
    # HeadlessTransformerModule expects inputs in alphabetical name order:
    # wb=10, wh=8, wt=2
    with torch.no_grad():
        result = module(torch.tensor([[10.0, 8.0, 2.0]])).numpy().reshape(12, 3)

    # Wall rows should show the texture color, not big_offset garbage
    for row in range(2, 10):
        actual_r = result[row, 0]
        assert abs(actual_r - 0.9) < 1.0, (
            f"Row {row}: expected R~0.9, got {actual_r:.1f}"
        )
