"""Tests for :mod:`torchwright.debug.probe`.

The probe compares a compiled ``HeadlessTransformer`` against its
source graph's ``node.compute`` output using the per-sublayer snapshots
the compiler writes into ``residual_assignment``.  A healthy graph
must produce an empty divergence report.
"""

import numpy as np
import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.doom.game_graph import build_game_graph
from torchwright.reference_renderer.scenes import box_room
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


@pytest.fixture
def tiny_config():
    """Smallest healthy game-graph config the existing suite uses.

    16×12 frame with ``fov=8`` and no texture atlas — this is the
    config where the compiled renderer and the reference renderer
    agree exactly in the current test suite, so the probe has to
    report no divergence for it to be trustworthy.
    """
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=generate_trig_table(),
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


def _build_seed_input_row(n_inputs: int) -> torch.Tensor:
    """Step-0 input row for the DOOM game graph.

    Matches the layout that ``torchwright.doom.compile`` builds for a
    first-frame call with no player input: all zeros with
    ``cur_col=cur_patch=seed_x=seed_y=seed_angle=0``, no button
    presses.  That's the input the probe will evaluate against.
    """
    return torch.zeros(1, n_inputs)


def test_reference_eval_matches_direct_compute_tiny():
    """Build a tiny graph (small enough that unmemoised direct compute
    is fast) and check that ``reference_eval`` returns exactly the same
    top-level value as ``output_node.compute``.  Also checks that the
    class-level ``compute`` monkey-patches are restored cleanly: a
    second unmemoised compute after reference_eval must still produce
    the identical tensor.
    """
    from torchwright.graph.linear import Linear
    from torchwright.graph.misc import InputNode
    from torchwright.ops.arithmetic_ops import add, add_const, multiply_const

    x = InputNode("x", 1)
    y = InputNode("y", 1)
    # Graph: output = 2*x + (y + 3)  -- ~4 Linear nodes, easy to reason about.
    scaled = multiply_const(x, 2.0)
    shifted = add_const(y, 3.0)
    output = add(scaled, shifted)

    input_values = {
        "x": torch.tensor([[5.0], [0.5], [-1.0]]),
        "y": torch.tensor([[1.0], [2.0], [7.0]]),
    }
    n_pos = 3
    expected = torch.tensor([[14.0], [6.0], [8.0]])

    cache = reference_eval(output, input_values, n_pos)

    assert output in cache
    assert torch.allclose(cache[output], expected, atol=1e-6)
    # The un-memoised direct call must still work — i.e. the class-level
    # monkey-patches were restored.
    direct = output.compute(n_pos, input_values)
    assert torch.allclose(direct, expected, atol=1e-6)
    # Every intermediate is also cached.
    assert x in cache and y in cache
    assert torch.allclose(cache[x], input_values["x"])
    assert torch.allclose(cache[y], input_values["y"])


def test_probe_clean_on_untextured_box_room(tiny_config):
    """Untextured box_room at 16×12 — the config used by the existing
    ``test_compiled_no_input`` test — is known good.  The probe must
    find zero divergent nodes.
    """
    segs = box_room()
    output_node, pos_encoding = build_game_graph(
        segs, tiny_config, max_coord=10.0, move_speed=0.3, turn_speed=4,
    )
    input_values = {
        "cur_col_idx": torch.tensor([[0.0]]),
        "cur_patch_idx_in_col": torch.tensor([[0.0]]),
        "input_backward": torch.tensor([[0.0]]),
        "input_forward": torch.tensor([[0.0]]),
        "input_strafe_left": torch.tensor([[0.0]]),
        "input_strafe_right": torch.tensor([[0.0]]),
        "input_turn_left": torch.tensor([[0.0]]),
        "input_turn_right": torch.tensor([[0.0]]),
        "seed_angle": torch.tensor([[0.0]]),
        "seed_x": torch.tensor([[0.0]]),
        "seed_y": torch.tensor([[0.0]]),
    }

    report = probe_graph(
        output_node, pos_encoding, input_values, n_pos=1,
        d=1024, d_head=16, verbose=False, atol=1e-2,
    )

    assert report.nodes_checked, (
        "probe checked zero nodes — residual snapshots likely not populated"
    )
    assert report.first_divergent is None, (
        f"probe reported divergence on known-good graph:\n{report.format_short()}"
    )
