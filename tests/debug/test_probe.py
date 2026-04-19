"""Tests for :mod:`torchwright.debug.probe`.

The probe compares a compiled ``HeadlessTransformer`` against its
source graph's ``node.compute`` output using the per-sublayer snapshots
the compiler writes into ``residual_assignment``.  A healthy graph
must produce an empty divergence report.
"""

import numpy as np
import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import (
    probe_attention,
    probe_graph,
    probe_layer_diff,
    probe_residual,
    reference_eval,
)
from torchwright.graph import Attn
from torchwright.doom.game_graph import build_game_graph
from torchwright.ops.arithmetic_ops import add_const, multiply_const
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.reference_renderer.scenes import box_room
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig


@pytest.fixture
def tiny_config():
    """Smallest healthy game-graph config the existing suite uses.

    16×20 frame with ``fov=16`` — matches the v2 test fixture shape
    where the compiled renderer and the reference renderer agree, so
    the probe has to report no divergence for it to be trustworthy.
    """
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=16,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
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

    x = InputNode("x", 1, value_range=(-100.0, 100.0))
    y = InputNode("y", 1, value_range=(-100.0, 100.0))
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


def test_probe_residual_reads_intermediate_node():
    """Compile a tiny chain and confirm ``probe_residual`` reports the
    intermediate node's value at the layer that materialises it.

    Graph: y = 3*x, z = y + 1.  Compile overlaid on x, probe y.  Every
    layer that has y live must hold 3*x; empty per_layer would indicate
    the probe failed to surface the node.
    """
    pos = create_pos_encoding()
    x = create_input(2)
    y = multiply_const(x, 3.0)
    z = add_const(y, 1.0)
    module = compile_headless(
        pos,
        io={"x": (x, z)},
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[2.0, 4.0]])
    report = probe_residual(module, inp, y)

    assert report.layers, "probe_residual surfaced no layers for y"
    expected_y = torch.tensor([[6.0, 12.0]])
    for layer_i in report.layers:
        v = report.at(layer_i)
        assert v is not None
        assert torch.allclose(
            v, expected_y, atol=0.1
        ), f"layer {layer_i}: expected {expected_y.tolist()} got {v.tolist()}"
    # at_layer filter: restrict to the last layer that holds y.
    one = probe_residual(module, inp, y, at_layer=report.layers[-1])
    assert one.layers == [report.layers[-1]]


def test_probe_attention_captures_softmax_weights():
    """Build a tiny single-Attn graph, compile, and confirm
    ``probe_attention`` returns per-head softmax weights that sum to
    one, with at least one non-trivial top contributor at the queried
    row.  This exercises the monkey-patch + forward path end-to-end on
    a graph small enough that layer-hosting lookup is unambiguous.
    """
    pos = create_pos_encoding()
    x = create_input("x", 4)
    torch.manual_seed(0)
    attn = Attn(
        query_in=x,
        key_in=x,
        value_in=x,
        query_matrix=torch.randn(4, 4),
        key_matrix=torch.randn(4, 4),
        value_matrix=torch.randn(4, 4),
        output_matrix=torch.randn(4, 4),
    )
    module = compile_headless(
        pos,
        io={"x": (x, attn)},
        d=256,
        d_head=16,
        verbose=False,
    )

    inp = torch.randn(3, 4)
    report = probe_attention(module, inp, attn, query_pos=2)

    assert report.layer_index >= 0
    # Softmax sums to 1 per head (within fp tolerance).
    sums = report.weights.sum(dim=-1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-4
    ), f"weights don't sum to 1 per head: {sums.tolist()}"
    # Causal mask: query row 2 can attend to positions 0, 1, 2 only.
    # Positions 3..n_keys-1 (if any) must have negligible weight.
    if report.weights.shape[1] > 3:
        future_mass = report.weights[:, 3:].sum().item()
        assert (
            future_mass < 1e-3
        ), f"causal mask leak — future positions got {future_mass:.3g}"
    # top() returns a ranked list.
    top = report.top(k=3, head=0)
    assert len(top) == 3
    # Sorted descending by weight.
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_probe_layer_diff_drift_and_sentinel():
    """Compile y = 3*x, sample at multiple positions, and verify
    ``probe_layer_diff``:

    * reports ``first_drift_layer is None`` when given the correct
      reference (no drift beyond tolerance),
    * reports a populated ``first_drift_layer`` when given a wrong
      reference,
    * reports ``first_sentinel_layer`` at the earliest layer where
      the node equals a caller-supplied sentinel (set to the first
      per-layer value).
    """
    pos = create_pos_encoding()
    x = create_input("x", 2)
    y = multiply_const(x, 3.0)
    module = compile_headless(
        pos,
        io={"x": (x, y)},
        d=64,
        verbose=False,
    )

    inp = torch.tensor([[2.0, 4.0], [1.0, 5.0]])
    positions = [0, 1]
    true_ref = torch.tensor([[6.0, 12.0], [3.0, 15.0]])

    # Correct reference → no drift.
    ok = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        drift_threshold=0.5,
    )
    assert ok.records, "no layers traced"
    assert ok.first_drift_layer is None, (
        f"unexpected drift at layer {ok.first_drift_layer}: "
        f"max_abs_delta={ok.records[0].max_abs_delta}"
    )

    # Wrong reference → drift flagged at the first layer.
    bad_ref = torch.zeros_like(true_ref)
    bad = probe_layer_diff(
        module,
        inp,
        y,
        reference=bad_ref,
        positions=positions,
        drift_threshold=0.5,
    )
    assert bad.first_drift_layer is not None
    assert bad.first_drift_layer == bad.records[0].layer_index

    # Sentinel detection: the first layer's value should contain 6.0
    # (the top-left element).  Using that as the sentinel must flag the
    # first layer; a value that never surfaces must not flag anything.
    s_hit = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        sentinel=6.0,
        sentinel_tol=0.1,
    )
    assert s_hit.first_sentinel_layer == s_hit.records[0].layer_index
    assert s_hit.sentinel_value == 6.0

    s_miss = probe_layer_diff(
        module,
        inp,
        y,
        reference=true_ref,
        positions=positions,
        sentinel=-999.0,
        sentinel_tol=0.01,
    )
    assert s_miss.first_sentinel_layer is None


def test_probe_clean_on_v2_box_room(tiny_config):
    """V2 box_room at 16×20 — the config used by the existing
    ``test_v2_renders_box_room`` test — is known good.  The probe must
    find zero divergent nodes.
    """
    textures = default_texture_atlas()
    max_walls = 8
    max_bsp_nodes = 48
    graph_io, pos_encoding = build_game_graph(
        tiny_config,
        textures,
        max_walls=max_walls,
        max_coord=10.0,
        move_speed=0.3,
        turn_speed=4,
        max_bsp_nodes=max_bsp_nodes,
    )
    output_node = graph_io.concat_output()

    from torchwright.doom.game_graph import E8_INPUT

    d_render_fb = 2 * max_walls + 11
    d_sort_out = 8 + 5 + 3 + max_walls
    tex_h = textures[0].shape[1]
    input_values = {
        "input_backward": torch.tensor([[0.0]]),
        "input_forward": torch.tensor([[0.0]]),
        "input_strafe_left": torch.tensor([[0.0]]),
        "input_strafe_right": torch.tensor([[0.0]]),
        "input_turn_left": torch.tensor([[0.0]]),
        "input_turn_right": torch.tensor([[0.0]]),
        "player_angle": torch.tensor([[0.0]]),
        "player_x": torch.tensor([[0.0]]),
        "player_y": torch.tensor([[0.0]]),
        "render_feedback": torch.zeros(1, d_render_fb),
        "sort_feedback": torch.zeros(1, d_sort_out),
        "tex_col_input": torch.tensor([[0.0]]),
        "tex_pixels": torch.zeros(1, tex_h * 3),
        "texture_id_e8": torch.zeros(1, 8),
        "token_type": E8_INPUT.unsqueeze(0),
        "wall_ax": torch.tensor([[0.0]]),
        "wall_ay": torch.tensor([[0.0]]),
        "wall_bx": torch.tensor([[0.0]]),
        "wall_by": torch.tensor([[0.0]]),
        "wall_index": torch.tensor([[0.0]]),
        "wall_tex_id": torch.tensor([[0.0]]),
        "bsp_plane_nx": torch.tensor([[0.0]]),
        "bsp_plane_ny": torch.tensor([[0.0]]),
        "bsp_plane_d": torch.tensor([[0.0]]),
        "bsp_node_id_onehot": torch.zeros(1, max_bsp_nodes),
        "wall_bsp_coeffs": torch.zeros(1, max_bsp_nodes),
        "wall_bsp_const": torch.tensor([[0.0]]),
    }

    # The v2 graph has large intermediate values in square_signed and
    # signed_multiply chains (10^4–10^5 range).  The central ray sort
    # adds more chained multiplications, pushing fp32 accumulation
    # errors up to ~325 on values of ~10^4 (~3% relative).  atol=500
    # still catches real compilation bugs (missing layers, swapped
    # inputs) which show as errors of 1000+.
    report = probe_graph(
        output_node,
        pos_encoding,
        input_values,
        n_pos=1,
        d=2048,
        d_head=32,
        verbose=False,
        atol=500.0,
    )

    assert (
        report.nodes_checked
    ), "probe checked zero nodes — residual snapshots likely not populated"
    assert (
        report.first_divergent is None
    ), f"probe reported divergence on known-good graph:\n{report.format_short()}"
