"""Integration tests for the DOOM game graph's patch_row_start / col_idx outputs.

The game graph emits two self-identifying scalars at every token
position — ``col_idx`` and ``patch_row_start`` — that tell the host
where to paste each rendered patch.  Since Fix B, those values are
computed by a **one-step delta** on inputs that the host threads back
from the previous step's output (see ``game_graph.py`` around
``prev_col_idx`` / ``prev_patch_idx_in_col``):

    patch_new = (prev_patch + 1) mod shards_per_col
    wrap      = floor_div(prev_patch + 1, shards_per_col)   # 0 or 1
    col_new   = prev_col + wrap

At position 0 both outputs are forced to 0 via the existing
``is_pos_0`` select, which is driven by ``position_scalar`` but only
needs the threshold at 0.5 (where the sin approximation is exact).

These tests build a minimal graph that reproduces exactly the delta
logic in ``game_graph.py`` (no rendering, no game logic — just the two
scalars) and verify the compiled module's output against the
mathematical ground truth at every position, across a parameter sweep
that covers the real DOOM shipping scale.

Pre-existing coverage was limited to ``test_compiled_patch_equivalence``
at ``(W=16, H=12, rp=6)`` — 32 positions, ``shards_per_col=2``.  Nothing
exercised larger configurations numerically.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.graph.misc import LiteralValue
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    compare,
    mod_const,
    multiply_const,
    thermometer_floor_div,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding
from torchwright.ops.map_select import select


# (W, H, rp) — parametrized over the configs the DOOM code has been
# asked to handle.  ``shards_per_col = H // rp`` sets how often the
# patch index wraps, but unlike the old position_scalar-driven design
# the per-step cost no longer depends on ``total_positions``.
#
#   (16, 12, 6)    —   32 positions, shards=2  (existing test_game_graph scale)
#   (16, 12, 4)    —   48 positions, shards=3
#   (16, 12, 3)    —   64 positions, shards=4
#   (32, 40, 10)   —  320 positions, shards=10
#   (64, 80, 8)    —  640 positions, shards=10 (old to_onnx default)
#   (40, 100, 10)  —  400 positions, shards=10
#   (160, 100, 10) — 1600 positions, shards=10 (new to_onnx default)
SHARDING_CASES = [
    (16, 12, 6),
    (16, 12, 4),
    (16, 12, 3),
    (32, 40, 10),
    (64, 80, 8),
    (40, 100, 10),
    (160, 100, 10),
]


def _build_sharding_graph(W: int, H: int, rp: int):
    """Reproduce game_graph.py's delta-based col_idx / patch_row_start.

    Returns ``(output_node, pos_encoding, total_positions)`` where
    output_node is a width-2 Concatenate of (col_idx, patch_row_start).
    The graph has exactly two user inputs — ``prev_col_idx`` and
    ``prev_patch_idx_in_col`` — mirroring the new game graph's per-step
    feedback scheme.
    """
    assert H % rp == 0, f"screen_height {H} must be divisible by rows_per_patch {rp}"
    shards_per_col = H // rp
    total_positions = W * shards_per_col

    pos_encoding = create_pos_encoding()
    prev_col_idx = create_input("prev_col_idx", 1)
    prev_patch_idx_in_col = create_input("prev_patch_idx_in_col", 1)

    # is_pos_0: +1 at position 0, -1 elsewhere.  This drives the select
    # that overrides the delta output to (0, 0) at the first step.
    # position_scalar(0) = 0 exactly, so this threshold is safe at any
    # sequence length.
    position_scalar = pos_encoding.get_position_scalar()
    is_pos_0 = compare(
        position_scalar, 0.5, true_level=-1.0, false_level=1.0,
    )

    patch_plus_one = add_const(prev_patch_idx_in_col, 1.0)
    patch_new_from_delta = mod_const(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    wrap = thermometer_floor_div(
        patch_plus_one, shards_per_col, max_value=shards_per_col,
    )
    col_new_from_delta = add(prev_col_idx, wrap)

    zero = LiteralValue(torch.tensor([0.0]), name="zero_col_patch")
    col_idx = select(is_pos_0, zero, col_new_from_delta)
    patch_idx_in_col = select(is_pos_0, zero, patch_new_from_delta)
    patch_row_start = multiply_const(patch_idx_in_col, float(rp))

    output = Concatenate([col_idx, patch_row_start])
    return output, pos_encoding, total_positions


def _build_input_tensor(W: int, H: int, rp: int) -> torch.Tensor:
    """Per-position (prev_col, prev_patch) that the host would thread.

    Position t > 0 sees the (col, patch) emitted by step t-1, i.e.
    ``divmod(t-1, shards_per_col)``.  Position 0 gets whatever (the
    select overrides it to (0, 0)), so we pass zeros there.
    """
    shards_per_col = H // rp
    total = W * shards_per_col

    inputs = torch.zeros(total, 2)
    for t in range(1, total):
        prev_step = t - 1
        inputs[t, 0] = float(prev_step // shards_per_col)  # prev_col_idx
        inputs[t, 1] = float(prev_step % shards_per_col)   # prev_patch_idx_in_col
    return inputs


@pytest.mark.parametrize("W,H,rp", SHARDING_CASES)
def test_patch_sharding_outputs_match_ground_truth(W, H, rp):
    """At every token position, the compiled graph emits (col_idx,
    patch_row_start) equal to the divmod of the position."""
    output, pos_encoding, total_positions = _build_sharding_graph(W, H, rp)
    shards_per_col = H // rp

    module = compile_headless(
        output, pos_encoding,
        d=1024, d_head=16, max_layers=50, verbose=False,
    )

    inputs = _build_input_tensor(W, H, rp)
    with torch.no_grad():
        out = module(inputs)  # (total_positions, 2)

    col_idx_out = out[:, 0]
    patch_row_start_out = out[:, 1]

    positions = torch.arange(total_positions)
    expected_col = (positions // shards_per_col).to(col_idx_out.dtype)
    expected_pr = ((positions % shards_per_col) * rp).to(patch_row_start_out.dtype)

    col_err = (col_idx_out - expected_col).abs()
    pr_err = (patch_row_start_out - expected_pr).abs()
    col_max = col_err.max().item()
    pr_max = pr_err.max().item()
    col_at = int(col_err.argmax().item())
    pr_at = int(pr_err.argmax().item())
    assert col_max < 0.5, (
        f"col_idx max error {col_max:.3f} at W={W}, H={H}, rp={rp}, "
        f"position {col_at} (got {col_idx_out[col_at].item():.3f}, "
        f"expected {expected_col[col_at].item():.0f})"
    )
    assert pr_max < 0.5, (
        f"patch_row_start max error {pr_max:.3f} at W={W}, H={H}, rp={rp}, "
        f"position {pr_at} (got {patch_row_start_out[pr_at].item():.3f}, "
        f"expected {expected_pr[pr_at].item():.0f})"
    )


@pytest.mark.parametrize("W,H,rp", SHARDING_CASES)
def test_patch_sharding_outputs_cover_full_range(W, H, rp):
    """Every (col_idx, patch_row_start) pair in the expected grid must
    appear in the output.  Catches any mode-collapse where the delta
    logic would pin to a subset."""
    output, pos_encoding, total_positions = _build_sharding_graph(W, H, rp)
    shards_per_col = H // rp

    module = compile_headless(
        output, pos_encoding,
        d=1024, d_head=16, max_layers=50, verbose=False,
    )

    inputs = _build_input_tensor(W, H, rp)
    with torch.no_grad():
        out = module(inputs)

    col_int = out[:, 0].round().long().tolist()
    pr_int = out[:, 1].round().long().tolist()
    observed_pairs = set(zip(col_int, pr_int))

    expected_pairs = {
        (c, r * rp)
        for c in range(W)
        for r in range(shards_per_col)
    }
    missing = expected_pairs - observed_pairs
    extras = observed_pairs - expected_pairs
    assert not missing, (
        f"W={W}, H={H}, rp={rp}: {len(missing)} expected (col, patch_row_start) "
        f"pairs never emitted by the compiled graph. Sample missing: "
        f"{sorted(missing)[:5]}"
    )
    assert not extras, (
        f"W={W}, H={H}, rp={rp}: compiled graph emitted {len(extras)} pairs "
        f"outside the expected grid. Sample: {sorted(extras)[:5]}"
    )
