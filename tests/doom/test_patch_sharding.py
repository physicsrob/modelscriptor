"""Integration tests for the DOOM game graph's (col, patch) delta.

Since Option A, the game graph takes the **current** (col, patch) as
inputs and predicts the **next** (col, patch) as outputs — the host
threads the output back into the next step's input.  The delta is:

    next_patch = (cur_patch + 1) mod shards_per_col
    wrap       = floor_div(cur_patch + 1, shards_per_col)    # 0 or 1
    next_col   = cur_col + wrap

There is no longer any position-0 special case in the graph; the host
just writes ``(cur_col=0, cur_patch=0)`` at step 0 and the same delta
runs uniformly at every step.

These tests build a minimal graph that reproduces only the delta math
(no rendering, no game logic), compile it, and run a single prefill
call where every position ``t`` gets its canonical input
``(cur_col=t//shards, cur_patch=t%shards)``.  The output at position
``t`` should match the ground-truth ``next`` for that ``cur``.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    mod_const,
    thermometer_floor_div,
)
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

# (W, H, rp) — parametrized over the configs the DOOM code uses.
# ``shards_per_col = H // rp`` drives the delta's wrap divisor; the
# per-step cost of the compiled delta logic no longer depends on
# ``total_positions`` because the staircase ops run with max_value =
# shards_per_col (≤ ~20), not the full frame size.
#
#   (16, 12, 6)    —   32 positions, shards=2
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
    """Reproduce game_graph.py's next-col/next-patch delta in isolation.

    Returns ``(output_node, pos_encoding, total_positions)`` where
    ``output_node`` is a width-2 Concatenate of
    ``(next_col_idx, next_patch_idx_in_col)``.  The graph has exactly
    two user inputs — ``cur_col_idx`` and ``cur_patch_idx_in_col`` —
    mirroring the post-Option-A game graph's feedback scheme.
    """
    assert H % rp == 0, f"screen_height {H} must be divisible by rows_per_patch {rp}"
    shards_per_col = H // rp
    total_positions = W * shards_per_col

    pos_encoding = create_pos_encoding()
    cur_col_idx = create_input("cur_col_idx", 1)
    cur_patch_idx_in_col = create_input("cur_patch_idx_in_col", 1)

    patch_plus_one = add_const(cur_patch_idx_in_col, 1.0)
    next_patch_idx_in_col = mod_const(
        patch_plus_one,
        shards_per_col,
        max_value=shards_per_col,
    )
    wrap = thermometer_floor_div(
        patch_plus_one,
        shards_per_col,
        max_value=shards_per_col,
    )
    next_col_idx = add(cur_col_idx, wrap)

    output = Concatenate([next_col_idx, next_patch_idx_in_col])
    return output, pos_encoding, total_positions


def _build_input_tensor(W: int, H: int, rp: int) -> torch.Tensor:
    """One row per position ``t`` with the canonical cur = divmod(t, shards)."""
    shards_per_col = H // rp
    total = W * shards_per_col

    t = torch.arange(total, dtype=torch.float32)
    inputs = torch.empty(total, 2)
    inputs[:, 0] = t // shards_per_col  # cur_col_idx
    inputs[:, 1] = t % shards_per_col  # cur_patch_idx_in_col
    return inputs


def _expected_next(cur_col: torch.Tensor, cur_patch: torch.Tensor, shards_per_col: int):
    patch_plus_one = cur_patch + 1
    next_patch = patch_plus_one % shards_per_col
    wrap = patch_plus_one // shards_per_col
    next_col = cur_col + wrap
    return next_col, next_patch


@pytest.mark.parametrize("W,H,rp", SHARDING_CASES)
def test_delta_matches_ground_truth(W, H, rp):
    """For every input ``(cur_col, cur_patch)`` the graph emits the
    mathematically-exact ``(next_col, next_patch)`` from the wrap rule."""
    output, pos_encoding, total_positions = _build_sharding_graph(W, H, rp)
    shards_per_col = H // rp

    module = compile_headless(
        output,
        pos_encoding,
        d=1024,
        d_head=16,
        max_layers=50,
        verbose=False,
    )

    inputs = _build_input_tensor(W, H, rp)
    with torch.no_grad():
        out = module(inputs)  # (total_positions, 2)

    next_col_out = out[:, 0]
    next_patch_out = out[:, 1]

    cur_col = inputs[:, 0]
    cur_patch = inputs[:, 1]
    expected_next_col, expected_next_patch = _expected_next(
        cur_col,
        cur_patch,
        shards_per_col,
    )

    col_err = (next_col_out - expected_next_col).abs()
    patch_err = (next_patch_out - expected_next_patch).abs()
    col_max = col_err.max().item()
    patch_max = patch_err.max().item()
    col_at = int(col_err.argmax().item())
    patch_at = int(patch_err.argmax().item())
    assert col_max < 0.5, (
        f"next_col max error {col_max:.3f} at W={W}, H={H}, rp={rp}, "
        f"position {col_at} (cur=({int(cur_col[col_at])}, "
        f"{int(cur_patch[col_at])}); got {next_col_out[col_at].item():.3f}, "
        f"expected {expected_next_col[col_at].item():.0f})"
    )
    assert patch_max < 0.5, (
        f"next_patch max error {patch_max:.3f} at W={W}, H={H}, rp={rp}, "
        f"position {patch_at} (cur=({int(cur_col[patch_at])}, "
        f"{int(cur_patch[patch_at])}); got {next_patch_out[patch_at].item():.3f}, "
        f"expected {expected_next_patch[patch_at].item():.0f})"
    )


@pytest.mark.parametrize("W,H,rp", SHARDING_CASES)
def test_delta_iterated_covers_full_grid(W, H, rp):
    """Seeding with ``(0, 0)`` and iterating the delta must visit every
    ``(col, patch)`` in the expected grid exactly once in raster order.

    We run the graph in *prefill* with the canonical inputs, then
    simulate the autoregressive feedback on the host: start at (0, 0),
    read output row 0, use it as input for row 1, etc., and check that
    every step lands on the correct cell.  This is the same check
    ``step_frame_iter`` effectively does in decode mode.
    """
    output, pos_encoding, total_positions = _build_sharding_graph(W, H, rp)
    shards_per_col = H // rp

    module = compile_headless(
        output,
        pos_encoding,
        d=1024,
        d_head=16,
        max_layers=50,
        verbose=False,
    )

    inputs = _build_input_tensor(W, H, rp)
    with torch.no_grad():
        out = module(inputs)

    # Walk the delta chain starting from (0, 0).
    visited = [(0, 0)]
    cur_col, cur_patch = 0, 0
    for t in range(total_positions - 1):
        # The prefill output at position t is what the graph would emit
        # given the input at row t.  Since the test inputs contain the
        # canonical cur = divmod(t, shards), out[t] should give
        # next = divmod(t+1, shards).
        next_col = int(round(out[t, 0].item()))
        next_patch = int(round(out[t, 1].item()))
        visited.append((next_col, next_patch))
        cur_col, cur_patch = next_col, next_patch

    expected = [
        (t // shards_per_col, t % shards_per_col) for t in range(total_positions)
    ]
    assert visited == expected, (
        f"W={W}, H={H}, rp={rp}: iterated delta diverged from raster order. "
        f"First mismatch at step "
        f"{next(i for i, (a, b) in enumerate(zip(visited, expected)) if a != b)}"
    )
