"""Round-trip layout test for the sort_feedback loop.

The SORTED→SORTED feedback path in ``game_graph.py`` writes fields into
``sort_feedback_out`` via one Concatenate (``_assemble_output``) and reads
them back via hardcoded offsets into ``extract_from`` (``_create_inputs``).
If either side's offsets drift, the loop silently feeds garbage back —
``prev_bsp_rank`` in particular is the sole surviving feedback signal
after Phase E, so an offset bug here corrupts the sort exhaustion logic.

This test builds a minimal graph that mirrors the layout of
``_assemble_output``'s sort_feedback Concatenate, extracts fields at the
same offsets ``_create_inputs`` uses, and verifies known values survive
the round trip.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.graph import Concatenate
from torchwright.ops.inout_nodes import (
    create_input, create_literal_value, create_pos_encoding,
)

from torchwright.doom.graph_utils import extract_from


_MAX_WALLS = 4
# Mirror of game_graph.py's d_sort_out formula.  Hardcoded here so a
# change to the formula on one side but not the other trips this test.
_D_SORT_OUT = 8 + 5 + 3 + _MAX_WALLS


@pytest.fixture(scope="module")
def feedback_roundtrip_module():
    """Compile a graph that writes sort_feedback_out then reads fields back.

    The write side mirrors ``_assemble_output``'s Concatenate layout; the
    read side mirrors ``_create_inputs``'s extract_from offsets.
    """
    pos = create_pos_encoding()

    # Inputs represent the SORTED stage's outputs that feed the Concatenate.
    sel_wall_data = create_input("sel_wall_data", 5)
    sel_bsp_rank = create_input("sel_bsp_rank", 1)
    vis_lo = create_input("vis_lo", 1)
    vis_hi = create_input("vis_hi", 1)
    sel_onehot = create_input("sel_onehot", _MAX_WALLS)

    # An 8-wide E8_SORTED_WALL constant — placeholder values, only width
    # matters for layout.  Matches the 8-wide token-type vector prepended
    # by _assemble_output.
    token_type = create_literal_value(
        torch.zeros(8), name="fake_token_type",
    )

    # Write side — must match _assemble_output's Concatenate exactly.
    sort_feedback_out = Concatenate([
        token_type,      # 8
        sel_wall_data,   # 5
        sel_bsp_rank,    # 1
        vis_lo,          # 1
        vis_hi,          # 1
        sel_onehot,      # max_walls
    ])
    assert len(sort_feedback_out) == _D_SORT_OUT, (
        f"sort_feedback_out width {len(sort_feedback_out)} != "
        f"_D_SORT_OUT {_D_SORT_OUT} — update the Concatenate layout "
        f"or the formula."
    )

    # Read side — must match _create_inputs's extract_from offsets.
    extracted_prev_bsp_rank = extract_from(
        sort_feedback_out, _D_SORT_OUT, 8 + 5, 1, "extracted_prev_bsp_rank",
    )

    output = extracted_prev_bsp_rank
    return compile_headless(
        output, pos, d=512, d_head=16, max_layers=40, verbose=False,
    )


def _pack(module, rows: list[dict]) -> torch.Tensor:
    d_input = max(s + w for _, s, w in module._input_specs)
    T = len(rows)
    t = torch.zeros(T, d_input, dtype=torch.float32)
    for i, row in enumerate(rows):
        for name, start, width in module._input_specs:
            t[i, start:start + width] = torch.tensor(
                row[name], dtype=torch.float32,
            ).reshape(width)
    return t


@pytest.mark.parametrize("sel_bsp_rank,label", [
    (-1.0, "eos_seed_value"),
    (0.0,  "first_pick"),
    (3.0,  "last_pick"),
    (2.5,  "non_integer_guard"),
])
def test_sort_feedback_round_trip(
    feedback_roundtrip_module, sel_bsp_rank, label,
):
    """Writing ``sel_bsp_rank`` at offset 13 must round-trip through
    the Concatenate and out of extract_from unchanged.
    """
    row = {
        "sel_wall_data": [0.1, 0.2, 0.3, 0.4, 0.5],
        "sel_bsp_rank": sel_bsp_rank,
        "vis_lo": 7.0,
        "vis_hi": 11.0,
        "sel_onehot": [0.5, 0.5, 0.5, 0.5],
    }
    # compile_headless requires at least 2 positions.
    pad = {k: ([0.0] * len(v) if isinstance(v, list) else 0.0) for k, v in row.items()}
    inputs = _pack(feedback_roundtrip_module, [row, pad])
    with torch.no_grad():
        out = feedback_roundtrip_module(inputs)

    got_bsp_rank = out[0, 0].item()
    assert abs(got_bsp_rank - sel_bsp_rank) < 1e-3, (
        f"{label}: prev_bsp_rank round-trip mismatch; "
        f"wrote {sel_bsp_rank}, read {got_bsp_rank}"
    )


def test_eos_seed_layout_preserves_prev_bsp_rank_slot():
    """The EOS seed in _assemble_output writes -1 at the prev_bsp_rank
    slot.  This test rebuilds the seed locally and verifies extract_from
    at the same offset returns -1.
    """
    pos = create_pos_encoding()

    token_type = create_literal_value(
        torch.zeros(8), name="eos_fake_token_type",
    )
    resolved_x = create_input("resolved_x", 1)
    resolved_y = create_input("resolved_y", 1)
    new_angle = create_input("new_angle", 1)

    # Mirror of _assemble_output's eos_sort_seed Concatenate.
    eos_sort_seed = Concatenate([
        token_type,                                                   # 8
        resolved_x, resolved_y, new_angle,                            # 3
        create_literal_value(torch.zeros(2), name="pad1"),            # 2
        create_literal_value(torch.tensor([-1.0]), name="prev_bsp"),  # 1
        create_literal_value(
            torch.zeros(2 + _MAX_WALLS), name="pad2",
        ),                                                            # 2 + max_walls
    ])
    assert len(eos_sort_seed) == _D_SORT_OUT, (
        f"eos_sort_seed width {len(eos_sort_seed)} != _D_SORT_OUT "
        f"{_D_SORT_OUT}; pad widths are out of sync with the feedback layout."
    )

    extracted = extract_from(
        eos_sort_seed, _D_SORT_OUT, 8 + 5, 1, "extracted_eos_prev_bsp_rank",
    )
    module = compile_headless(
        extracted, pos, d=256, d_head=16, max_layers=30, verbose=False,
    )
    row = {"resolved_x": 0.0, "resolved_y": 0.0, "new_angle": 0.0}
    pad = {k: 0.0 for k in row}
    inputs = _pack(module, [row, pad])
    with torch.no_grad():
        out = module(inputs)
    got = out[0, 0].item()
    assert abs(got - (-1.0)) < 1e-3, (
        f"EOS seed prev_bsp_rank slot: wrote -1, read {got}"
    )
