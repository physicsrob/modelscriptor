"""Tests for torchwright.doom.embedding.

Properties verified:

* Vocabulary size and ID range agree with ``docs/phase_a_plan.md``.
* Every named vocab entry round-trips: ``DOOM_VOCAB[vocab_id(n)] == n``.
* VALUE rows share the ``E8_VALUE`` category code and their payload
  columns are a 4+4+4+4 factored one-hot of the 16-bit ID.
* Non-VALUE rows have zero payload columns.
* Category codes are pairwise distinct.
* ``equals_vector``-style detection succeeds for cross-category
  specific-ID queries (the case Part 1 actually uses).
"""

import torch

from torchwright.doom.embedding import (
    D_EMBED,
    DOOM_VOCAB,
    E8_VALUE,
    N_VALUES,
    V,
    W_EMBED,
    build_doom_embedding,
    category_code,
    embed_lookup,
    value_id,
    vocab_id,
)


def test_vocab_size_matches_plan():
    assert V == 65571
    assert len(DOOM_VOCAB) == V
    assert W_EMBED.shape == (V, 72)
    assert D_EMBED == 72


def test_id_ranges():
    # VALUE: 0..65535
    assert vocab_id("VALUE_0") == 0
    assert vocab_id("VALUE_65535") == 65535
    # THINKING_WALL: 65536..65543
    assert vocab_id("THINKING_WALL_0") == 65536
    assert vocab_id("THINKING_WALL_7") == 65543
    # Per-wall identifiers: 65544..65556
    assert vocab_id("BSP_RANK") == 65544
    assert vocab_id("HIT_Y") == 65556
    # RESOLVED: 65557..65559
    assert vocab_id("RESOLVED_X") == 65557
    assert vocab_id("RESOLVED_ANGLE") == 65559
    # Decode: 65560..65562
    assert vocab_id("SORTED_WALL") == 65560
    assert vocab_id("DONE") == 65562
    # Prompt-position: 65563..65570
    assert vocab_id("INPUT") == 65563
    assert vocab_id("PLAYER_ANGLE") == 65570


def test_vocab_roundtrip():
    # Every named token's id → name via DOOM_VOCAB matches.
    for name in [
        "VALUE_0",
        "VALUE_42",
        "VALUE_65535",
        "THINKING_WALL_3",
        "BSP_RANK",
        "HIT_FULL",
        "RESOLVED_Y",
        "DONE",
        "INPUT",
        "PLAYER_X",
    ]:
        assert DOOM_VOCAB[vocab_id(name)] == name


def test_value_id_helper_identity():
    assert value_id(0) == 0
    assert value_id(12345) == 12345
    assert value_id(N_VALUES - 1) == 65535


def test_value_row_layout():
    """Verify the 4+4+4+4 factored-one-hot encoding for a spot check."""
    vid = 0x1234  # h3=1, h2=2, h1=3, h0=4
    row = W_EMBED[vid]
    # Category: E8_VALUE in cols [0:8].
    assert torch.equal(row[0:8], E8_VALUE)
    # Payload one-hots at cols 8+1, 24+2, 40+3, 56+4.
    assert row[8 + 1].item() == 1.0
    assert row[24 + 2].item() == 1.0
    assert row[40 + 3].item() == 1.0
    assert row[56 + 4].item() == 1.0
    # Rest of the payload is zero.
    payload = row[8:72].clone()
    payload[0 * 16 + 1] = 0
    payload[1 * 16 + 2] = 0
    payload[2 * 16 + 3] = 0
    payload[3 * 16 + 4] = 0
    assert torch.all(payload == 0)


def test_value_row_all_ids_single_hot_per_block():
    """Across all 65,536 VALUE rows, each 16-wide block has exactly one hot."""
    value_rows = W_EMBED[:N_VALUES, 8:72]  # (65536, 64)
    for digit_idx in range(4):
        block = value_rows[:, digit_idx * 16 : (digit_idx + 1) * 16]
        assert torch.all(block.sum(dim=1) == 1.0)


def test_non_value_row_has_zero_payload():
    non_value_rows = W_EMBED[N_VALUES:, 8:72]  # (35, 64)
    assert torch.all(non_value_rows == 0)


def test_category_codes_pairwise_distinct():
    names = [
        "VALUE",
        *[f"THINKING_WALL_{i}" for i in range(8)],
        "BSP_RANK",
        "IS_RENDERABLE",
        "CROSS_A",
        "DOT_A",
        "CROSS_B",
        "DOT_B",
        "T_LO",
        "T_HI",
        "VIS_LO",
        "VIS_HI",
        "HIT_FULL",
        "HIT_X",
        "HIT_Y",
        "RESOLVED_X",
        "RESOLVED_Y",
        "RESOLVED_ANGLE",
        "SORTED_WALL",
        "RENDER",
        "DONE",
        "INPUT",
        "BSP_NODE",
        "WALL",
        "EOS",
        "TEX_COL",
        "PLAYER_X",
        "PLAYER_Y",
        "PLAYER_ANGLE",
    ]
    codes = [category_code(n) for n in names]
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            assert not torch.equal(
                codes[i], codes[j]
            ), f"duplicate category code for {names[i]} / {names[j]}"


def test_specific_id_detector_gap_sufficient():
    """For every cross-category pair used in Part 1 detectors, the
    ``equals_vector`` safety margin (``self_dot − cross_dot ≥ 1``) holds.

    Specifically: for a non-VALUE target row ``t``, every other row's
    dot product with ``t`` is at most ``t·t − 1``.  This guarantees
    ``equals_vector(row, t)`` saturates to −1 when the row is not ``t``
    (``embedding_step_sharpness=1.0`` → margin = 1).
    """
    # Check a representative set of "query" names that Part 1 actually
    # uses as specific-ID detectors.
    check_names = [
        "INPUT",
        "WALL",
        "EOS",
        "TEX_COL",
        "BSP_NODE",
        "SORTED_WALL",
        "RENDER",
        "PLAYER_X",
        "PLAYER_Y",
        "PLAYER_ANGLE",
        "HIT_FULL",
        "HIT_X",
        "HIT_Y",
        *[f"THINKING_WALL_{i}" for i in range(8)],
    ]
    for name in check_names:
        target = embed_lookup(name)
        self_dot = (target * target).sum().item()
        # Dot target against every row of W_EMBED.
        all_dots = W_EMBED @ target  # (V,)
        target_id = vocab_id(name)
        # Mask out the row itself (which matches by definition).
        mask = torch.ones(V, dtype=torch.bool)
        mask[target_id] = False
        max_off = all_dots[mask].max().item()
        gap = self_dot - max_off
        assert gap >= 1.0, (
            f"equals_vector margin too small for {name!r}: "
            f"self_dot={self_dot:.3f}, max off-target dot={max_off:.3f}, "
            f"gap={gap:.3f}"
        )


def test_category_only_value_detector():
    """``is_this_any_VALUE`` must saturate positive on every VALUE row
    and negative on every non-VALUE row, when the detector compares
    just cols [0:8] against ``E8_VALUE``."""
    self_dot = (E8_VALUE * E8_VALUE).sum().item()
    # Every VALUE row's first 8 cols ARE E8_VALUE — dot = self_dot.
    value_dots = W_EMBED[:N_VALUES, 0:8] @ E8_VALUE
    assert torch.all(value_dots == self_dot)
    # Every non-VALUE row's first 8 cols are a *different* category
    # code.  Dot must be ≤ self_dot − 1 (the safety gap).
    non_value_dots = W_EMBED[N_VALUES:, 0:8] @ E8_VALUE
    assert torch.all(non_value_dots <= self_dot - 1.0), (
        f"some non-VALUE row's E8_VALUE dot exceeds safety margin: "
        f"max = {non_value_dots.max().item()}, self_dot = {self_dot}"
    )


def test_build_doom_embedding_factory():
    emb = build_doom_embedding()
    assert emb.d_embed == 72
    assert emb.input_name == "token_ids"
    assert emb.max_vocab == V
    # compute() with a (n, 1) integer tensor (how CompiledHeadless slices input_specs).
    ids = torch.tensor([[vocab_id("THINKING_WALL_0")], [42], [vocab_id("HIT_FULL")]])
    out = emb.compute(n_pos=3, input_values={"token_ids": ids})
    assert out.shape == (3, 72)
    assert torch.equal(out[0], embed_lookup("THINKING_WALL_0"))
    assert torch.equal(out[1], W_EMBED[42])
    assert torch.equal(out[2], embed_lookup("HIT_FULL"))
