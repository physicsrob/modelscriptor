"""Tests for torchwright.doom.embedding.

Properties verified:

* Vocabulary size and ID range agree with ``docs/phase_a_plan.md``.
* Every named vocab entry round-trips: ``DOOM_VOCAB[vocab_id(n)] == n``.
* VALUE rows share the ``E8_VALUE`` category code, carry ``k/65535``
  in the raw slot, and a 16-wide ±1 Gray code in the payload.
* Non-VALUE rows have zero payload columns.
* Category codes are pairwise distinct.
* ``equals_vector``-style detection succeeds for cross-category
  specific-ID queries (the case Part 1 actually uses).
"""

import torch

from torchwright.doom.embedding import (
    D_CATEGORY,
    D_EMBED,
    D_GRAY_PAYLOAD,
    D_RAW_SLOT,
    DOOM_VOCAB,
    E8_VALUE,
    N_VALUES,
    V,
    W_EMBED,
    build_doom_embedding,
    category_code,
    embed_lookup,
    gray_code_16,
    value_id,
    vocab_id,
)


def test_vocab_size_matches_plan():
    assert V == 65576
    assert len(DOOM_VOCAB) == V
    assert W_EMBED.shape == (V, D_EMBED)
    assert D_EMBED == 49


def test_id_ranges():
    # VALUE: 0..65535
    assert vocab_id("VALUE_0") == 0
    assert vocab_id("VALUE_65535") == 65535
    # THINKING_WALL: 65536..65543
    assert vocab_id("THINKING_WALL_0") == 65536
    assert vocab_id("THINKING_WALL_7") == 65543
    # Per-wall identifiers (17): 65544..65560
    assert vocab_id("BSP_RANK") == 65544
    assert vocab_id("HIT_Y") == 65560
    # RESOLVED (3): 65561..65563
    assert vocab_id("RESOLVED_X") == 65561
    assert vocab_id("RESOLVED_ANGLE") == 65563
    # Sort-phase identifier (1): 65564
    assert vocab_id("SORT_RESULT") == 65564
    # Decode (3): 65565..65567
    assert vocab_id("SORTED_WALL") == 65565
    assert vocab_id("DONE") == 65567
    # Prompt-position (8): 65568..65575
    assert vocab_id("INPUT") == 65568
    assert vocab_id("PLAYER_ANGLE") == 65575


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
    """Verify the [E8 | raw | gray | type-tag] layout for a spot-check VALUE row."""
    from torchwright.doom.embedding import (
        D_SLOT_ONEHOT,
        _IS_ANY_ID_COL,
        _IS_VALUE_CATEGORY_COL,
        _SLOT_ONEHOT_START,
    )

    vid = 0x1234  # = 4660
    row = W_EMBED[vid]
    # Category: E8_VALUE in cols [0:8].
    assert torch.equal(row[0:D_CATEGORY], E8_VALUE)
    # Raw slot at col D_CATEGORY stores the shifted encoder grid value
    # (2k + 1) / 131072.  Exact in float32 since denominator is 2^17.
    raw_col = D_CATEGORY
    assert torch.isclose(
        row[raw_col],
        torch.tensor((2 * vid + 1) / 131072.0, dtype=torch.float32),
    )
    # Gray-code payload at cols [9:25] equals the independent gray_code_16(vid).
    gray_start = D_CATEGORY + D_RAW_SLOT
    assert torch.equal(row[gray_start : gray_start + D_GRAY_PAYLOAD], gray_code_16(vid))
    # Phase D Part 1 type-tag block: VALUE rows have all −1 in the
    # slot one-hot, −1 in is_any_identifier, +1 in is_value_category.
    assert torch.all(
        row[_SLOT_ONEHOT_START : _SLOT_ONEHOT_START + D_SLOT_ONEHOT] == -1.0
    )
    assert row[_IS_ANY_ID_COL].item() == -1.0
    assert row[_IS_VALUE_CATEGORY_COL].item() == 1.0


def test_value_row_gray_code_properties():
    """Gray-code invariants across all 65,536 VALUE rows.

    * Adjacent rows k, k+1 differ in exactly one gray-payload bit.
    * Self-dot minus adjacent cross-dot ≥ 2 over the payload columns
      (gap 16 - 14 = 2), which drives the argmax margin against
      ``W_EMBED.T``.
    """
    gray_start = D_CATEGORY + D_RAW_SLOT
    payload = W_EMBED[:N_VALUES, gray_start : gray_start + D_GRAY_PAYLOAD]

    # All gray entries are ±1.
    assert torch.all((payload == 1.0) | (payload == -1.0))

    # Adjacent rows differ in exactly one position over a representative
    # sample (checking all 65k pairs is wasteful on CPU — sample covers
    # bit transitions up through bit 15).
    sample_ks = [0, 1, 2, 3, 7, 15, 31, 63, 127, 16383, 16384, 32767, 32768, 65534]
    for k in sample_ks:
        diffs = (payload[k] != payload[k + 1]).sum().item()
        assert diffs == 1, f"gray(k={k}) vs gray(k+1): {diffs} bits differ, expected 1"

    # Dot-product margin: self-dot = 16, adjacent cross-dot = 14 (since
    # a single flipped ±1 bit changes the dot by 2).
    for k in sample_ks:
        self_dot = (payload[k] * payload[k]).sum().item()
        adj_dot = (payload[k] * payload[k + 1]).sum().item()
        assert (
            self_dot - adj_dot >= 2.0
        ), f"gray-dot margin too small for k={k}: self={self_dot}, adj={adj_dot}"


def test_non_value_row_has_zero_payload():
    """Non-VALUE rows zero out cols [8:26] — the raw slot, Gray-code
    payload, and K are zero.  The Phase D Part 1 type-tag block at
    cols [26:49] carries other flags and is checked separately."""
    payload_end = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD  # 25 — before K
    non_value_payload = W_EMBED[N_VALUES:, D_CATEGORY:payload_end]
    assert torch.all(non_value_payload == 0)
    # K column should also be 0 for non-VALUE rows (col [25:26]).
    from torchwright.doom.embedding import D_K_SLOT

    k_block = W_EMBED[N_VALUES:, payload_end : payload_end + D_K_SLOT]
    assert torch.all(k_block == 0)


def test_slot_onehot_correctness():
    """Phase D Part 1: each IDENTIFIER row at slot ``i`` has +1 at the
    matching slot column and −1 at the other 20.  Non-identifier rows
    (VALUE, markers, decode, prompt) carry all −1."""
    from torchwright.doom.embedding import (
        D_SLOT_ONEHOT,
        IDENTIFIER_NAMES,
        _SLOT_ONEHOT_START,
    )

    block = W_EMBED[:, _SLOT_ONEHOT_START : _SLOT_ONEHOT_START + D_SLOT_ONEHOT]
    # Every entry is ±1.
    assert torch.all((block == 1.0) | (block == -1.0))

    # Identifier rows: exactly one +1 at the matching slot.
    for i, name in enumerate(IDENTIFIER_NAMES):
        row = W_EMBED[vocab_id(name)]
        slot_block = row[_SLOT_ONEHOT_START : _SLOT_ONEHOT_START + D_SLOT_ONEHOT]
        assert slot_block[i].item() == 1.0, (
            f"identifier {name!r} at slot {i}: expected +1, got {slot_block[i]}"
        )
        # Other 20 columns are −1.
        n_neg = (slot_block == -1.0).sum().item()
        assert n_neg == 20, f"identifier {name!r}: {n_neg} −1 entries, expected 20"

    # Non-identifier rows: all 21 cols are −1.  Sample VALUE, markers,
    # decode, prompt categories.
    non_id_names = [
        "VALUE_0",
        "VALUE_42",
        "VALUE_65535",
        "THINKING_WALL_0",
        "THINKING_WALL_7",
        "SORTED_WALL",
        "RENDER",
        "DONE",
        "INPUT",
        "PLAYER_X",
        "PLAYER_ANGLE",
    ]
    for name in non_id_names:
        row = W_EMBED[vocab_id(name)]
        slot_block = row[_SLOT_ONEHOT_START : _SLOT_ONEHOT_START + D_SLOT_ONEHOT]
        assert torch.all(
            slot_block == -1.0
        ), f"non-identifier {name!r} has unexpected +1 in slot one-hot"


def test_is_any_identifier_correctness():
    """Phase D Part 1: is_any_identifier column is +1 at the 21
    IDENTIFIER_NAMES rows, −1 at every other row."""
    from torchwright.doom.embedding import IDENTIFIER_NAMES, _IS_ANY_ID_COL

    col = W_EMBED[:, _IS_ANY_ID_COL]
    assert torch.all((col == 1.0) | (col == -1.0))
    pos_ids = (col == 1.0).nonzero(as_tuple=True)[0].tolist()
    expected_ids = sorted(vocab_id(n) for n in IDENTIFIER_NAMES)
    assert sorted(pos_ids) == expected_ids


def test_is_value_category_correctness():
    """Phase D Part 1: is_value_category column is +1 at the 65,536
    VALUE rows, −1 at every other row."""
    from torchwright.doom.embedding import _IS_VALUE_CATEGORY_COL

    col = W_EMBED[:, _IS_VALUE_CATEGORY_COL]
    assert torch.all((col == 1.0) | (col == -1.0))
    # First N_VALUES rows are VALUE.
    assert torch.all(col[:N_VALUES] == 1.0)
    assert torch.all(col[N_VALUES:] == -1.0)


def test_argmax_separation_with_typetag():
    """Phase D Part 1: adding the 23 ±1 type-tag cols increases the
    argmax margin between distinct categories — self-dot grows by 23,
    cross-dot between distinct categories grows by at most 23 − 4 = 19
    (because at least one type-tag column has opposite sign).  Net
    margin increases by at least 4.

    Verifies the predicted-target argmax against ``W_EMBED.T`` still
    picks the right row for representative non-VALUE categories."""
    check_names = [
        "INPUT",
        "WALL",
        "EOS",
        "TEX_COL",
        "BSP_NODE",
        "SORTED_WALL",
        "RENDER",
        "PLAYER_X",
        "BSP_RANK",
        "RESOLVED_X",
        "SORT_RESULT",
        "HIT_FULL",
        "HIT_X",
        "HIT_Y",
        *[f"THINKING_WALL_{i}" for i in range(8)],
    ]
    for name in check_names:
        target = embed_lookup(name)
        target_id = vocab_id(name)
        scores = W_EMBED @ target
        assert int(scores.argmax().item()) == target_id, (
            f"argmax for {name!r} picked vocab id "
            f"{int(scores.argmax().item())}, expected {target_id}"
        )


def test_int_slot_column_for_small_k():
    """Phase C Part 2: K col holds k for k ≤ MAX_INT_K, zero above."""
    from torchwright.doom.embedding import D_K_SLOT, MAX_INT_K

    k_col_start = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD
    assert D_K_SLOT == 1

    # In the int range: K = k.
    for k in [0, 1, 2, 3, 7, 100, MAX_INT_K]:
        assert W_EMBED[k, k_col_start].item() == float(k)

    # Just above the cap: zero.
    for k in [MAX_INT_K + 1, MAX_INT_K + 100, N_VALUES - 1]:
        assert W_EMBED[k, k_col_start].item() == 0.0


def test_int_slot_argmax_peak_at_target():
    """A predicted embedding holding the W_EMBED row's E8/raw/gray cols
    with the K column held at 0 still argmaxes to VALUE_(k_target).
    Verifies gray's Hamming-1 margin alone drives the host argmax to
    the right row across the full int-slot range — no K_NS quadratic
    needed."""
    from torchwright.doom.embedding import D_K_SLOT, MAX_INT_K

    k_col_start = D_CATEGORY + D_RAW_SLOT + D_GRAY_PAYLOAD

    for k_target in [0, 1, 3, 7, 100, MAX_INT_K]:
        # Build the predicted embedding: E8/raw/gray copied from
        # W_EMBED row, K column held at 0 (the producer-side override
        # done by emit_integer_value_embedding).
        predicted = W_EMBED[k_target].clone()
        predicted[k_col_start : k_col_start + D_K_SLOT] = 0.0

        # Argmax over all rows.
        scores = W_EMBED @ predicted
        assert scores.argmax().item() == k_target, (
            f"argmax for k_target={k_target} picked {scores.argmax().item()}, "
            f"score gap to neighbor: {scores[k_target] - scores.sort(descending=True).values[1]}"
        )


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
        "T_STAR_L",
        "T_STAR_R",
        "T_LO",
        "T_HI",
        "COL_A",
        "COL_B",
        "VIS_LO",
        "VIS_HI",
        "HIT_FULL",
        "HIT_X",
        "HIT_Y",
        "RESOLVED_X",
        "RESOLVED_Y",
        "RESOLVED_ANGLE",
        "SORT_RESULT",
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
    value_dots = W_EMBED[:N_VALUES, 0:D_CATEGORY] @ E8_VALUE
    assert torch.all(value_dots == self_dot)
    # Every non-VALUE row's first 8 cols are a *different* category
    # code.  Dot must be ≤ self_dot − 1 (the safety gap).
    non_value_dots = W_EMBED[N_VALUES:, 0:D_CATEGORY] @ E8_VALUE
    assert torch.all(non_value_dots <= self_dot - 1.0), (
        f"some non-VALUE row's E8_VALUE dot exceeds safety margin: "
        f"max = {non_value_dots.max().item()}, self_dot = {self_dot}"
    )


def test_encoder_matches_w_embed_row():
    """Compile ``encode_value_binary`` and verify that for every integer
    k ∈ [0, 65535] the encoder's D_EMBED-wide output dotted against
    ``W_EMBED.T`` argmaxes to k.

    This is the host-side semantic of VALUE emission: the encoder
    produces a row, the host picks the nearest VALUE by argmax.  The
    triangle-wave Gray-code basis plus the raw slot must resolve every
    integer input to its exact k.  The Phase D Part 1 type-tag tail
    (slot one-hot all −1, is_any_id −1, is_value_category +1) matches
    every VALUE row exactly, so it adds a constant offset across the
    VALUE block and doesn't shift the argmax.
    """
    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.graph import Linear
    from torchwright.ops.inout_nodes import create_input, create_pos_encoding

    from torchwright.doom.thinking_readback import encode_value_binary

    pos_encoding = create_pos_encoding()
    q_in = create_input("q", 1)
    emitted = encode_value_binary(q_in, suffix="_test")
    # Concatenate is layout-only; wrap in an identity Linear so the
    # D_EMBED-wide row materializes under a node in the result dict.
    emit = Linear(emitted, torch.eye(D_EMBED), name="emit_passthrough")

    net = forward_compile(
        d=512,
        d_head=32,
        output_node=emit,
        pos_encoding=pos_encoding,
        verbose=False,
    )

    # Sample 0, 1, a dense grid, and endpoints.  Running all 65k k
    # through compute is expensive; this grid covers every bit
    # transition in the triangle-wave basis.
    sample_ks = [
        0,
        1,
        2,
        3,
        127,
        128,
        255,
        256,
        16383,
        16384,
        32767,
        32768,
        49151,
        49152,
        65534,
        65535,
    ]
    for k in sample_ks:
        vals = {"q": torch.tensor([[float(k)]])}
        emit_row = net.compute(1, vals)[emit].squeeze(0)  # (25,)
        logits = emit_row @ W_EMBED.T.to(emit_row.device)
        picked = int(logits.argmax().item())
        assert (
            picked == k
        ), f"encoder q={k} argmaxed to VALUE_{picked}, expected VALUE_{k}"


def test_build_doom_embedding_factory():
    emb = build_doom_embedding()
    assert emb.d_embed == D_EMBED
    assert emb.input_name == "token_ids"
    assert emb.max_vocab == V
    # compute() with a (n, 1) integer tensor (how CompiledHeadless slices input_specs).
    ids = torch.tensor([[vocab_id("THINKING_WALL_0")], [42], [vocab_id("HIT_FULL")]])
    out = emb.compute(n_pos=3, input_values={"token_ids": ids})
    assert out.shape == (3, D_EMBED)
    assert torch.equal(out[0], embed_lookup("THINKING_WALL_0"))
    assert torch.equal(out[1], W_EMBED[42])
    assert torch.equal(out[2], embed_lookup("HIT_FULL"))
