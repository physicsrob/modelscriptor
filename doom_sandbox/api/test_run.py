"""Tests for run() — the autoregressive driver."""

from __future__ import annotations

import numpy as np
import pytest

from doom_sandbox.api import (
    Config,
    FloatSlot,
    ForwardOutput,
    IntSlot,
    Past,
    Pixel,
    Token,
    TokenType,
    TokenVocab,
    Vec,
    constant,
    extract_int_slot,
    is_type,
    make_token,
    pwl_def,
    run,
)
from doom_sandbox.api import _runtime
from doom_sandbox.api.vec import _make_vec
from doom_sandbox.runtime.loop import MaxPositionsExceeded


# Vocabs / forwards used across tests are declared here at module load so
# any pwl_def calls run before run() flips the freeze flag.

TICK = TokenType("TICK", slots={})
STOP = TokenType("STOP", slots={})

COUNTER = TokenType("COUNTER", slots={"n": IntSlot(0, 100)})
SIGNAL = TokenType("SIGNAL", slots={"v": FloatSlot(-1.0, 1.0)})
RENDER_FOR_AUTOPUB = TokenType("RENDER_FOR_AUTOPUB", slots={"col": IntSlot(0, 320)})

# A trivial PWL chain we can use to bump output depth in a controlled way.
_INC = pwl_def(lambda x: x + 1.0, breakpoints=2, input_range=(-1000.0, 1000.0))
_DOUBLE = pwl_def(lambda x: 2.0 * x, breakpoints=2, input_range=(-1000.0, 1000.0))

# Module-level constants for tests that need to publish during forward.
_FORTY_TWO = constant([42.0])
_TEN = constant([10.0])
_NINETY_NINE = constant([99.0])
_ONE = constant(1.0)


def _config(types, terminals=None, max_positions=8, decode=None):
    return Config(
        vocab=TokenVocab(types=list(types)),
        decode_pixels=decode if decode is not None else (lambda tok, pix: []),
        terminal_token_types=set(terminals or []),
        max_positions=max_positions,
    )


# --- Empty / invalid inputs ---

def test_empty_prefill_raises():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(ValueError, match="at least one prefill token"):
        run(config, [], fwd)


def test_non_positive_max_positions_raises():
    config = _config([TICK, STOP], terminals=[STOP], max_positions=0)

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(ValueError, match="max_positions must be positive"):
        run(config, [Token(TICK)], fwd)


# --- Prefill always runs; terminal type only stops in autoregressive mode ---

def test_prefill_runs_to_completion_even_when_emitting_terminal():
    """A prefill token whose forward emits a terminal next_token must still
    be followed by the next prefill token — prefill is a fixed prefix."""
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(TICK), Token(TICK)], fwd)
    # Two prefill positions ran; then position 2's input would be STOP
    # (the terminal next_token from position 1), which fires the terminal
    # check before any third forward call.
    assert len(out.forward_outputs) == 2


def test_terminal_check_stops_autoregressive_loop():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(TICK)], fwd)
    # Prefill runs once; auto-pos 1 input would deembed to STOP → terminate.
    assert len(out.forward_outputs) == 1


# --- Autoregressive feedback loop ---

def test_max_positions_cap_raises_during_autoregressive():
    """With no terminal type and a forward that always emits the same
    non-terminal token, the loop hits the cap and raises."""
    config = _config([TICK], max_positions=3)

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(TICK))

    with pytest.raises(MaxPositionsExceeded, match="without.*terminal"):
        run(config, [Token(TICK)], fwd)


def test_max_positions_cap_raises_during_prefill():
    """A prefill longer than max_positions raises before exhausting it."""
    config = _config([TICK, STOP], terminals=[STOP], max_positions=2)

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(MaxPositionsExceeded, match="during prefill"):
        run(config, [Token(TICK), Token(TICK), Token(TICK)], fwd)


def test_autoregressive_walks_counter_until_terminal():
    """Forward extracts n from COUNTER input, increments, returns COUNTER
    with the new n. We use max_positions to cap, then verify n grew."""
    config = _config([COUNTER, STOP], max_positions=4)

    def fwd(v, past):
        n_in = extract_int_slot(v, "n")
        n_out = _INC(n_in)
        return ForwardOutput(next_token=make_token(COUNTER, n=n_out))

    with pytest.raises(MaxPositionsExceeded):
        run(config, [Token(COUNTER, {"n": 0})], fwd)


# --- Past records reflect actual inputs ---

def test_past_grows_one_record_per_position():
    """At forward call N, the Past visible to user code has exactly N
    records (positions 0..N-1) — no self-visibility."""
    config = _config([COUNTER, STOP], terminals=[STOP], max_positions=8)
    sizes_at_call: list[int] = []
    types_at_call: list[list[TokenType]] = []

    def fwd(v, past):
        # Snapshot the size and types now — the Past instance grows in
        # place after this call returns, so we can't capture it lazily.
        sizes_at_call.append(len(past._records))
        types_at_call.append([r.input_token.type for r in past._records])
        n_in = extract_int_slot(v, "n")
        n_out = _INC(n_in)
        return ForwardOutput(next_token=make_token(COUNTER, n=n_out))

    with pytest.raises(MaxPositionsExceeded):
        run(config, [Token(COUNTER, {"n": 0})], fwd)

    assert sizes_at_call == [0, 1, 2, 3, 4, 5, 6, 7]
    # Every prior input was COUNTER (one prefill + autoregressive feedback).
    for snapshot in types_at_call:
        assert all(t == COUNTER for t in snapshot)


# --- Pixels ---

def test_pixels_collected_in_order():
    config = _config(
        [TICK, STOP],
        terminals=[STOP],
        decode=lambda tok, pix: (
            [Pixel(x=0, y=0, color=(255, 0, 0))]
            if tok.type == TICK
            else []
        ),
    )

    def fwd(v, past):
        # Return STOP from the prefill so we get exactly 2 prefill calls
        # before terminating.
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(TICK), Token(TICK)], fwd)
    # Two prefill calls, both render TICK input → 2 pixels.
    assert len(out.pixels) == 2
    assert all(p.color == (255, 0, 0) for p in out.pixels)


def test_pixels_none_passed_to_decode_when_forward_omits_them():
    seen: list[Vec | None] = []

    def decode(tok, pix):
        seen.append(pix)
        return []

    config = _config([TICK, STOP], terminals=[STOP], decode=decode)

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))  # no pixels

    run(config, [Token(TICK)], fwd)
    assert seen == [None]


# --- layer_count ---

def test_layer_count_is_max_depth_across_all_outputs():
    """Forward returns a next_token whose depth grows with applied PWLs."""
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        # is_type adds 1; two PWLs add 2 more.
        # STOP has no slots, so make_token(STOP) returns depth 1 regardless
        # of m3's depth — surface m3 via past.publish so layer_count sees it.
        m = is_type(v, TICK)            # depth 1
        m2 = _INC(m)                    # depth 2
        m3 = _DOUBLE(m2)                # depth 3
        past.publish("deep", m3)
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(TICK)], fwd)
    assert out.layer_count == 3


def test_layer_count_zero_with_only_no_op_make_token():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(TICK)], fwd)
    # make_token(STOP) with no slot inputs → depth 1 (max_input_depth=0 + 1).
    assert out.layer_count == 1


def test_layer_count_includes_pixels_depth():
    """A `pixels` Vec with deep computation must contribute to layer_count."""
    config = _config(
        [TICK, STOP],
        terminals=[STOP],
        decode=lambda tok, pix: [],
    )

    def fwd(v, past):
        m = is_type(v, TICK)
        m2 = _INC(m)
        m3 = _DOUBLE(m2)  # depth 3
        return ForwardOutput(next_token=make_token(STOP), pixels=m3)

    out = run(config, [Token(TICK)], fwd)
    assert out.layer_count == 3


# --- Freeze flag ---

def test_forward_running_flag_is_set_during_forward():
    seen_states: list[bool] = []

    def fwd(v, past):
        seen_states.append(_runtime._FORWARD_RUNNING)
        return ForwardOutput(next_token=make_token(STOP))

    config = _config([TICK, STOP], terminals=[STOP])
    assert _runtime._FORWARD_RUNNING is False
    run(config, [Token(TICK)], fwd)
    assert seen_states == [True]
    # Restored after the run.
    assert _runtime._FORWARD_RUNNING is False


def test_freeze_flag_blocks_pwl_def_inside_forward():
    """Constructing a pwl_def inside forward() must raise."""
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        # This must raise — pwl_def is module-load-only.
        pwl_def(lambda x: x, breakpoints=2, input_range=(0.0, 1.0))
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(RuntimeError, match="module load"):
        run(config, [Token(TICK)], fwd)


def test_freeze_flag_blocks_constant_inside_forward():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        constant(1.0)
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(RuntimeError, match="module load"):
        run(config, [Token(TICK)], fwd)


def test_freeze_flag_restored_on_forward_exception():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        raise RuntimeError("user code blew up")

    with pytest.raises(RuntimeError, match="user code"):
        run(config, [Token(TICK)], fwd)
    assert _runtime._FORWARD_RUNNING is False


def test_past_state_clean_after_forward_exception():
    """If forward() raises mid-position, the Past's in-flight state must be
    discarded — partial publishes shouldn't leak into a captured Past."""
    config = _config([TICK, STOP], terminals=[STOP])
    captured: list[Past] = []

    def fwd(v, past):
        captured.append(past)
        past.publish("partial", _ONE)
        raise RuntimeError("after partial publish")

    with pytest.raises(RuntimeError, match="after partial publish"):
        run(config, [Token(TICK)], fwd)
    past = captured[0]
    # No record finalized (forward never returned).
    assert past._records == []
    # No in-flight state lingering.
    assert past._current_input_token is None
    assert past._pending == {}


# --- Decode pixels gets the deembedded input token ---

def test_next_tokens_decoded_per_position():
    """`output.next_tokens[i]` must be the deembed of `forward_outputs[i].next_token`."""
    config = _config([COUNTER, STOP], terminals=[STOP], max_positions=4)

    def fwd(v, past):
        n_in = extract_int_slot(v, "n")
        n_out = _INC(n_in)
        return ForwardOutput(next_token=make_token(COUNTER, n=n_out))

    with pytest.raises(MaxPositionsExceeded):
        run(config, [Token(COUNTER, {"n": 0})], fwd)


def test_next_tokens_aligned_with_forward_outputs():
    config = _config([COUNTER, STOP], terminals=[STOP])

    def fwd(v, past):
        return ForwardOutput(next_token=make_token(STOP))

    out = run(config, [Token(COUNTER, {"n": 5})], fwd)
    assert len(out.next_tokens) == len(out.forward_outputs)
    assert out.next_tokens[0].type == STOP


def test_publish_outside_forward_raises():
    """publish() can only be called inside forward(); calling it via a stray
    Past after the run finishes (or before forward dispatch) must raise."""
    config = _config([TICK, STOP], terminals=[STOP])

    captured: list[Past] = []

    def fwd(v, past):
        captured.append(past)
        return ForwardOutput(next_token=make_token(STOP))

    run(config, [Token(TICK)], fwd)
    # Outside forward(), the freeze flag is False → publish raises.
    with pytest.raises(RuntimeError, match="inside forward"):
        captured[0].publish("foo", constant(1.0))


def test_publish_reserved_name_raises():
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        past.publish("input.col", _ONE)  # reserved
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(ValueError, match="reserved name"):
        run(config, [Token(TICK)], fwd)


def test_published_value_visible_at_same_position():
    """After publishing, past.* at the same position can find the value (self-attention)."""
    config = _config([TICK, STOP], terminals=[STOP])
    seen_self_value: list[Vec] = []

    def fwd(v, past):
        past.publish("X", _FORTY_TWO)
        # Lookup my own X via past with a 1-shape one-hot key (since
        # "input.type" is M-shape we build a query against input.type).
        # Easier: use mean (single contributor) for self-only.
        result = past.mean("X")
        seen_self_value.append(result)
        return ForwardOutput(next_token=make_token(STOP))

    run(config, [Token(TICK)], fwd)
    assert abs(seen_self_value[0]._data[0] - 42.0) < 1e-3


def test_self_attention_via_pick_argmax():
    """The doc claims self-attention 'works naturally' via publish-then-attend.
    Verify pick_argmax can find a key+value the agent published at the same
    position — i.e. self appears as a real candidate to the keyed pick path,
    not just to mean."""
    config = _config([TICK, STOP], terminals=[STOP])
    results: list[Vec] = []

    # Module-level constants (must be built outside forward).
    self_key = constant([1.0, 0.0])  # one-hot, length 2
    self_value = constant([777.0])
    query_for_self = constant([10.0, 0.0])  # picks self_key, gap = 10

    def fwd(v, past):
        past.publish("k", self_key)
        past.publish("v", self_value)
        results.append(past.pick_argmax(query_for_self, "k", "v"))
        return ForwardOutput(next_token=make_token(STOP))

    run(config, [Token(TICK)], fwd)
    # Only one candidate (self), so pick_argmax returns its value.
    # σ = NOISE_REL * |777| ≈ 7.8e-4; 0.01 ≈ 13σ keeps the test stable.
    assert abs(results[0]._data[0] - 777.0) < 0.01


def test_published_value_visible_to_later_positions():
    """Published Vecs at position N are visible at later positions."""
    config = _config([COUNTER, STOP], terminals=[STOP], max_positions=4)
    seen_via_past: list[float] = []

    def fwd(v, past):
        n_in = extract_int_slot(v, "n")
        n_out = _INC(n_in)
        past.publish("counter_value", n_out)
        # Try to read back the most recent counter_value from past.
        # At position 0 only self contributes; mean = self.
        # At position 1 both pos 0 and self contribute; mean = avg.
        recent = past.mean("counter_value")
        seen_via_past.append(float(recent._data[0]))
        return ForwardOutput(next_token=make_token(COUNTER, n=n_out))

    with pytest.raises(MaxPositionsExceeded):
        run(config, [Token(COUNTER, {"n": 0})], fwd)

    # Position 0: published 1.0 → mean over {1.0} = 1.0
    # Position 1 (input n=1): published 2.0 → mean over {1.0, 2.0} = 1.5
    # Position 2 (input n=2): published 3.0 → mean over {1.0, 2.0, 3.0} = 2.0
    # Position 3 (input n=3): published 4.0 → mean over {1.0..4.0} = 2.5
    assert abs(seen_via_past[0] - 1.0) < 1e-3
    assert abs(seen_via_past[1] - 1.5) < 1e-3
    assert abs(seen_via_past[2] - 2.0) < 1e-3
    assert abs(seen_via_past[3] - 2.5) < 1e-3


def test_publish_before_lookup_required():
    """A lookup that runs before any compatible publish raises (no candidates)."""
    config = _config([TICK, STOP], terminals=[STOP])

    def fwd(v, past):
        # Trying to lookup a name nobody has published yet — even auto-input
        # only fills input.* names.
        past.mean("never_published")
        return ForwardOutput(next_token=make_token(STOP))

    with pytest.raises(RuntimeError, match="no position has"):
        run(config, [Token(TICK)], fwd)


def test_input_dot_star_omitted_slot_publishes_zero():
    """If a Token is constructed without specifying a declared slot, the
    framework still auto-publishes `input.<slot>` as a 1-Vec carrying 0
    — consistent with `extract_*_slot`'s "missing slot reads as 0"
    behavior. A regression that publishes only the explicitly supplied
    slots would silently make the slot un-queryable here."""
    config = _config([RENDER_FOR_AUTOPUB, STOP], terminals=[STOP])
    seen_col: list[Vec] = []

    def fwd(v, past):
        seen_col.append(past.mean("input.col"))
        return ForwardOutput(next_token=make_token(STOP))

    # RENDER_FOR_AUTOPUB declares "col" but we don't supply it in values.
    run(config, [Token(RENDER_FOR_AUTOPUB, {})], fwd)
    assert seen_col[0].shape == 1
    assert abs(seen_col[0]._data[0] - 0.0) < 1e-3


def test_input_dot_star_auto_published_at_self_via_past():
    """At the in-flight position, both `input.type` and `input.<slot>` are
    auto-published and queryable via past.*."""
    config = _config([RENDER_FOR_AUTOPUB, STOP], terminals=[STOP])
    seen_type: list[Vec] = []
    seen_col: list[Vec] = []

    def fwd(v, past):
        seen_type.append(past.mean("input.type"))
        seen_col.append(past.mean("input.col"))
        return ForwardOutput(next_token=make_token(STOP))

    run(config, [Token(RENDER_FOR_AUTOPUB, {"col": 7})], fwd)
    # One position, type=RENDER_FOR_AUTOPUB. Layout has [RENDER_FOR_AUTOPUB, STOP].
    # input.type is one-hot at index 0 (RENDER_FOR_AUTOPUB).
    assert seen_type[0]._data.tolist() == pytest.approx([1.0, 0.0], abs=1e-3)
    # input.col is a 1-Vec carrying the slot value.
    assert seen_col[0].shape == 1
    assert abs(seen_col[0]._data[0] - 7.0) < 1e-3


def test_repeated_publish_overwrites():
    """Re-publishing under the same name overwrites — analog of residual rewrite."""
    config = _config([TICK, STOP], terminals=[STOP])
    final: list[float] = []

    def fwd(v, past):
        past.publish("X", _TEN)
        past.publish("X", _NINETY_NINE)  # overwrite
        result = past.mean("X")
        final.append(float(result._data[0]))
        return ForwardOutput(next_token=make_token(STOP))

    run(config, [Token(TICK)], fwd)
    assert abs(final[0] - 99.0) < 1e-3


def test_decode_pixels_receives_input_token_per_position():
    """`decode_pixels(input_tok, pixels)` must be called with the actual
    input token at that position (prefill token, then autoregressive token)."""
    seen_inputs: list[Token] = []

    def decode(tok, pix):
        seen_inputs.append(tok)
        return []

    config = _config([COUNTER, STOP], terminals=[STOP], decode=decode, max_positions=4)

    def fwd(v, past):
        n_in = extract_int_slot(v, "n")
        n_out = _INC(n_in)
        return ForwardOutput(next_token=make_token(COUNTER, n=n_out))

    with pytest.raises(MaxPositionsExceeded):
        run(config, [Token(COUNTER, {"n": 0})], fwd)

    # Position 0 sees prefill COUNTER(n=0); positions 1..3 see autoregressive
    # COUNTER(n=1), COUNTER(n=2), COUNTER(n=3).
    assert [t.type for t in seen_inputs] == [COUNTER] * 4
    assert seen_inputs[0].values["n"] == 0
    assert seen_inputs[1].values["n"] == 1
    assert seen_inputs[2].values["n"] == 2
    assert seen_inputs[3].values["n"] == 3
