"""The autoregressive `run()` driver.

Manages the layout activation, the freeze flag, the prefill →
autoregressive transition, terminal-type detection, the max_positions
cap, pixel decoding, and the layer_count tally.

The agent-facing `run()` in `api/forward.py` is a thin wrapper around
`run_loop` here.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from ..api import _runtime
from ..api.forward import Config, ForwardOutput, Pixel, RunOutput
from ..api.past import Past
from ..api.tokens import Token
from ..api.vec import Vec
from .embedding import Layout, active_layout, deembed, embed


class MaxPositionsExceeded(RuntimeError):
    """Raised when a `run()` call hits `config.max_positions` without terminating."""


@contextmanager
def _forward_session() -> Iterator[None]:
    """While the body runs, `pwl_def` / `constant` raise on construction."""
    prev = _runtime._FORWARD_RUNNING
    _runtime._FORWARD_RUNNING = True
    try:
        yield
    finally:
        _runtime._FORWARD_RUNNING = prev


def run_loop(
    config: Config,
    prefill_tokens: list[Token],
    forward: Callable[[Vec, Past], ForwardOutput],
) -> RunOutput:
    """Drive the autoregressive loop. See `api.forward.run` for the public contract."""
    if not prefill_tokens:
        raise ValueError(
            "run() requires at least one prefill token to bootstrap "
            "the autoregressive loop"
        )
    if config.max_positions <= 0:
        raise ValueError(
            f"max_positions must be positive, got {config.max_positions}"
        )

    layout = Layout(list(config.vocab.types))
    past = Past(layout)
    forward_outputs: list[ForwardOutput] = []
    next_tokens: list[Token] = []
    pixels: list[Pixel] = []

    with active_layout(layout), _forward_session():
        # Prefill: every prefill token is fed in regardless of what the
        # model would have predicted. We do not check terminal types
        # during prefill — prefill is a fixed prefix.
        for prefill_tok in prefill_tokens:
            if len(forward_outputs) >= config.max_positions:
                raise MaxPositionsExceeded(
                    f"hit max_positions={config.max_positions} during prefill"
                )
            _step(
                forward, layout, past, prefill_tok, config,
                forward_outputs, next_tokens, pixels,
            )

        # Autoregressive: the previous position's `next_token` becomes
        # the next position's input. Stop on terminal types.
        while True:
            next_input_tok = next_tokens[-1]

            if next_input_tok.type in config.terminal_token_types:
                break

            if len(forward_outputs) >= config.max_positions:
                raise MaxPositionsExceeded(
                    f"hit max_positions={config.max_positions} without "
                    f"emitting a terminal token"
                )

            _step(
                forward, layout, past, next_input_tok, config,
                forward_outputs, next_tokens, pixels,
            )

    layer_count = _compute_layer_count(forward_outputs, past)
    return RunOutput(
        forward_outputs=forward_outputs,
        next_tokens=next_tokens,
        pixels=pixels,
        layer_count=layer_count,
    )


def _step(
    forward: Callable[[Vec, Past], ForwardOutput],
    layout: Layout,
    past: Past,
    input_tok: Token,
    config: Config,
    forward_outputs: list[ForwardOutput],
    next_tokens: list[Token],
    pixels: list[Pixel],
) -> None:
    """Embed, open a position, run forward, finalize, decode pixels."""
    input_vec = embed(input_tok, layout)
    past._begin_position(input_tok)
    try:
        fwd = forward(input_vec, past)
    except BaseException:
        # Forward blew up — discard the half-baked in-flight position so
        # the Past doesn't carry partial publishes into any later context
        # (e.g. a test that captured the Past, or a caller that catches
        # and inspects).
        past._abort_position()
        raise
    past._end_position()
    forward_outputs.append(fwd)
    next_tokens.append(deembed(fwd.next_token, layout))
    pixels.extend(config.decode_pixels(input_tok, fwd.pixels))


def _compute_layer_count(
    forward_outputs: list[ForwardOutput], past: Past
) -> int:
    """Max `.depth` across every Vec the user returned or published."""
    layer_count = 0
    for fwd in forward_outputs:
        layer_count = max(layer_count, fwd.next_token.depth)
        if fwd.pixels is not None:
            layer_count = max(layer_count, fwd.pixels.depth)
    for record in past._records:
        for v in record.exports.values():
            layer_count = max(layer_count, v.depth)
    return layer_count
