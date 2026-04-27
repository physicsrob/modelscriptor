"""Per-position forward contract: ForwardOutput, Pixel, Config, run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .past import Past
from .tokens import Token, TokenType
from .vec import Vec


@dataclass
class ForwardOutput:
    """Returned by your `forward(input_vec, past)` at every position.

    `next_token` is required and is always deembedded — at autoregressive
    positions it becomes the input for position N+1; at prefill positions
    it is computed but not used as feedback.

    `pixels` is optional. If supplied, the framework calls your phase's
    `decode_pixels(input_tok, pixels)` to extract `Pixel`s for the frame.
    Pixels are blitted, never re-embedded, never visible to future
    positions.

    There is no `exports` field. To make a Vec queryable via `past.*`,
    call `past.publish(name, vec)` during your `forward()`.
    """

    next_token: Vec
    pixels: Vec | None = None


@dataclass(frozen=True)
class Pixel:
    """A pixel emission decoded from a render-emitting position."""

    x: int
    y: int
    color: tuple[int, int, int]   # RGB


@dataclass
class TokenVocab:
    """The set of `TokenType`s used by a phase."""

    types: list[TokenType]


@dataclass
class Config:
    """Returned by your `setup()`. Declares vocab, lifecycle hooks, and limits.

    Attributes
    ----------
    vocab : TokenVocab
        The full set of token types this phase uses.
    decode_pixels : Callable[[Token, Vec | None], list[Pixel]]
        Called at every position. Returns `[]` for non-render positions
        and a list of `Pixel`s for render positions.
    terminal_token_types : set[TokenType]
        Token types that, when emitted as `next_token` and deembedded,
        terminate the autoregressive loop.
    max_positions : int
        Hard safety cap on the autoregressive loop. Default 8192.
    """

    vocab: TokenVocab
    decode_pixels: Callable[[Token, Vec | None], list[Pixel]] = (
        lambda input_tok, pixels: []
    )
    terminal_token_types: set[TokenType] = field(default_factory=set)
    max_positions: int = 8192


@dataclass
class RunOutput:
    """Result of a `run(...)` invocation.

    `forward_outputs` is the per-position output your `forward()`
    returned. `next_tokens` is the framework's deembed of each
    position's `next_token` Vec — `next_tokens[i]` is the discrete
    `Token` decoded from `forward_outputs[i].next_token`. Phase code
    reads from here to extract emitted tokens (since `Vec.data` is
    framework-private). `pixels` is every `Pixel` emitted across all
    positions (in order). `layer_count` is the max `.depth` across
    every Vec your `forward()` returned (`next_token`, `pixels`) or
    passed to `past.publish` — and is the floor on how deep a
    transformer this would compile to.
    """

    forward_outputs: list[ForwardOutput]
    next_tokens: list[Token]
    pixels: list[Pixel]
    layer_count: int


def run(
    config: Config,
    prefill_tokens: list[Token],
    forward: Callable[[Vec, Past], ForwardOutput],
) -> RunOutput:
    """Run the autoregressive loop.

    Embeds each prefill token, calls `forward(input_vec, past)` at every
    position, deembeds the returned `next_token`, decodes pixels via
    `config.decode_pixels`, and continues until either a terminal token
    fires or `config.max_positions` is reached. Raises
    `MaxPositionsExceeded` (a `RuntimeError` subclass) on overrun and
    `ValueError` on empty prefill or non-positive `max_positions`.
    """
    # Deferred to avoid a cycle: runtime/loop.py imports Config et al
    # from this module.
    from ..runtime.loop import run_loop

    return run_loop(config, prefill_tokens, forward)
