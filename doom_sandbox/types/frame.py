"""Frame — the complete pixel buffer output of one render pass.

Intentionally a plain dataclass rather than a pydantic model. Frames
are computed by the renderer (or the reference function) and compared
in tests — they don't need JSON round-trip, so the pydantic surface
would be dead weight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    """The screen image produced by the renderer for a single game tick.

    Distinct from `Texture` (input scene asset) despite the same
    pixel-grid shape. A Frame is the *output* of a render pass — what
    the host would blit to the screen after all `RENDER` tokens have
    emitted their chunks.

    Attributes
    ----------
    height : int
        Pixel height of the frame.
    width : int
        Pixel width of the frame.
    pixels : list[list[tuple[float, float, float]]]
        RGB pixel data organized as `pixels[y][x] = (r, g, b)`, all
        values in `[0, 1]`. The list-of-list shape is mutable in
        practice (the dataclass is frozen at the field level, not at
        the contained-list level), so `extract_frame`-style consumers
        can build a Frame by allocating the grid once and filling it
        with pixel emissions.
    """

    height: int
    width: int
    pixels: list[list[tuple[float, float, float]]]

    def __post_init__(self) -> None:
        if len(self.pixels) != self.height:
            raise ValueError(
                f"pixels has {len(self.pixels)} rows but height={self.height}"
            )
        for y, row in enumerate(self.pixels):
            if len(row) != self.width:
                raise ValueError(
                    f"pixels[{y}] has {len(row)} columns but width={self.width}"
                )
