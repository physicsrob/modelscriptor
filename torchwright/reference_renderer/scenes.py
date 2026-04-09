from typing import List, Tuple

import numpy as np

from torchwright.reference_renderer.types import Segment


def box_room(size: float = 10.0) -> List[Segment]:
    """A square room centered at the origin with four colored walls.

    Each wall has a distinct color so orientation is visible:
        +x (east)  = red
        -x (west)  = green
        +y (north) = blue
        -y (south) = yellow

    Args:
        size: Side length of the room.

    Returns:
        List of 4 Segments forming the room walls.
    """
    h = size / 2.0
    return [
        Segment(ax=h, ay=-h, bx=h, by=h, color=(1.0, 0.0, 0.0)),      # east (red)
        Segment(ax=-h, ay=h, bx=-h, by=-h, color=(0.0, 1.0, 0.0)),    # west (green)
        Segment(ax=-h, ay=h, bx=h, by=h, color=(0.0, 0.0, 1.0)),      # north (blue)
        Segment(ax=h, ay=-h, bx=-h, by=-h, color=(1.0, 1.0, 0.0)),    # south (yellow)
    ]


def multi_room() -> List[Segment]:
    """Two rooms connected by a corridor, with diagonal walls.

    Layout (top-down, +x right, +y up)::

        Room A (left)         Corridor          Room B (right)
        (-12,-6)──────(-4,-6)         (4,-4)──────(12,-4)
        |              |     ╲       ╱      |              |
        |              |      (-2,-2)──(2,-2)      |              |
        |     ★A       |                    |       ★B     |
        |              |      (-2, 2)──(2, 2)      |              |
        |              |     ╱       ╲      |              |
        (-12, 6)──────(-4, 6)         (4, 4)──────(12, 4)

    ★A = player start for room A tests (e.g. -8, 0)
    ★B = player start for room B tests (e.g.  8, 0)

    22 segments total. Includes two diagonal walls in the corridor
    transitions. Each room/section has a distinct color palette.

    Returns:
        List of 22 Segments.
    """
    RED = (0.8, 0.2, 0.1)
    GREEN = (0.1, 0.7, 0.2)
    BLUE = (0.2, 0.3, 0.8)
    YELLOW = (0.8, 0.8, 0.1)
    CYAN = (0.1, 0.7, 0.7)
    MAGENTA = (0.7, 0.1, 0.6)
    ORANGE = (0.9, 0.5, 0.1)
    GRAY = (0.5, 0.5, 0.5)

    return [
        # Room A walls
        Segment(ax=-12, ay=-6, bx=-4, by=-6, color=RED),        # south
        Segment(ax=-12, ay=6, bx=-12, by=-6, color=GREEN),      # west
        Segment(ax=-4, ay=6, bx=-12, by=6, color=BLUE),         # north
        Segment(ax=-4, ay=-6, bx=-4, by=-2, color=YELLOW),      # east-south (doorway)
        Segment(ax=-4, ay=2, bx=-4, by=6, color=YELLOW),        # east-north (doorway)

        # Corridor south wall
        Segment(ax=-4, ay=-2, bx=-2, by=-2, color=GRAY),        # entry south
        Segment(ax=-2, ay=-2, bx=2, by=-2, color=GRAY),         # middle south
        Segment(ax=2, ay=-2, bx=4, by=-4, color=GRAY),          # exit south (diagonal!)

        # Corridor north wall
        Segment(ax=-4, ay=2, bx=-2, by=2, color=GRAY),          # entry north
        Segment(ax=-2, ay=2, bx=2, by=2, color=GRAY),           # middle north
        Segment(ax=2, ay=2, bx=4, by=4, color=GRAY),            # exit north (diagonal!)

        # Room B walls
        Segment(ax=4, ay=-4, bx=12, by=-4, color=CYAN),         # south
        Segment(ax=12, ay=-4, bx=12, by=4, color=MAGENTA),      # east
        Segment(ax=12, ay=4, bx=4, by=4, color=ORANGE),         # north
        Segment(ax=4, ay=4, bx=4, by=2, color=BLUE),            # west-north (doorway)
        Segment(ax=4, ay=-2, bx=4, by=-4, color=BLUE),          # west-south (doorway)

        # Extra interior walls for complexity
        Segment(ax=-10, ay=0, bx=-8, by=0, color=RED),          # shelf in room A
        Segment(ax=8, ay=-2, bx=10, by=0, color=CYAN),          # diagonal wall in room B!
        Segment(ax=8, ay=2, bx=10, by=0, color=ORANGE),         # diagonal wall in room B!

        # Pillars (short segments)
        Segment(ax=0, ay=-1.5, bx=0, by=-0.5, color=MAGENTA),  # corridor pillar south
        Segment(ax=0, ay=0.5, bx=0, by=1.5, color=MAGENTA),    # corridor pillar north
        Segment(ax=-7, ay=-3, bx=-7, by=-2, color=GREEN),       # pillar in room A
    ]


def box_room_textured(
    size: float = 10.0,
    wad_path: str = None,
    tex_size: int = 8,
) -> Tuple[List[Segment], List[np.ndarray]]:
    """Box room with textured walls.

    When *wad_path* is provided, loads real DOOM textures from the WAD
    and downscales them to *tex_size* x *tex_size*.  Otherwise falls
    back to the built-in procedural textures.

    Returns ``(segments, textures)`` where each wall uses a different
    texture: east=STARTAN3/brick, west=STARG3/stone, north=BROWN1/stripe,
    south=BROWNGRN/checker.
    """
    h = size / 2.0

    if wad_path is not None:
        from torchwright.doom.wad import WADReader
        from torchwright.reference_renderer.textures import downscale_texture

        wad = WADReader(wad_path)
        names = ["STARTAN3", "STARG3", "BROWN1", "BROWNGRN"]
        textures = [
            downscale_texture(wad.get_texture(n), tex_size, tex_size)
            for n in names
        ]
    else:
        from torchwright.reference_renderer.textures import default_texture_atlas
        textures = default_texture_atlas()

    segments = [
        Segment(ax=h, ay=-h, bx=h, by=h, color=(1, 0, 0), texture_id=0),
        Segment(ax=-h, ay=h, bx=-h, by=-h, color=(0, 1, 0), texture_id=1),
        Segment(ax=-h, ay=h, bx=h, by=h, color=(0, 0, 1), texture_id=2),
        Segment(ax=h, ay=-h, bx=-h, by=-h, color=(1, 1, 0), texture_id=3),
    ]
    return segments, textures
