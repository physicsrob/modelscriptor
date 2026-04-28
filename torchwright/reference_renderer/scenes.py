from typing import List, Tuple

import numpy as np

from torchwright.reference_renderer.types import Segment

# DOOM convention: floor at z=0, ceiling at z=128 = a "standard" ceiling
# for a normal-sized room.  Player eye sits 41 units above the floor.
_BOX_FLOOR = 0.0
_BOX_CEILING = 128.0


def box_room(size: float = 256.0) -> List[Segment]:
    """A square one-sector room centred at the origin.

    DOOM-shaped: floor at z=0, ceiling at z=128, walls 128 units tall.
    Default ``size=256`` gives a 256×256 floor plan with the four walls
    at ±128.  Each segment carries the front sector's floor / ceiling
    so the renderer projects them on the same code path as WAD-loaded
    segs.

    DOOM convention: a sector's boundary linedefs are wound so that
    FRONT (= right of the a→b direction) points into the sector.  For
    a closed room that's *clockwise* traversal viewed from above
    (with +y north): east wall going south, south going west, west
    going north, north going east.

    Each wall has a distinct color (one-sided seg, no texture):
        +x (east)  = red
        -x (west)  = green
        +y (north) = blue
        -y (south) = yellow
    """
    h = size / 2.0
    common = dict(front_floor=_BOX_FLOOR, front_ceiling=_BOX_CEILING)
    return [
        # East wall going south (right side faces west = inside)
        Segment(ax=h, ay=h, bx=h, by=-h, color=(1.0, 0.0, 0.0), **common),
        # South wall going west (right side faces north = inside)
        Segment(ax=h, ay=-h, bx=-h, by=-h, color=(1.0, 1.0, 0.0), **common),
        # West wall going north (right side faces east = inside)
        Segment(ax=-h, ay=-h, bx=-h, by=h, color=(0.0, 1.0, 0.0), **common),
        # North wall going east (right side faces south = inside)
        Segment(ax=-h, ay=h, bx=h, by=h, color=(0.0, 0.0, 1.0), **common),
    ]


def box_room_textured(
    size: float = 256.0,
    wad_path: str | None = None,
    tex_size: int = 1024,
) -> Tuple[List[Segment], List[np.ndarray]]:
    """Box room with textured walls — sector-aware, DOOM-shaped.

    Same geometry and orientation as :func:`box_room`, but each wall
    references a texture index.  When *wad_path* is provided, loads
    real DOOM textures (STARTAN3, STARG3, BROWN1, BROWNGRN) at native
    resolution (capped at *tex_size* per axis).  Otherwise falls back
    to the built-in procedural texture atlas.
    """
    h = size / 2.0

    if wad_path is not None:
        from torchwright.doom.wad import WADReader
        from torchwright.reference_renderer.textures import downscale_texture

        wad = WADReader(wad_path)
        names = ["STARTAN3", "STARG3", "BROWN1", "BROWNGRN"]
        raw = [wad.get_texture(n) for n in names]
        assert all(t is not None for t in raw), "WAD texture not found"
        textures = [
            downscale_texture(t, tex_size, tex_size) for t in raw if t is not None
        ]
    else:
        from torchwright.reference_renderer.textures import default_texture_atlas

        textures = default_texture_atlas()

    common = dict(front_floor=_BOX_FLOOR, front_ceiling=_BOX_CEILING)
    segments = [
        # Same clockwise winding as box_room.
        Segment(ax=h, ay=h, bx=h, by=-h, color=(1, 0, 0), texture_id=0, **common),
        Segment(ax=h, ay=-h, bx=-h, by=-h, color=(1, 1, 0), texture_id=3, **common),
        Segment(ax=-h, ay=-h, bx=-h, by=h, color=(0, 1, 0), texture_id=1, **common),
        Segment(ax=-h, ay=h, bx=h, by=h, color=(0, 0, 1), texture_id=2, **common),
    ]
    return segments, textures
