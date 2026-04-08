from typing import List

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
