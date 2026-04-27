"""Sandbox-side data types — schemas mirroring the parent project's data shapes.

These pydantic models are the bridge between project-side fixture
generation (which serializes to JSON) and sandbox-side fixture loading
(which deserializes through these classes). The schemas use
`extra="ignore"` so unknown fields produced by the project (e.g.,
texture pixel arrays, name → id maps) don't break loading; phase 1
doesn't need them.
"""

from .game_state import GameState
from .map_subset import BSPNode, MapSubset, Segment, Texture

__all__ = [
    "MapSubset",
    "Segment",
    "BSPNode",
    "Texture",
    "GameState",
]
