"""Free utilities — concat and split. No compute, no extra depth."""

from __future__ import annotations

import numpy as np

from .vec import Vec, _make_vec


def concat(*vecs: Vec) -> Vec:
    """Pack Vecs end-to-end into a single Vec.

    The result has `shape = sum(v.shape for v in vecs)` and
    `depth = max(v.depth for v in vecs)`. Free — no compute, adds
    no depth beyond the deepest input.

    Raises if called with zero Vecs.
    """
    if not vecs:
        raise ValueError("concat requires at least one Vec")
    data = np.concatenate([v._data for v in vecs])
    depth = max(v.depth for v in vecs)
    return _make_vec(data, depth=depth)


def split(vec: Vec, sizes: list[int]) -> list[Vec]:
    """Split a Vec into pieces of the given sizes.

    Each returned piece carries the parent Vec's depth. Free — no
    compute, adds no depth.

    Raises if `sum(sizes) != vec.shape` or any size is non-positive.
    """
    if any(s <= 0 for s in sizes):
        raise ValueError(f"split sizes must be positive, got {sizes}")
    total = sum(sizes)
    if total != vec.shape:
        raise ValueError(
            f"split sizes {sizes} sum to {total}, "
            f"but vec has shape {vec.shape}"
        )
    pieces: list[Vec] = []
    start = 0
    for size in sizes:
        pieces.append(_make_vec(vec._data[start : start + size], depth=vec.depth))
        start += size
    return pieces
