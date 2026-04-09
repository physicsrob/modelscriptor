"""Texture definitions and atlas utilities for the DOOM renderer.

Textures are small 2-D arrays of RGB floats stored as
``(tex_width, tex_height, 3)`` numpy arrays.  Column-major layout:
``texture[col, row]`` gives the RGB triple at that texel.
"""

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Built-in 8x8 textures
# ---------------------------------------------------------------------------

TEX_WIDTH = 8
TEX_HEIGHT = 8


def _make_texture(rows: List[List[Tuple[float, float, float]]]) -> np.ndarray:
    """Build a (W, H, 3) texture from row-major RGB data."""
    arr = np.array(rows, dtype=np.float64)  # (H, W, 3)
    return arr.transpose(1, 0, 2)  # (W, H, 3) — column-major


def texture_brick() -> np.ndarray:
    """8x8 brick pattern — red/dark-red rows with offset mortar."""
    R = (0.7, 0.2, 0.1)
    D = (0.5, 0.15, 0.08)
    M = (0.4, 0.4, 0.35)  # mortar
    return _make_texture([
        [R, R, R, M, D, D, D, M],
        [R, R, R, M, D, D, D, M],
        [R, R, R, M, D, D, D, M],
        [M, M, M, M, M, M, M, M],
        [D, D, M, R, R, R, M, D],
        [D, D, M, R, R, R, M, D],
        [D, D, M, R, R, R, M, D],
        [M, M, M, M, M, M, M, M],
    ])


def texture_stone() -> np.ndarray:
    """8x8 stone wall — gray blocks with dark gaps."""
    L = (0.6, 0.6, 0.55)
    G = (0.45, 0.45, 0.4)
    D = (0.3, 0.3, 0.28)
    return _make_texture([
        [L, L, L, D, G, G, G, G],
        [L, L, L, D, G, G, G, G],
        [L, L, L, D, G, G, G, G],
        [D, D, D, D, D, D, D, D],
        [G, G, D, L, L, L, L, D],
        [G, G, D, L, L, L, L, D],
        [G, G, D, L, L, L, L, D],
        [D, D, D, D, D, D, D, D],
    ])


def texture_stripe() -> np.ndarray:
    """8x8 vertical stripes — alternating blue and dark blue."""
    B = (0.2, 0.3, 0.7)
    D = (0.1, 0.15, 0.4)
    row = [B, B, D, D, B, B, D, D]
    return _make_texture([row] * 8)


def texture_checker() -> np.ndarray:
    """8x8 checkerboard — white and dark gray 2x2 blocks."""
    W = (0.8, 0.8, 0.8)
    D = (0.3, 0.3, 0.3)
    return _make_texture([
        [W, W, D, D, W, W, D, D],
        [W, W, D, D, W, W, D, D],
        [D, D, W, W, D, D, W, W],
        [D, D, W, W, D, D, W, W],
        [W, W, D, D, W, W, D, D],
        [W, W, D, D, W, W, D, D],
        [D, D, W, W, D, D, W, W],
        [D, D, W, W, D, D, W, W],
    ])


DEFAULT_TEXTURES = [texture_brick, texture_stone, texture_stripe, texture_checker]


def default_texture_atlas() -> List[np.ndarray]:
    """Return the four built-in 8x8 textures as a list.

    Index 0 = brick, 1 = stone, 2 = stripe, 3 = checker.
    """
    return [fn() for fn in DEFAULT_TEXTURES]


# ---------------------------------------------------------------------------
# Atlas utilities
# ---------------------------------------------------------------------------


def solid_color_texture(color: Tuple[float, float, float]) -> np.ndarray:
    """Create a 1x1 texture from a solid color."""
    tex = np.empty((1, 1, 3), dtype=np.float64)
    tex[0, 0] = color
    return tex
