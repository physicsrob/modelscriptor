"""Deterministic noise simulation for PWL ops.

The sandbox separates two sources of error cleanly:

1. PWL approximation error — modeled by performing the actual linear
   (or bilinear) interpolation on the breakpoint grid. Affine functions
   are exact for free.
2. Other transformer-runtime noise (FP32 accumulation, GPU matmul
   nondeterminism, etc.) — modeled by adding `N(0, sigma)` gaussian
   noise where `sigma = NOISE_REL * |value|`.

Determinism: the seed is derived from the PWL's stable construction-
order ID and the byte representation of the input array, via blake2b.
Same inputs to the same PWLDef produce identical outputs.
"""

from __future__ import annotations

import hashlib

import numpy as np


NOISE_REL: float = 1e-6


def add_noise(values: np.ndarray, pwl_id: int) -> np.ndarray:
    """Return `values` plus a per-element gaussian with sigma = NOISE_REL * |value|.

    Deterministic: identical (`values`, `pwl_id`) produces identical noise.
    Output shape matches `values`.
    """
    seed = _seed_for(pwl_id, values)
    rng = np.random.default_rng(seed)
    scale = NOISE_REL * np.abs(values)
    # rng.normal accepts an array `scale`; produces noise with per-element std.
    noise = rng.normal(loc=0.0, scale=scale)
    return values + noise


def _seed_for(pwl_id: int, values: np.ndarray) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(int(pwl_id).to_bytes(8, "little", signed=False))
    h.update(np.ascontiguousarray(values).tobytes())
    return int.from_bytes(h.digest(), "little")
