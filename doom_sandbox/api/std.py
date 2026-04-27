"""Stdlib of pre-built primitives.

These mirror specific operations from the parent project's
`torchwright/ops/`. Each named primitive has a documented approximation
strategy. Adding new primitives is a platform-side decision; ask before
introducing one.

The stdlib will grow as phases need things. The starter set covers
dispatch (`type_switch`), common 1D PWLs as factories returning
`PWLDef`, and common 2D ops as factories returning `PWLDef2D`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..runtime.noise import add_noise
from .pwl import PWLDef, PWLDef2D
from .vec import Vec, _make_vec


# Stable per-op ID for noise seeding (separate range from PWL/past IDs).
_ID_TYPE_SWITCH = 8_000_001


# --- Dispatch ---

def type_switch(*branches: tuple[Vec, Vec]) -> Vec:
    """Mutually-exclusive branch selection.

    Each branch is `(mask, value)` where `mask` is a 1-shape Vec of 1.0
    (selected) or 0.0 (not). Returns the value paired with the
    selected mask. Exactly one mask should be 1.0 across all branches.

    Adds depth +1 on top of the deepest input.
    """
    if not branches:
        raise ValueError("type_switch requires at least one branch")
    value_shape = branches[0][1].shape
    for i, (mask, value) in enumerate(branches):
        if mask.shape != 1:
            raise ValueError(
                f"branch {i} mask must be 1-shape, got shape {mask.shape}"
            )
        if value.shape != value_shape:
            raise ValueError(
                f"branch {i} value has shape {value.shape}, "
                f"branch 0 has shape {value_shape} — all branches must agree"
            )
    result = np.zeros(value_shape, dtype=np.float64)
    max_depth = 0
    for mask, value in branches:
        result += float(mask._data[0]) * value._data
        max_depth = max(max_depth, mask.depth, value.depth)
    return _make_vec(add_noise(result, _ID_TYPE_SWITCH), depth=max_depth + 1)


# --- 1D PWL factories (call at module level; result is a PWLDef) ---

def relu(input_range: tuple[float, float]) -> PWLDef:
    """`max(0.0, x)` — exact for ReLU within `input_range` because the
    grid places a breakpoint at the kink (0.0).

    `input_range` must contain 0 (i.e., `lo <= 0 <= hi`); otherwise the
    function is just identity (when lo > 0) or zero (when hi < 0) and
    you should use `clamp` or `pwl_def` directly instead of `relu`.
    """
    lo, hi = input_range
    if not (lo <= 0.0 <= hi):
        raise ValueError(
            f"relu input_range must contain 0, got ({lo}, {hi})"
        )
    if lo == 0.0 or hi == 0.0:
        # Degenerate: ReLU is affine across the range, so 2 breakpoints
        # at the endpoints suffice (no kink inside the range).
        return PWLDef(
            lambda x: max(0.0, x),
            breakpoints=2,
            input_range=input_range,
        )
    # Place an explicit breakpoint at 0.0 so the kink lands on the grid.
    return PWLDef(
        lambda x: max(0.0, x),
        breakpoints=3,
        input_range=input_range,
        _xs=[lo, 0.0, hi],
    )


def clamp(lo: float, hi: float) -> PWLDef:
    """Clamp inputs to `[lo, hi]`.

    Equivalent to `pwl_def(lambda x: x, breakpoints=2, input_range=(lo, hi))`
    — the framework's runtime clamping handles the bounds and the
    identity PWL handles the interior. Mirrors the project's
    `clamp(inp, lo, hi)` in `torchwright/ops/arithmetic_ops.py`.
    """
    return PWLDef(
        lambda x: x,
        breakpoints=2,
        input_range=(lo, hi),
    )


def compare_const(c: float, input_range: tuple[float, float]) -> PWLDef:
    """Step function returning ~1.0 if `x > c`, else ~0.0.

    PWL can't represent a true step exactly; the result is a steep
    ramp around `c` with a small deadband whose width is `0.001 * (hi - lo)`.
    Inputs in the deadband interpolate linearly between 0 and 1.

    `c` must be strictly inside `input_range`.
    """
    lo, hi = input_range
    if not lo < hi:
        raise ValueError(
            f"input_range=({lo}, {hi}) must have lo < hi"
        )
    if not lo < c < hi:
        raise ValueError(
            f"compare_const requires lo < c < hi, got lo={lo}, c={c}, hi={hi}"
        )
    span = hi - lo
    eps = span * 0.001
    if not (lo < c - eps and c + eps < hi):
        raise ValueError(
            f"compare_const(c={c}) deadband [{c - eps}, {c + eps}] "
            f"falls outside input_range ({lo}, {hi}); choose a c further "
            f"from the boundary"
        )
    return PWLDef(
        lambda x: 1.0 if x > c else 0.0,
        breakpoints=4,
        input_range=input_range,
        _xs=[lo, c - eps, c + eps, hi],
    )


def piecewise_linear(
    fn: Callable[[float], float],
    breakpoints: int,
    input_range: tuple[float, float],
) -> PWLDef:
    """Generic 1D PWL — equivalent to `pwl_def(fn, breakpoints, input_range)`.

    Provided as a stdlib alias so phase code reads symmetrically with
    the 2D forms.
    """
    return PWLDef(fn, breakpoints, input_range)


# --- 2D PWL factories (call at module level; result is a PWLDef2D) ---

def multiply(
    input_range: tuple[tuple[float, float], tuple[float, float]],
    breakpoints: int | tuple[int, int] = 2,
) -> PWLDef2D:
    """Returns a PWLDef2D for elementwise product `a_i * b_i` over `input_range`.

    Bilinear PWL approximation. Apply the returned PWLDef2D to two Vecs
    of the same shape; each application adds depth +1. Mirrors the
    project's elementwise product op.

    Defaults to `breakpoints=2` because `f(a, b) = a * b` is exactly
    bilinear, so two points per axis reproduce it (modulo runtime FP
    noise).
    """
    return PWLDef2D(
        lambda a, b: a * b,
        breakpoints=breakpoints,
        input_range=input_range,
    )


def piecewise_linear_2d(
    fn: Callable[[float, float], float],
    breakpoints: int | tuple[int, int],
    input_range: tuple[tuple[float, float], tuple[float, float]],
) -> PWLDef2D:
    """Generic 2D PWL with bilinear-interpolation approximation.

    Apply the returned PWLDef2D to two Vecs of the same shape; each
    application adds depth +1. The product of per-dim breakpoints must
    be ≤ 1024.

    For specific common 2D ops (e.g. `multiply`), use those named
    primitives instead — their approximation strategy is fixed and
    matched to a corresponding op in the parent project.
    """
    return PWLDef2D(fn, breakpoints, input_range)
