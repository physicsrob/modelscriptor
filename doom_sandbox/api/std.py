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


# Stable per-op IDs for noise seeding (separate range from PWL/past IDs).
_ID_TYPE_SWITCH = 8_000_001
_ID_LINEAR = 8_000_002
_ID_SUM = 8_000_003
_ID_REDUCE_SUM = 8_000_004
_ID_ONE_HOT = 8_000_005


# Trapezoidal one-hot kernel parameters — chosen to mirror the parent
# project's `in_range` (`torchwright/ops/map_select.py:266`) which uses
# `step_sharpness = 10.0` (`torchwright/ops/const.py:9`), giving a
# transition zone of width `1/S = 0.1` at each ±0.5 boundary.
_ONE_HOT_PLATEAU_HALF_WIDTH = 0.45
_ONE_HOT_TRANSITION_WIDTH = 0.1


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


# --- Linear projections and pointwise sums (free in transformer terms) ---

def linear(vec: Vec, output_matrix) -> Vec:
    """Linear projection: `result = vec @ output_matrix`.

    `output_matrix` is a numpy array (or nested-list) of shape
    `(vec.shape, d_output)` — left-multiply by the input row vector.
    Mirrors the parent project's `Linear(input_node, output_matrix)`.

    Foundational for everything that needs a fixed linear map of a Vec:

    - **sum-reduce a width-M Vec to a 1-Vec:**
      `linear(v, np.ones((M, 1)))`
    - **pick element `i` from a width-M Vec via a one-hot column:**
      `linear(v, onehot_matrix)`
    - **arbitrary fixed linear combinations.**

    `output_matrix` must be plain data (numpy / list), not a Vec — the
    matrix is the analog of model weights, fixed across positions. The
    framework does not enforce a module-level construction call here
    (the matrix is just data), but a runtime-computed matrix has no
    transformer analog. Stick to constants.

    Adds depth +1.
    """
    matrix = np.asarray(output_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"linear output_matrix must be 2-D, got shape {matrix.shape}"
        )
    if matrix.shape[0] != vec.shape:
        raise ValueError(
            f"linear output_matrix shape {matrix.shape} does not match "
            f"input width {vec.shape}; expected first dim == {vec.shape}"
        )
    result = vec._data @ matrix
    return _make_vec(add_noise(result, _ID_LINEAR), depth=vec.depth + 1)


def sum(*vecs: Vec) -> Vec:
    """Pointwise sum of N Vecs of identical shape.

    Variadic: `sum(a, b)`, `sum(a, b, c, …)`, all the way up to any N
    ≥ 1. All inputs must share a single shape; the result has that
    shape. Mirrors the parent project's `sum_nodes` — a free linear
    op in transformer terms.

    Sugar for `linear(concat(*vecs), tile-identity)`; named directly
    because residual-style adds and combining branch outputs are the
    most common case.

    Note: shadows Python's built-in `sum`. If you import `sum` from
    `doom_sandbox.api`, use `builtins.sum` (or rename on import) for
    the Python list-summing builtin.

    Adds depth +1.
    """
    if not vecs:
        raise ValueError("sum requires at least one Vec")
    shape = vecs[0].shape
    for i, v in enumerate(vecs[1:], start=1):
        if v.shape != shape:
            raise ValueError(
                f"sum: vec at index {i} has shape {v.shape}, "
                f"expected {shape} (all inputs must agree)"
            )
    stacked = np.stack([v._data for v in vecs], axis=0)
    result = stacked.sum(axis=0)
    depth = max(v.depth for v in vecs) + 1
    return _make_vec(add_noise(result, _ID_SUM), depth=depth)


def one_hot(scalar_vec: Vec, n: int) -> Vec:
    """Convert a 1-Vec carrying an integer index in `[0, n)` to a width-`n` one-hot Vec.

    For an exact integer input `idx = k`, returns a clean one-hot:
    slot `k` = 1.0, all others = 0.0.

    For a fractional input the kernel is **trapezoidal** — a flat
    plateau at 1.0 across `|idx − k| < 0.45`, a linear ramp down to
    0.0 across the 0.1-wide transition zone at each ±0.5 boundary,
    0.0 outside. Mirrors the parent project's `in_range`
    (`torchwright/ops/map_select.py:266`), which uses ramp-based
    steps with `step_sharpness = 10` (transition width `1/S = 0.1`).

    The plateau width means moderate floating-point drift in the
    input doesn't move you off the integer bin; only inputs landing
    inside the narrow ±0.05 boundary zones blend across slots. This
    is the same robustness the real transformer's `in_range`
    provides — sandbox semantics intentionally match.

    Out-of-range inputs clamp to `[0, n − 1]` (so `idx = -3` lights
    slot 0; `idx = n + 5` lights slot `n − 1`).

    Adds depth +1.
    """
    if scalar_vec.shape != 1:
        raise ValueError(
            f"one_hot requires a 1-shape Vec, got shape {scalar_vec.shape}"
        )
    if n < 1:
        raise ValueError(f"one_hot n must be >= 1, got {n}")
    idx = float(scalar_vec._data[0])
    idx_clamped = max(0.0, min(n - 1.0, idx))
    plateau = _ONE_HOT_PLATEAU_HALF_WIDTH
    transition = _ONE_HOT_TRANSITION_WIDTH
    half_outer = plateau + transition
    result = np.zeros(n, dtype=np.float64)
    for k in range(n):
        d = abs(idx_clamped - k)
        if d <= plateau:
            result[k] = 1.0
        elif d < half_outer:
            result[k] = (half_outer - d) / transition
        # else: 0.0 (already)
    return _make_vec(add_noise(result, _ID_ONE_HOT), depth=scalar_vec.depth + 1)


def reduce_sum(vec: Vec) -> Vec:
    """Sum across the indices of a single Vec; returns a 1-Vec.

    `reduce_sum(v)` is the natural complement to `multiply` for
    inner-product chains: `reduce_sum(multiply_2d(a, b))` is the
    elementwise dot product of two same-shape Vecs.

    Sugar for `linear(v, np.ones((v.shape, 1)))`; named directly
    because reducing a width-M Vec to a scalar (or extracting a
    single element after a one-hot mask multiply) is the standard
    way to collapse a wide Vec at one position.

    Adds depth +1.
    """
    result = np.array([float(vec._data.sum())], dtype=np.float64)
    return _make_vec(add_noise(result, _ID_REDUCE_SUM), depth=vec.depth + 1)


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
