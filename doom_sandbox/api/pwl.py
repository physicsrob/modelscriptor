"""PWLDef and PWLDef2D — the user-defined compute primitives.

PWL definitions are created at module load (via `pwl_def(...)` for 1D,
or `piecewise_linear_2d(...)` / sibling factories in `doom_sandbox.api.std`
for 2D) and applied many times inside `forward()`. Construction during
`forward()` raises — they are the analog of model weights, fixed across
positions.

Approximation model: at construction we sample `fn` on the breakpoint
grid spanning `input_range`. At call time the input is clamped to
`input_range` and evaluated by linear (1D) or bilinear (2D)
interpolation against the sampled grid. This is the real PWL
approximation — affine functions are exact for free. On top of that we
add a small per-element gaussian (`runtime.noise.add_noise`) to
represent FP32 accumulation and other transformer-runtime noise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..runtime.noise import add_noise
from . import _runtime
from .vec import Vec, _make_vec


class PWLDef:
    """A piecewise-linear approximation of a 1-input scalar function.

    Constructed via `pwl_def(...)`. Applied to Vec inputs via
    `__call__`. Each call adds 1 to the result Vec's depth.
    """

    def __init__(
        self,
        fn: Callable[[float], float],
        breakpoints: int,
        input_range: tuple[float, float],
        *,
        _xs: np.ndarray | list[float] | None = None,
    ):
        if _runtime._FORWARD_RUNNING:
            raise RuntimeError(
                "PWL definitions must be created at module load, before "
                "forward() runs. Move this pwl_def(...) call to module "
                "level."
            )
        if not 2 <= breakpoints <= 1024:
            raise ValueError(
                f"breakpoints must be in [2, 1024], got {breakpoints}"
            )
        lo, hi = input_range
        if not lo < hi:
            raise ValueError(
                f"input_range=({lo}, {hi}) must have lo < hi"
            )
        self._fn = fn
        self._breakpoints = breakpoints
        self._input_range = input_range
        if _xs is None:
            self._xs = np.linspace(lo, hi, breakpoints, dtype=np.float64)
        else:
            xs = np.asarray(_xs, dtype=np.float64)
            if xs.shape != (breakpoints,):
                raise ValueError(
                    f"_xs length must equal breakpoints "
                    f"({breakpoints}), got {xs.shape}"
                )
            if not (xs[0] == lo and xs[-1] == hi):
                raise ValueError(
                    f"_xs must span input_range exactly: first/last "
                    f"sample must equal lo/hi, got {xs[0]}/{xs[-1]}"
                )
            if not np.all(np.diff(xs) > 0):
                raise ValueError(f"_xs must be strictly increasing, got {xs}")
            self._xs = xs
        self._ys = np.array(
            [float(fn(float(x))) for x in self._xs], dtype=np.float64
        )
        self._id = _runtime.next_pwl_id()

    def __call__(self, input: Vec) -> Vec:
        """Apply this PWL elementwise to `input`. Returns a Vec of the
        same shape, with depth = input.depth + 1."""
        clamped = np.clip(input._data, self._xs[0], self._xs[-1])
        interp = np.interp(clamped, self._xs, self._ys)
        result = add_noise(interp, self._id)
        return _make_vec(result, depth=input.depth + 1)


class PWLDef2D:
    """A piecewise-linear approximation of a 2-input scalar function.

    Constructed via `piecewise_linear_2d(...)` or sibling stdlib
    factories in `doom_sandbox.api.std` (e.g. `multiply(...)`). Applied
    to two Vec inputs of the same shape via `__call__`. Each call adds
    1 to the result Vec's depth.

    The default approximation is bilinear interpolation over a per-dim
    breakpoint grid. Specific named ops in `std` may commit to a
    different strategy internally; the agent only sees the typed
    PWLDef2D interface.
    """

    def __init__(
        self,
        fn: Callable[[float, float], float],
        breakpoints: int | tuple[int, int],
        input_range: tuple[tuple[float, float], tuple[float, float]],
    ):
        if _runtime._FORWARD_RUNNING:
            raise RuntimeError(
                "PWL definitions must be created at module load, before "
                "forward() runs. Move this piecewise_linear_2d(...) (or "
                "sibling) call to module level."
            )
        bp = (breakpoints, breakpoints) if isinstance(breakpoints, int) else tuple(breakpoints)
        if len(bp) != 2:
            raise ValueError(
                f"breakpoints must be an int or a 2-tuple, got {bp!r}"
            )
        if not all(2 <= b <= 1024 for b in bp):
            raise ValueError(
                f"per-dim breakpoints must be in [2, 1024], got {bp}"
            )
        if bp[0] * bp[1] > 1024:
            raise ValueError(
                f"product of per-dim breakpoints must be <= 1024, "
                f"got {bp[0]}*{bp[1]}={bp[0] * bp[1]}"
            )
        for i, (lo, hi) in enumerate(input_range):
            if not lo < hi:
                raise ValueError(
                    f"input_range[{i}]=({lo}, {hi}) must have lo < hi"
                )
        self._fn = fn
        self._breakpoints = bp
        self._input_range = input_range
        (lo_a, hi_a), (lo_b, hi_b) = input_range
        self._xs = np.linspace(lo_a, hi_a, bp[0], dtype=np.float64)
        self._ys = np.linspace(lo_b, hi_b, bp[1], dtype=np.float64)
        self._table = np.array(
            [
                [float(fn(float(x), float(y))) for y in self._ys]
                for x in self._xs
            ],
            dtype=np.float64,
        )
        self._id = _runtime.next_pwl_id()

    def __call__(self, a: Vec, b: Vec) -> Vec:
        """Apply this 2D PWL pairwise to `a` and `b` (must have same shape).
        Returns a Vec of the same shape, with depth = max(a.depth, b.depth) + 1."""
        if a.shape != b.shape:
            raise ValueError(
                f"PWLDef2D requires inputs of the same shape, "
                f"got {a.shape} and {b.shape}"
            )
        ax = np.clip(a._data, self._xs[0], self._xs[-1])
        by = np.clip(b._data, self._ys[0], self._ys[-1])
        interp = _bilinear(self._xs, self._ys, self._table, ax, by)
        result = add_noise(interp, self._id)
        return _make_vec(result, depth=max(a.depth, b.depth) + 1)


def _bilinear(
    xs: np.ndarray,
    ys: np.ndarray,
    table: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a regular 2-D grid. Inputs assumed clipped."""
    nx = len(xs)
    ny = len(ys)
    ix = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, ny - 2)
    x0 = xs[ix]
    x1 = xs[ix + 1]
    y0 = ys[iy]
    y1 = ys[iy + 1]
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)
    f00 = table[ix, iy]
    f01 = table[ix, iy + 1]
    f10 = table[ix + 1, iy]
    f11 = table[ix + 1, iy + 1]
    return (
        (1 - fx) * (1 - fy) * f00
        + (1 - fx) * fy * f01
        + fx * (1 - fy) * f10
        + fx * fy * f11
    )


def pwl_def(
    fn: Callable[[float], float],
    breakpoints: int,
    input_range: tuple[float, float],
) -> PWLDef:
    """Define a 1D piecewise-linear approximation of a scalar function.

    Must be called at module load — before `forward()` runs. The
    returned PWLDef is then applied to Vec inputs many times inside
    `forward()`.

    Parameters
    ----------
    fn : Callable[[float], float]
        A 1-input scalar Python function. Applied elementwise to Vec
        inputs at runtime.
    breakpoints : int
        Number of breakpoints in the PWL grid. Must be in [2, 1024].
    input_range : tuple[float, float]
        (lo, hi) declaring the domain the breakpoint grid spans.
        Inputs outside this range are clamped at runtime.

    Returns
    -------
    PWLDef
        Callable on Vecs of any shape (applied elementwise).

    Notes
    -----
    For 2-input operations (products, bilinear functions), use the
    named primitives in `doom_sandbox.api.std` rather than `pwl_def`.
    The choice of 2D approximation strategy is op-specific and lives
    on the platform side.
    """
    return PWLDef(fn, breakpoints, input_range)
