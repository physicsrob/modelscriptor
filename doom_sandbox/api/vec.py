"""The Vec data type — opaque container for computed values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import _runtime


@dataclass(frozen=True)
class Vec:
    """A computed value flowing through user code.

    Vecs are produced by the framework (input embedding, slot
    extractions, `past.*` results), by `PWLDef.__call__`, by `constant`
    (at module load), and by the free utilities `concat` and `split`.
    Phase code should not construct a Vec directly — the framework uses
    `_make_vec` internally.

    Attributes
    ----------
    shape : int
        Length of the Vec.
    depth : int
        Longest op chain that produced this Vec.
    """

    _data: np.ndarray
    depth: int

    @property
    def shape(self) -> int:
        return self._data.shape[0]

    def __repr__(self) -> str:
        return f"Vec(shape={self.shape}, depth={self.depth})"


def _make_vec(data: np.ndarray, depth: int) -> Vec:
    """Framework-internal Vec constructor.

    Coerces `data` to a contiguous 1-D float64 array. Phases must not
    call this — they receive Vecs from the framework, from PWL calls,
    or from `concat` / `split`.
    """
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(
            f"Vec data must be 1-D, got shape {arr.shape}"
        )
    if depth < 0:
        raise ValueError(f"Vec depth must be >= 0, got {depth}")
    return Vec(_data=arr, depth=int(depth))


def constant(value: float | Sequence[float]) -> Vec:
    """Construct a constant Vec at module load.

    Returns a Vec with `depth=0`. If `value` is a scalar, the result is
    a 1-Vec carrying that value. If `value` is a sequence, the result
    is a Vec of `len(value)` carrying those values in order.

    Intended for module-level use. The framework freezes constant
    construction during `forward()` (same as `pwl_def`) and raises if
    you try then; calls outside `forward()` are not enforced beyond
    that, but any constant you construct after the first `run()` will
    not participate in earlier positions.

    Parameters
    ----------
    value : float | Sequence[float]
        Scalar (1-Vec) or sequence (multi-element Vec).

    Examples
    --------
    >>> ZERO = constant(0.0)                    # 1-Vec
    >>> SENTINEL = constant(-1000.0)            # 1-Vec
    >>> PALETTE_K = constant([0.0, 1.0, 2.0])   # 3-Vec
    """
    if _runtime._FORWARD_RUNNING:
        raise RuntimeError(
            "constant(...) must be called at module load, before forward() "
            "runs. Move this call to module level."
        )
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(
            f"constant expects a scalar or 1-D sequence, got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("constant requires at least one value")
    return _make_vec(arr, depth=0)
