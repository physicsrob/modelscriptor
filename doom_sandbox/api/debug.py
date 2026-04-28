"""Debugging and assertion primitives.

These access Vec values internally but never expose them to user code
as Python values. Use them to inspect during development and to enforce
invariants at runtime.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .vec import Vec


# --- Printing ---

def print_vec(vec: Vec, label: str = "") -> None:
    """Print `vec`'s values to stdout. Always prints when called.

    For autoregressive loops over many positions, this floods. Prefer
    `debug_watch` with a predicate that fires only on values worth
    seeing.
    """
    prefix = f"{label}: " if label else ""
    print(f"{prefix}{vec._data.tolist()}")


def debug_watch(
    vec: Vec,
    predicate: Callable[[np.ndarray], bool],
    label: str = "",
) -> None:
    """Print `vec`'s values when `predicate(data)` returns True.

    Use to filter which positions emit output — predicates that rarely
    fire keep output volume manageable.
    """
    if predicate(vec._data):
        print_vec(vec, label=label)


# --- Named assertion helpers (mirror the parent project's API) ---

def assert_in_range(vec: Vec, lo: float, hi: float) -> None:
    """Assert all values in `vec` lie in `[lo, hi]`."""
    data = vec._data
    if not (data >= lo).all() or not (data <= hi).all():
        actual_lo = float(data.min())
        actual_hi = float(data.max())
        raise AssertionError(
            f"assert_in_range failed: values span [{actual_lo}, {actual_hi}], "
            f"expected within [{lo}, {hi}]"
        )


def assert_close(vec: Vec, expected: Vec, atol: float = 1e-3) -> None:
    """Assert `vec` and `expected` have matching shape and values within `atol`."""
    if vec.shape != expected.shape:
        raise AssertionError(
            f"assert_close failed: shapes {vec.shape} != {expected.shape}"
        )
    diff = np.abs(vec._data - expected._data)
    max_diff = float(diff.max()) if diff.size else 0.0
    if max_diff > atol:
        raise AssertionError(
            f"assert_close failed: max abs diff {max_diff} > atol {atol}"
        )


def assert_bool(vec: Vec, atol: float = 1e-3) -> None:
    """Assert all values are within `atol` of 0.0 or 1.0."""
    data = vec._data
    dist = np.minimum(np.abs(data), np.abs(data - 1.0))
    if not (dist <= atol).all():
        worst = float(dist.max())
        raise AssertionError(
            f"assert_bool failed: max distance to 0 or 1 is {worst} > atol {atol}"
        )


def assert_integer(vec: Vec, atol: float = 1e-3) -> None:
    """Assert all values are within `atol` of an integer."""
    data = vec._data
    dist = np.abs(data - np.round(data))
    if not (dist <= atol).all():
        worst = float(dist.max())
        raise AssertionError(
            f"assert_integer failed: max distance to nearest integer is "
            f"{worst} > atol {atol}"
        )


# --- Escape hatch ---

def assert_(
    vec: Vec,
    predicate: Callable[[np.ndarray], bool],
    message: str = "",
) -> None:
    """Run `predicate(data)` over `vec`; raise AssertionError(message) if False.

    Prefer the named helpers above when they fit — they read better and
    port more directly to the parent project's assertion API.
    """
    if not predicate(vec._data):
        raise AssertionError(message or "assert_ predicate returned False")
