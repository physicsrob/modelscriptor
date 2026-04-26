"""Quantize/dequantize primitives for thinking-token boundaries.

Intermediate values cross thinking-token boundaries as 16-bit integer
token IDs.  The transformer graph itself never rounds — the
integer-cast happens at the host boundary when a float output is
interpreted as a ``uint16`` token ID, and when that token ID is
re-fed as the next step's input.

These two ops are the graph-side halves of that boundary:

- :func:`quantize_to_range` — scale a continuous float in ``[lo, hi]``
  to a continuous float in ``[0, n_levels - 1]``.  The producing
  token's final Linear runs this; the host's uint16 cast (outside the
  graph) does the actual rounding.
- :func:`dequantize_from_range` — scale an integer-valued float in
  ``[0, n_levels - 1]`` back to ``[lo, hi]``.  The consuming token's
  first Linear runs this on the freshly-embedded token value.

Both ops are pure affine transforms (one Linear each, ``Linear(Linear(x))``
composes into a single weight matrix in the compiled graph).  There is
no piecewise-linear approximation and therefore no compile-time
approximation error; the only error anywhere in the pipeline is the
``(hi - lo) / (2 · (n_levels - 1))`` quantization LSB, introduced by
the host's uint16 cast between the two ops.

Typical usage::

    # Producer (thinking-token output layer):
    q_out = quantize_to_range(cross_a_float, lo=-40.0, hi=40.0)
    # host reads q_out, casts to uint16, re-injects as next token.

    # Consumer (next thinking-token's input layer):
    cross_a_restored = dequantize_from_range(q_in, lo=-40.0, hi=40.0)
"""

from __future__ import annotations

from torchwright.graph import Node
from torchwright.graph.asserts import assert_in_range
from torchwright.ops.arithmetic_ops import add_const, multiply_const

# Default number of integer levels: 2**16 (one uint16 token ID).
DEFAULT_N_LEVELS = 65536


def quantize_to_range(
    value: Node,
    lo: float,
    hi: float,
    n_levels: int = DEFAULT_N_LEVELS,
) -> Node:
    """Scale a float in ``[lo, hi]`` to a float in ``[0, n_levels - 1]``.

    Pure affine transform — compiles to a single Linear (and composes
    cleanly with any following Linear into one weight matrix).  **Does
    not round**: the producing token's output is a continuous float,
    and the integer cast happens at the host boundary when the
    transformer's emitted value is interpreted as a ``uint16`` token
    ID.

    Mathematical contract::

        q = (value - lo) * (n_levels - 1) / (hi - lo)

    At ``value = lo`` the output is ``0``; at ``value = hi`` the output
    is ``n_levels - 1``.

    **LSB granularity.**  When paired with a host-side round-to-uint16
    and a subsequent :func:`dequantize_from_range`, the accumulated
    error is ``(hi - lo) / (2 · (n_levels - 1))`` per quantization
    boundary.

    Args:
        value: Scalar node whose value range sits inside ``[lo, hi]``.
            Inputs outside ``[lo, hi]`` produce outputs outside
            ``[0, n_levels - 1]`` (this op does not clamp).
        lo: Lower bound of the float range this value type covers.
        hi: Upper bound of the float range this value type covers.
        n_levels: Number of discrete integer levels (default
            ``2**16 = 65536``, matching a uint16 token ID).  Must be
            ≥ 2.

    Returns:
        Scalar node whose value range is ``[0, n_levels - 1]``.
        Stamped with ``assert_in_range`` so the claim is checked at
        runtime under ``debug=True``.
    """
    assert len(value) == 1, "quantize_to_range expects a 1D scalar"
    assert hi > lo, f"quantize_to_range requires hi > lo, got lo={lo}, hi={hi}"
    assert n_levels >= 2, f"n_levels must be >= 2, got {n_levels}"

    scale = (n_levels - 1) / (hi - lo)
    shifted = add_const(value, -lo)
    q = multiply_const(shifted, scale)
    return assert_in_range(q, 0.0, float(n_levels - 1))


def dequantize_from_range(
    q: Node,
    lo: float,
    hi: float,
    n_levels: int = DEFAULT_N_LEVELS,
) -> Node:
    """Scale an integer-valued float in ``[0, n_levels - 1]`` to a
    float in ``[lo, hi]``.

    Inverse of :func:`quantize_to_range`.  Pure affine — one Linear.
    The consuming token runs this on the dequantization-ready input
    slot immediately after embedding.

    Mathematical contract::

        value = lo + q * (hi - lo) / (n_levels - 1)

    Args:
        q: Scalar node whose value range sits inside
            ``[0, n_levels - 1]``.  Typically the output of the host's
            uint16 → float re-injection.
        lo: Lower bound of the target float range.
        hi: Upper bound of the target float range.
        n_levels: Number of discrete integer levels.  Must match the
            ``n_levels`` used by the paired :func:`quantize_to_range`.

    Returns:
        Scalar node whose value range is ``[lo, hi]``.  Stamped with
        ``assert_in_range``.
    """
    assert len(q) == 1, "dequantize_from_range expects a 1D scalar"
    assert hi > lo, f"dequantize_from_range requires hi > lo, got lo={lo}, hi={hi}"
    assert n_levels >= 2, f"n_levels must be >= 2, got {n_levels}"

    inv_scale = (hi - lo) / (n_levels - 1)
    scaled = multiply_const(q, inv_scale)
    value = add_const(scaled, lo)
    return assert_in_range(value, lo, hi)
