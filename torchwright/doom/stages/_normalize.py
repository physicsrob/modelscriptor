"""Per-coord normalization helper for the in-graph scale-invariant rendering.

Computes ``coord · inv_scale`` via log-domain decomposition, where
``log_inv_scale`` is the broadcast scalar produced by the scale-find pass
in ``thinking_wall.py``.  The mathematical identity is:

    coord · inv_scale  =  sign(coord) · |coord| · inv_scale
                       =  sign(coord) · exp( log|coord| + log(inv_scale) )

All four pieces are implemented as separate ops so the chain stays
inside the float32-precision regime each individual op is measured for:

* ``log_abs(coord)`` — single-sublayer fused ``log(clamp(|coord|, …))``.
* ``Linear`` — adds ``log_abs_coord + log_inv_scale`` exactly.
* ``exp`` — single-sublayer piecewise-linear, uniform breakpoints.
* ``cond_gate(approximate=False)`` — float-exact sign multiplication via
  two parallel gates and a subtract.

Phase 0's measurement on the operating range
(|coord| in [0.1, 100], inv_scale in [1/100, 1], coupled by
|coord · inv_scale| ≤ 1) measured 0.07 % worst-case relative error
end-to-end at 256 breakpoints — comfortably below the 0.5 % per-multiply
budget that the texture-column path requires.

Critical-path cost per call: ~4 MLP sublayers (1 for ``log_abs`` + 1
for ``exp`` + 2 for ``cond_gate(approximate=False)``).  ``log_inv_scale``
is computed once at scale-find time and amortised across every
``normalize_coord`` call in the graph.

See ``docs/op_noise_data.json`` for the per-op noise reference and
``scripts/phase0_measure_normalize.py`` for the measurement harness.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.ops.arithmetic_ops import (
    compare,
    exp,
    log_abs,
    multiply_const,
    subtract,
)
from torchwright.ops.logic_ops import cond_gate


# Default operating bounds.  ``MAX_ABS`` matches the user-stated DOOM
# envelope.  ``MIN_ABS`` is the floor below which ``log_abs`` clamps;
# the residual at coord=0 is ``-MIN_ABS · inv_scale`` (the asymmetric
# sign mux picks ``false_level`` at the threshold and produces
# ``-abs_normalized``), so MIN_ABS sets a floor on the absolute bias
# we accept at coord=0:
#
#   bias  ≤  MIN_ABS · max(inv_scale)  =  MIN_ABS / min(scene_extent)
#
# At MIN_ABS = 0.01 and max_coord = 100, the worst-case bias at the
# full envelope is 0.01 · 0.01 = 1e-4, i.e. 0.01 % in normalized space.
# Smaller scenes (box room, scale ≈ 5) hit ~0.2 %.  Both sit well
# below the 0.5 % per-multiply budget.  The ratio max_abs/min_abs =
# 1e4 keeps ``log_abs`` on its single-sublayer fast path.
DEFAULT_MAX_ABS = 100.0
DEFAULT_MIN_ABS = 0.01

# Breakpoint counts for the underlying ops.  256 is the Phase 0 sweet
# spot — 7× headroom on the 0.5 % per-multiply target with the cheapest
# section count.  Bumping these gives diminishing returns; halving them
# costs <1 sublayer and a few % of headroom (still under target).
_LOG_ABS_BREAKPOINTS = 256
_EXP_BREAKPOINTS = 256


def normalize_coord(
    coord: Node,
    log_inv_scale: Node,
    *,
    max_abs: float = DEFAULT_MAX_ABS,
    min_abs: float = DEFAULT_MIN_ABS,
    name: str = "normalize_coord",
) -> Node:
    """Compute ``coord · inv_scale`` via log-domain decomposition.

    Args:
        coord: Signed scalar node carrying the raw world coord.  Magnitude
            below ``min_abs`` is clamped (handled inside ``log_abs``);
            the resulting absolute error in the normalized output is
            bounded by ``min_abs · |inv_scale|`` and is exactly cancelled
            by the sign-mux at ``coord = 0``.
        log_inv_scale: Scalar node holding ``log(1 / global_max_abs_coord)``
            from the Phase-1 scale-find pass.  Same value at every
            position (broadcast).
        max_abs: Upper bound on ``|coord|``; sets ``log_abs``'s
            ``max_abs`` and the operating range of ``exp``.
        min_abs: Lower clamp on ``|coord|``; below this the function is
            constant.  Must be > 0.
        name: Debug label prefix for the produced nodes.

    Returns:
        Scalar node containing ``coord · inv_scale``, normalized to
        approximately ``[-1, 1]`` when the scale-find pass produced a
        valid ``log_inv_scale``.
    """
    assert max_abs > min_abs > 0.0
    assert len(coord) == 1, "coord must be a scalar"
    assert len(log_inv_scale) == 1, "log_inv_scale must be a scalar"

    # log|coord| (single fused sublayer at our envelope).
    log_abs_coord = log_abs(
        coord,
        min_abs=min_abs,
        max_abs=max_abs,
        n_breakpoints=_LOG_ABS_BREAKPOINTS,
    )

    # log_normalized = log_abs_coord + log_inv_scale.  Linear, exact —
    # fused into the next sublayer's input projection by the compiler.
    sum_in = Concatenate([log_abs_coord, log_inv_scale])
    sum_w = torch.tensor([[1.0], [1.0]])
    log_normalized = Linear(sum_in, sum_w, name=f"{name}_log_sum")

    # log_normalized's range:
    #   lower bound = log(min_abs) + log(min_inv_scale)
    #     where min_inv_scale = 1 / max_abs (the smallest scale).
    #   upper bound = 0  (because |coord · inv_scale| ≤ 1 on the
    #     operating manifold, so log of that is ≤ 0).
    # We declare the whole interval to exp; samples outside it would be
    # off-manifold and produce undefined behavior, but the scale-find
    # constraint forbids that.
    log_min = math.log(min_abs) + math.log(1.0 / max_abs)
    log_max = 0.0
    abs_normalized = exp(
        log_normalized,
        min_value=log_min,
        max_value=log_max,
        n_breakpoints=_EXP_BREAKPOINTS,
    )

    # Sign multiplication via two cond_gates and a subtract.  Float-exact
    # on clean ±1 conditions; at the ramp midpoint (coord = 0) both gates
    # return abs_normalized and the subtract cancels them — yielding 0,
    # which is the correct mathematical answer (0 · anything = 0).
    sign_pos = compare(coord, thresh=0.0, true_level=1.0, false_level=-1.0)
    sign_neg = compare(coord, thresh=0.0, true_level=-1.0, false_level=1.0)
    pos_part = cond_gate(sign_pos, abs_normalized, approximate=False)
    neg_part = cond_gate(sign_neg, abs_normalized, approximate=False)
    return subtract(pos_part, neg_part)


def normalize_coord_pair(
    coord_a: Node,
    coord_b: Node,
    log_inv_scale: Node,
    *,
    max_abs: float = DEFAULT_MAX_ABS,
    min_abs: float = DEFAULT_MIN_ABS,
    name_a: str = "normalize_coord_a",
    name_b: str = "normalize_coord_b",
) -> Tuple[Node, Node]:
    """Normalize two coords sharing one ``log_inv_scale``.

    No structural sharing yet — currently just two independent
    ``normalize_coord`` calls.  The compiler may fuse parallel ops with
    matching breakpoint grids; if that doesn't materialise we can revisit
    a more compact form (e.g., a single 2-wide ``log_abs`` and a single
    2-wide ``exp``) when we measure the post-Phase-3 compile depth.
    """
    a_norm = normalize_coord(
        coord_a,
        log_inv_scale,
        max_abs=max_abs,
        min_abs=min_abs,
        name=name_a,
    )
    b_norm = normalize_coord(
        coord_b,
        log_inv_scale,
        max_abs=max_abs,
        min_abs=min_abs,
        name=name_b,
    )
    return a_norm, b_norm
