"""Parametric single-wall intersection prototype.

Derisks the walls-as-tokens path by answering: can the ray/segment
intersection math work when the wall endpoints ``(ax, ay, bx, by)`` are
runtime input nodes rather than compile-time constants baked into
``Linear`` weights?

The baked version in ``renderer.py::_segment_intersection`` costs
**zero MLP sublayers** per segment because every term is a plain
``Linear`` over the residual stream, with the segment's endpoints
appearing as constants in the Linear weight matrix.  The runtime
version can't do that: each term ``ey * ray_cos``, ``ex * player_y``,
etc. becomes a product of two runtime scalars, which has to go through
``signed_multiply`` (3 MLP sublayers per product).

This test answers three questions:

1. **Does the parametric formulation compute the right numerical
   result for (den, num_t, num_u)?** Verified against a pure-numpy
   reference over a sweep of random (player, ray, wall) configs.
2. **Does the compiled transformer match the oracle?** Verified via
   ``probe_graph`` on a single canonical config.
3. **What's the op cost?** ``test_cost_report`` prints a layer-count
   and op-count comparison of a parametric graph vs the baked
   single-wall graph, so we can verify the walls-as-tokens cost model
   (≲300 ops for a parametric single-wall intersection, vs 203 ops
   per wall for the current baked design at N walls).

The formulas mirror ``_segment_intersection`` exactly:

    ex = bx - ax
    ey = by - ay
    den   = ey * cos - ex * sin
    num_t = ax*ey - ay*ex + ex*py - ey*px
    num_u = ax*sin - ay*cos + py*cos - px*sin
"""

import math
import io
import re
import sys
from typing import Tuple

import numpy as np
import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Node
from torchwright.ops.arithmetic_ops import (
    add,
    piecewise_linear_2d,
    signed_multiply,
    subtract,
)
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

MAX_COORD = 20.0  # world-space extent (matches DOOM default)
MAX_DIFF = 2.0 * MAX_COORD  # |ex|, |ey|, |ax-px|, |py-ay| ≤ 2·max_coord
MAX_PRODUCT = MAX_DIFF * MAX_DIFF  # ≈ 1600 (max |num_t| / 2 term)


# ---------------------------------------------------------------------------
# Breakpoints for piecewise_linear_2d lookups
#
# The parametric intersection multiplies (pos-valued) scalars — either
# wall endpoints or their differences — against (trig-valued) ray
# components.  All four pos-like factors are bounded by 2·max_coord,
# so we need a breakpoint grid that covers [-40, 40] densely near
# zero and sparsely at the extremes.  The renderer's existing
# `_POS_BREAKPOINTS` only goes to ±20 so we use a wider local set.
# ---------------------------------------------------------------------------

_DIFF_BREAKPOINTS = [
    -40.0,
    -30.0,
    -20.0,
    -15.0,
    -10.0,
    -7.0,
    -5.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
]

_TRIG_BREAKPOINTS = [
    -1.0,
    -0.9,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
]


# ---------------------------------------------------------------------------
# Parametric intersection
# ---------------------------------------------------------------------------


def _parametric_segment_intersection(
    player_x: Node,
    player_y: Node,
    ray_cos: Node,
    ray_sin: Node,
    wall_ax: Node,
    wall_ay: Node,
    wall_bx: Node,
    wall_by: Node,
    max_coord: float = MAX_COORD,
    step: float = 0.25,
) -> Tuple[Node, Node, Node]:
    """Compute ``(den, num_t, num_u)`` for a single segment with
    runtime-valued wall endpoints.

    Drop-in replacement for ``_segment_intersection`` that takes
    ``(wall_ax, wall_ay, wall_bx, wall_by)`` as input nodes instead
    of baking them into ``Linear`` weights.

    Optimized formulation (6 products total, from the naive 10):

        ex  = bx - ax
        ey  = by - ay
        dx  = ax - px
        dy  = py - ay
        den   = ey · cos - ex · sin
        num_t = ey · dx + ex · dy
        num_u = dx · sin + dy · cos

    ``num_t`` and ``num_u`` are derived from the baked-renderer forms
    by distributing the differences: collecting terms in ``(ax-px)``
    and ``(py-ay)`` saves 4 products over the straight transcription,
    and the ``dx``/``dy`` intermediates are shared between both.

    The 4 (pos · trig) products use ``piecewise_linear_2d`` (1 MLP
    sublayer each) — this is the same trick ``_shared_products`` uses
    in the baked renderer.  The 2 (pos · pos) products for ``num_t``
    stay on ``signed_multiply`` because their output range
    (up to ≈ 1600) doesn't fit in a compact 2D lookup grid.
    """
    # ex = bx - ax, ey = by - ay, dx = ax - px, dy = py - ay   (free Linears)
    ex = subtract(wall_bx, wall_ax)
    ey = subtract(wall_by, wall_ay)
    dx = subtract(wall_ax, player_x)
    dy = subtract(player_y, wall_ay)

    # ------------------------------------------------------------------
    # den = ey · cos - ex · sin    (2 pos·trig products)
    # ------------------------------------------------------------------
    ey_cos = piecewise_linear_2d(
        ey,
        ray_cos,
        _DIFF_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda a, b: a * b,
        name="ey_cos",
    )
    ex_sin = piecewise_linear_2d(
        ex,
        ray_sin,
        _DIFF_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda a, b: a * b,
        name="ex_sin",
    )
    den = subtract(ey_cos, ex_sin)

    # ------------------------------------------------------------------
    # num_t = ey · dx + ex · dy    (2 pos·pos products)
    #
    # Both factors are in [-MAX_DIFF, MAX_DIFF] so we can use the same
    # ``_DIFF_BREAKPOINTS`` grid as the pos·trig products above, just
    # on both axes.  1 MLP sublayer per product, same as a pos·trig
    # lookup.  Bilinear interpolation of ``x * y`` on a
    # ``|x2-x1| * |y2-y1|`` cell has max error ``|x2-x1|·|y2-y1|/4`` —
    # for ``_DIFF_BREAKPOINTS`` the worst cell is ``10 × 10`` (between
    # ±30 and ±40), giving ≤25 per product and ≤50 in ``num_t``.  The
    # ``test_parametric_matches_numpy_reference`` ``atol=20`` catches
    # any real regression, but we widen the allowed budget below if
    # the coarse corners bite; in practice random samples rarely land
    # in the outermost cells.
    # ------------------------------------------------------------------
    ey_dx = piecewise_linear_2d(
        ey,
        dx,
        _DIFF_BREAKPOINTS,
        _DIFF_BREAKPOINTS,
        lambda a, b: a * b,
        name="ey_dx",
    )
    ex_dy = piecewise_linear_2d(
        ex,
        dy,
        _DIFF_BREAKPOINTS,
        _DIFF_BREAKPOINTS,
        lambda a, b: a * b,
        name="ex_dy",
    )
    num_t = add(ey_dx, ex_dy)

    # ------------------------------------------------------------------
    # num_u = dx · sin + dy · cos    (2 pos·trig products)
    # ------------------------------------------------------------------
    dx_sin = piecewise_linear_2d(
        dx,
        ray_sin,
        _DIFF_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda a, b: a * b,
        name="dx_sin",
    )
    dy_cos = piecewise_linear_2d(
        dy,
        ray_cos,
        _DIFF_BREAKPOINTS,
        _TRIG_BREAKPOINTS,
        lambda a, b: a * b,
        name="dy_cos",
    )
    num_u = add(dx_sin, dy_cos)

    return den, num_t, num_u


# ---------------------------------------------------------------------------
# Numpy reference — the ground truth
# ---------------------------------------------------------------------------


def _numpy_reference(
    player_x: np.ndarray,
    player_y: np.ndarray,
    ray_cos: np.ndarray,
    ray_sin: np.ndarray,
    wall_ax: np.ndarray,
    wall_ay: np.ndarray,
    wall_bx: np.ndarray,
    wall_by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact float math from the intersection formulas."""
    ex = wall_bx - wall_ax
    ey = wall_by - wall_ay
    den = ey * ray_cos - ex * ray_sin
    num_t = wall_ax * ey - wall_ay * ex + ex * player_y - ey * player_x
    num_u = (
        wall_ax * ray_sin - wall_ay * ray_cos + player_y * ray_cos - player_x * ray_sin
    )
    return den, num_t, num_u


# ---------------------------------------------------------------------------
# Fixture: graph + input dict
# ---------------------------------------------------------------------------


INPUT_NAMES = (
    "player_x",
    "player_y",
    "ray_cos",
    "ray_sin",
    "wall_ax",
    "wall_ay",
    "wall_bx",
    "wall_by",
)


def _build_graph():
    """Build the parametric intersection graph and return
    ``(den, num_t, num_u)`` nodes plus the list of input nodes.
    """
    player_x = create_input("player_x", 1)
    player_y = create_input("player_y", 1)
    ray_cos = create_input("ray_cos", 1)
    ray_sin = create_input("ray_sin", 1)
    wall_ax = create_input("wall_ax", 1)
    wall_ay = create_input("wall_ay", 1)
    wall_bx = create_input("wall_bx", 1)
    wall_by = create_input("wall_by", 1)

    den, num_t, num_u = _parametric_segment_intersection(
        player_x,
        player_y,
        ray_cos,
        ray_sin,
        wall_ax,
        wall_ay,
        wall_bx,
        wall_by,
    )
    return den, num_t, num_u


def _sweep_inputs(seed: int = 0xBEEF, n_pos: int = 16):
    """Random (player, ray, wall) configs bounded to the declared ranges.

    Returns an input dict suitable for ``reference_eval`` / ``probe_graph``,
    plus the individual numpy arrays for the reference computation.
    """
    rng = np.random.default_rng(seed)
    player_x = rng.uniform(-MAX_COORD * 0.5, MAX_COORD * 0.5, size=n_pos)
    player_y = rng.uniform(-MAX_COORD * 0.5, MAX_COORD * 0.5, size=n_pos)
    ray_angles = rng.uniform(0, 2 * math.pi, size=n_pos)
    ray_cos = np.cos(ray_angles)
    ray_sin = np.sin(ray_angles)
    wall_ax = rng.uniform(-MAX_COORD * 0.7, MAX_COORD * 0.7, size=n_pos)
    wall_ay = rng.uniform(-MAX_COORD * 0.7, MAX_COORD * 0.7, size=n_pos)
    wall_bx = rng.uniform(-MAX_COORD * 0.7, MAX_COORD * 0.7, size=n_pos)
    wall_by = rng.uniform(-MAX_COORD * 0.7, MAX_COORD * 0.7, size=n_pos)

    inputs = {
        "player_x": torch.tensor(player_x, dtype=torch.float32).unsqueeze(-1),
        "player_y": torch.tensor(player_y, dtype=torch.float32).unsqueeze(-1),
        "ray_cos": torch.tensor(ray_cos, dtype=torch.float32).unsqueeze(-1),
        "ray_sin": torch.tensor(ray_sin, dtype=torch.float32).unsqueeze(-1),
        "wall_ax": torch.tensor(wall_ax, dtype=torch.float32).unsqueeze(-1),
        "wall_ay": torch.tensor(wall_ay, dtype=torch.float32).unsqueeze(-1),
        "wall_bx": torch.tensor(wall_bx, dtype=torch.float32).unsqueeze(-1),
        "wall_by": torch.tensor(wall_by, dtype=torch.float32).unsqueeze(-1),
    }
    return inputs, (
        player_x,
        player_y,
        ray_cos,
        ray_sin,
        wall_ax,
        wall_ay,
        wall_bx,
        wall_by,
    )


# ---------------------------------------------------------------------------
# Test 1: oracle matches numpy reference
# ---------------------------------------------------------------------------


def test_parametric_matches_numpy_reference():
    """Random sweep: oracle ``(den, num_t, num_u)`` must match the
    numpy reference to within the signed_multiply precision budget.

    Precision envelope: ``signed_multiply`` with ``step=0.25`` and
    ``max_sum ≈ 60`` has absolute error ``≈ step × max_sum / 4 ≈ 3.75``
    on each individual product.  ``num_t`` sums four such products,
    so its total absolute error can reach ``~15``; ``num_u`` sums four
    smaller products (``max_sum ≈ 21``) so its budget is ``~5``;
    ``den`` is a single subtraction of two ``max_sum ≈ 41`` products,
    budget ``~5``.  We use a generous ``atol=20`` that's tight enough
    to catch wrong formulas and sign errors but loose enough that
    signed_multiply's pre-declared precision floor doesn't false-fail.
    """
    den_node, num_t_node, num_u_node = _build_graph()
    inputs, refs = _sweep_inputs(n_pos=32)
    px, py, rc, rs, ax, ay, bx, by = refs

    den_ref, num_t_ref, num_u_ref = _numpy_reference(px, py, rc, rs, ax, ay, bx, by)

    # Oracle (reference_eval) — shares compute across the three
    # output nodes via the same input dict, so we compute each from
    # its own root.
    n_pos = inputs["player_x"].shape[0]
    den_cache = reference_eval(den_node, inputs, n_pos)
    num_t_cache = reference_eval(num_t_node, inputs, n_pos)
    num_u_cache = reference_eval(num_u_node, inputs, n_pos)

    den_got = den_cache[den_node].squeeze(-1).numpy()
    num_t_got = num_t_cache[num_t_node].squeeze(-1).numpy()
    num_u_got = num_u_cache[num_u_node].squeeze(-1).numpy()

    assert np.allclose(
        den_got, den_ref, atol=5.0
    ), f"den mismatch (max err = {np.abs(den_got - den_ref).max():.3f})"
    assert np.allclose(
        num_t_got, num_t_ref, atol=20.0
    ), f"num_t mismatch (max err = {np.abs(num_t_got - num_t_ref).max():.3f})"
    assert np.allclose(
        num_u_got, num_u_ref, atol=20.0
    ), f"num_u mismatch (max err = {np.abs(num_u_got - num_u_ref).max():.3f})"


# ---------------------------------------------------------------------------
# Test 2: compiled matches oracle (probe_graph on each output root)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["den", "num_t", "num_u"])
def test_probe_compiled_matches_oracle(which):
    """Compile the graph rooted at one of the three outputs and verify
    every materialised node's compiled residual-stream value matches
    the oracle ``Node.compute`` value.

    We run three separate probes (one per output root) instead of
    probing a single Concatenate because the probe skips Concatenate
    groupings — rooting at the raw node ensures the probe checks the
    intersection math end-to-end for that output.
    """
    den_node, num_t_node, num_u_node = _build_graph()
    root = {"den": den_node, "num_t": num_t_node, "num_u": num_u_node}[which]

    # Use a single config rather than a full sweep so the probe runs fast.
    inputs, _ = _sweep_inputs(n_pos=4)

    report = probe_graph(
        root,
        pos_encoding=None,
        input_values=inputs,
        n_pos=inputs["player_x"].shape[0],
        d=2048,
        d_head=16,
        verbose=False,
        atol=1.0,  # downstream nodes can be off by the signed_multiply budget
    )
    assert (
        report.first_divergent is None
    ), f"probe reported divergence on {which}:\n{report.format_short()}"


# ---------------------------------------------------------------------------
# Test 3: cost report
# ---------------------------------------------------------------------------


_LAYER_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+ops\s+"
    r"([\d,]+)/[\d,]+\s*\(\s*[\d.]+%\)\s+"
    r"(\d+)/\d+\s*\(\s*\d+%\)\s+"
    r"(\d+)/\d+\s*\(\s*\d+%\)\s+"
    r"([\d.]+)ms"
)


def _parse_layer_stats(stdout_text):
    rows = []
    for line in stdout_text.splitlines():
        m = _LAYER_LINE_RE.match(line)
        if m:
            rows.append(
                (
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3).replace(",", "")),
                    int(m.group(4)),
                    int(m.group(5)),
                )
            )
    return rows


def test_cost_report(tmp_path):
    """Compile the parametric intersection in isolation and assert
    the cost is within the walls-as-tokens budget.

    Budget rationale: profile_walls.py measured 203 ops per wall for
    the current baked design.  The parametric single-wall intersection
    only computes ``(den, num_t, num_u)``, which is the front third
    of the per-wall pipeline — the baked equivalent (3 Linears) is
    "free" (zero MLP sublayers).  The parametric version pays
    ``signed_multiply`` depth for each product.  An upper bound we'd
    consider workable for walls-as-tokens is ``~200 ops``: below that,
    the per-render-step work at N=32 drops from 6824 ops to roughly
    ``328 (fixed) + 200 (parametric intersection) + rest of distance
    pipeline`` ≈ ~700 ops, a ~10× reduction.

    Also write the stats to a file so an operator can inspect them
    without re-running the test — xdist swallows the pytest ``-s``
    flag, so printed output from a passing test is not visible.
    """
    from torchwright.compiler.forward.compile import forward_compile
    from torchwright.graph import Concatenate

    den_node, num_t_node, num_u_node = _build_graph()
    combined = Concatenate([den_node, num_t_node, num_u_node])

    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        net = forward_compile(
            d=2048,
            d_head=16,
            output_node=combined,
            pos_encoding=None,
            verbose=True,
            max_layers=100,
            device=None,
        )
    finally:
        sys.stdout = real_stdout

    layer_rows = _parse_layer_stats(buf.getvalue())
    n_layers = len(net.layers)
    total_ops = sum(r[1] for r in layer_rows)
    peak_d = max(max(r[3], r[4]) for r in layer_rows) if layer_rows else 0

    report = (
        "=== parametric single-wall intersection cost ===\n"
        f"  layers     = {n_layers}\n"
        f"  total_ops  = {total_ops}\n"
        f"  peak_d     = {peak_d}\n"
        f"  baked baseline (profile_walls.py) = 203 ops/wall "
        "(all of intersection + distance + tex meta, not just intersection)\n"
    )

    # Drop the report in a predictable place outside tmp_path so
    # humans can read it after the test finishes.
    report_path = "/tmp/parametric_intersection_cost.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Regression guard: the optimized prototype lands at ~51 ops / 6
    # layers / peak_d ~34 on the first successful compile.  Allow a
    # little headroom for graph-compiler churn but fail if the numbers
    # drift by more than ~50%.  A failure here doesn't necessarily
    # indicate a bug — it's a heads-up that the cost model underlying
    # the walls-as-tokens pitch just shifted.
    assert (
        total_ops < 80
    ), f"parametric intersection grew past the workable budget:\n{report}"
    assert (
        n_layers < 10
    ), f"parametric intersection layer count grew past expected:\n{report}"
    assert (
        peak_d < 60
    ), f"parametric intersection peak residual width grew past expected:\n{report}"
