"""Predicate helpers that wrap :class:`torchwright.graph.Assert`.

Each helper returns the :class:`Assert`-wrapped node so graph
construction reads naturally::

    from torchwright.graph.asserts import assert_01, assert_strictly_less

    position_onehot = assert_onehot(position_onehot)
    sentinel = assert_strictly_less(real_score, sentinel)

The wrapped node is stripped during compilation, so Asserts cost
nothing in the compiled transformer's weights or schedule.  They run
during reference evaluation (``reference_eval``) and, when available,
during compiled-graph probing (``probe_compiled``).

Naming convention (matches the rest of the codebase)
----------------------------------------------------
* ``bool`` values are ±1 — see ``ops/arithmetic_ops.bool_to_01`` at
  ``torchwright/ops/arithmetic_ops.py:97``.
* ``01`` values are {0, 1} — matches ``bool_to_01``'s output.

Safe placement notes live in each helper's docstring.  The core split
is **pre-attention vs. post-attention**: attention layers average
values under a softmax, which blends one-hot encodings and introduces
PL2D-scale fuzz.  Invariants that only hold pre-attention (exactly-one
one-hots, distinct values) should be asserted before the attention
layer; invariants that hold on aggregated values (``{0, 1}`` bools
after broadcast) take a tolerance large enough to absorb that fuzz.
"""

from typing import List, Set

import torch

from torchwright.graph import Node, Concatenate
from torchwright.graph.misc import Assert, Predicate


def collect_asserts(output_node: Node) -> List[Assert]:
    """Walk the graph from ``output_node`` and collect every reachable Assert.

    Call this **before** passing the graph to ``compile_headless`` —
    compile-time analysis strips Asserts in-place, so post-compile
    traversal won't find them.  Tests that want to verify Asserts hold
    on *compiled* values (via ``probe_compiled``) collect the list up
    front and pass it through.
    """
    seen: Set[int] = set()
    asserts: List[Assert] = []
    stack: List[Node] = [output_node]
    while stack:
        node = stack.pop()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        if isinstance(node, Assert):
            asserts.append(node)
        for inp in node.inputs:
            stack.append(inp)
    return asserts


def _format_bad(x: torch.Tensor, mask: torch.Tensor, *, max_show: int = 3) -> str:
    """Summarize the first ``max_show`` positions where ``mask`` is True."""
    bad_indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not bad_indices:
        return "no bad entries"
    head = bad_indices[:max_show]
    more = "" if len(bad_indices) <= max_show else f" (+{len(bad_indices) - max_show} more)"
    # Show the actual offending values at those indices.
    flat = x.flatten()
    shown = ", ".join(f"[{i}]={flat[i].item():.4f}" for i in head)
    return f"bad at {shown}{more}"


def assert_in_range(
    node: Node, lo: float, hi: float, *, atol: float = 1e-3,
) -> Node:
    """Assert every element of ``node`` lies in ``[lo - atol, hi + atol]``.

    **Safe placement**: anywhere.  The ``atol`` default (1e-3) absorbs
    typical post-PL2D noise.  Used for bounds like ``|score| ≤ 100``
    on ``attend_argmin_unmasked`` scores.
    """
    def predicate(x: torch.Tensor) -> tuple:
        bad = (x < lo - atol) | (x > hi + atol)
        if not bad.any():
            return True, ""
        return False, f"expected [{lo}, {hi}] (atol={atol}); {_format_bad(x, bad)}"

    return Assert(
        node, predicate,
        message=f"values in [{lo}, {hi}]",
    )


def assert_bool(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert every element is ≈ +1 or ≈ -1 (a "bool" in the codebase convention).

    **Safe placement**: outputs of ``compare``, ``equals_vector``,
    ``bool_any_true``, ``bool_all_true``, ``bool_not``, ``cond_gate``,
    and any ±1-valued node *before* it's averaged by attention.
    """
    def predicate(x: torch.Tensor) -> tuple:
        near_pos = (x - 1.0).abs() <= atol
        near_neg = (x + 1.0).abs() <= atol
        bad = ~(near_pos | near_neg)
        if not bad.any():
            return True, ""
        return False, f"expected ±1 (atol={atol}); {_format_bad(x, bad)}"

    return Assert(node, predicate, message="bool (±1)")


def assert_01(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert every element is ≈ 0 or ≈ 1.

    **Safe placement**: outputs of ``bool_to_01``, ``in_range`` flags,
    and post-broadcast {0, 1} vectors such as ``side_P_vec`` after the
    BSP ``attend_mean_where * max_bsp_nodes`` recovery.  Uses a looser
    ``atol`` than ``assert_bool`` because broadcast values accumulate
    more interpolation fuzz than raw ``compare`` outputs.
    """
    def predicate(x: torch.Tensor) -> tuple:
        near_zero = x.abs() <= atol
        near_one = (x - 1.0).abs() <= atol
        bad = ~(near_zero | near_one)
        if not bad.any():
            return True, ""
        return False, f"expected {{0,1}} (atol={atol}); {_format_bad(x, bad)}"

    return Assert(node, predicate, message="binary (0/1)")


def assert_onehot(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert each row is a one-hot vector: exactly one ≈ 1, rest ≈ 0.

    **Safe placement**: **pre-attention only.**  After an attention
    layer averages one-hot vectors across positions, the result is a
    *distribution*, not a one-hot — use ``assert_01`` on specific
    components instead.  Examples of safe sites: ``position_onehot`` at
    WALL positions, ``sort_rank_onehot`` at SORTED positions, *before*
    either is routed through an attention value.
    """
    def predicate(x: torch.Tensor) -> tuple:
        if x.ndim != 2:
            return False, f"expected 2D (n_pos, d); got shape {tuple(x.shape)}"
        near_zero = x.abs() <= atol
        near_one = (x - 1.0).abs() <= atol
        elementwise_ok = near_zero | near_one
        rows_elem_ok = elementwise_ok.all(dim=-1)
        row_sums = x.sum(dim=-1)
        rows_sum_ok = (row_sums - 1.0).abs() <= atol
        rows_ok = rows_elem_ok & rows_sum_ok
        if rows_ok.all():
            return True, ""
        bad_rows = (~rows_ok).nonzero(as_tuple=False).flatten().tolist()
        head = bad_rows[:3]
        more = "" if len(bad_rows) <= 3 else f" (+{len(bad_rows) - 3} more)"
        summary = ", ".join(
            f"row {i} sum={row_sums[i].item():.3f}"
            for i in head
        )
        return False, f"not one-hot (atol={atol}); {summary}{more}"

    return Assert(node, predicate, message="one-hot")


def assert_strictly_less(
    a: Node, b: Node, *, margin: float = 0.0,
) -> Node:
    """Assert every element of ``a`` is strictly less than the matching element of ``b``.

    Both nodes must have the same width.  Returns a wrapped version of
    ``b`` (the upper bound) — the natural thing to thread into
    downstream graph construction when the point is to guarantee the
    sentinel stays above the real values.

    Implemented as an Assert over ``Concatenate([a, b])`` so the
    predicate can split the value without a new two-input node shape.

    **Safe placement**: anywhere.  Used for sentinel ordering (e.g.
    ``bsp_sentinel`` < ``nonwall_sentinel``, or real ranks <
    sentinel).  For pre-attention ordering between positions, gate
    inputs with ``select(is_wall, ...)`` first.
    """
    if len(a) != len(b):
        raise ValueError(
            f"assert_strictly_less: width mismatch (a={len(a)}, b={len(b)})"
        )
    width = len(a)

    def predicate(x: torch.Tensor) -> tuple:
        a_val = x[:, :width]
        b_val = x[:, width:]
        diff = b_val - a_val
        bad = diff <= margin
        if not bad.any():
            return True, ""
        i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"expected a < b (margin={margin}); "
            f"row={i} col={j}: a={a_val[i, j].item():.4f}, "
            f"b={b_val[i, j].item():.4f}"
        )

    wrapped = Assert(
        Concatenate([a, b]), predicate,
        message=f"strictly_less (margin={margin})",
    )
    # The wrapped node is the concat of both; callers want ``b`` back so
    # the subgraph that *used* ``b`` for sentinel ordering still reads
    # natural.  Extract ``b``'s slice via a Linear projection so the
    # returned node has ``b``'s width and value.
    from torchwright.graph import Linear
    proj = torch.zeros(len(a) + len(b), len(b))
    for i in range(len(b)):
        proj[len(a) + i, i] = 1.0
    return Linear(wrapped, proj, name="strictly_less_b")


def assert_unique_values(
    node: Node, *, margin: float = 0.5,
) -> Node:
    """Assert every pair of distinct components in a row differs by at least ``margin``.

    Useful for sort-score vectors (like a stacked ``bsp_rank`` across
    WALL positions) where any two tied scores would make the downstream
    ``attend_argmin_unmasked`` softmax-blend their values.  ``margin``
    defaults to 0.5 — bigger than typical PL2D fuzz, small enough not
    to conflict with the real 1.0 rank spacing.

    **Safe placement**: **pre-attention only.**  The whole point is to
    catch ties *before* they can blend.
    """
    def predicate(x: torch.Tensor) -> tuple:
        if x.ndim != 2:
            return False, f"expected 2D (n_pos, d); got shape {tuple(x.shape)}"
        n_pos, d = x.shape
        if d < 2:
            return True, ""
        # Pairwise absolute differences within each row.
        # x.unsqueeze(-1) - x.unsqueeze(-2) has shape (n_pos, d, d).
        diffs = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs()
        # Mask out the diagonal (self-pairs).
        eye = torch.eye(d, dtype=torch.bool, device=x.device).unsqueeze(0)
        bad = (diffs < margin) & ~eye
        if not bad.any():
            return True, ""
        row_idx, i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"duplicate values (margin={margin}); "
            f"row={row_idx}: [{i}]={x[row_idx, i].item():.4f} "
            f"≈ [{j}]={x[row_idx, j].item():.4f}"
        )

    return Assert(node, predicate, message=f"unique values (margin={margin})")


def assert_distinct_across(
    value: Node, where: Node, *, margin: float = 0.5,
) -> Node:
    """Assert per-position ``value`` is pairwise-distinct across rows where ``where ≈ 1``.

    Use this for cross-position uniqueness invariants like "``bsp_rank``
    values at WALL positions are all different" — the precondition that
    keeps a downstream ``attend_argmin_unmasked`` softmax concentrated
    on a single key.

    Rows with ``where.squeeze(-1) ≤ 0.5`` are ignored (the predicate
    sees only the valid subset).  This makes the check tolerant of both
    ±1 bool validity (where ``≥ 0.5`` maps to "valid") and {0, 1}
    indicator validity.

    Pairwise comparison uses L∞ distance on the ``d_value``-wide vectors
    (for the common 1-wide scalar case, L∞ is just absolute difference);
    any pair closer than ``margin`` triggers failure.  Default margin
    (0.5) is larger than typical PL2D fuzz and smaller than the natural
    spacing of integer ranks (1.0).

    **Safe placement**: pre- or post-attention.  Prefer upstream of the
    attention that consumes ``value`` so a tie is caught before it
    can blend.

    Implementation mirrors :func:`assert_strictly_less` — composes the
    two inputs via ``Concatenate`` so no new two-input Assert shape is
    required; a trailing ``Linear`` projects the asserted composite
    back to ``value``'s width for downstream use.
    """
    d_value = len(value)
    d_where = len(where)

    def predicate(x: torch.Tensor) -> tuple:
        val = x[:, :d_value]
        valid = x[:, d_value:d_value + d_where]
        if d_where == 1:
            mask = valid.squeeze(-1) > 0.5
        else:
            # Multi-wide validity collapses to "any slot ≥ 0.5".
            mask = (valid > 0.5).any(dim=-1)
        rows = val[mask]
        if rows.shape[0] < 2:
            return True, ""
        diffs = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs().max(dim=-1).values
        eye = torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)
        bad = (diffs < margin) & ~eye
        if not bad.any():
            return True, ""
        i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"valid-subset rows {i},{j} within margin={margin}: "
            f"{rows[i].tolist()} vs {rows[j].tolist()}"
        )

    composite = Concatenate([value, where])
    wrapped = Assert(
        composite, predicate,
        message=f"distinct_across (margin={margin})",
    )
    from torchwright.graph import Linear
    proj = torch.zeros(d_value + d_where, d_value)
    for i in range(d_value):
        proj[i, i] = 1.0
    return Linear(wrapped, proj, name="distinct_across_value")


def assert_picked_from(
    result: Node, values: Node, keys: Node, *, atol: float = 1e-2,
) -> Node:
    """Assert an attention output matches exactly one valid per-position value row.

    For a pick-style attention (``attend_argmin_unmasked`` or
    ``attend_argmax_dot``), the output at every query row should equal
    ``values[k]`` for some key row ``k`` where ``keys ≈ 1`` — a blend
    of two or more values indicates a softmax that failed to
    concentrate.  That's exactly the angle-192 rendering artifact in
    DOOM: the compiled softmax returned a weighted average of walls
    instead of one wall.

    ``result`` and ``values`` must share width ``d`` (the attention
    op's contract — output width equals value width).  ``keys`` is a
    (n_pos, 1) validity flag; rows with ``keys > 0.5`` are treated
    as valid candidate picks.

    Predicate: for every query row ``q``, compute the L∞ distance from
    ``result[q]`` to every valid ``values[k]`` row; fail if the minimum
    exceeds ``atol``.  Default ``atol`` (1e-2) is loose enough to
    absorb PL-softmax jitter on clean picks and tight enough to catch
    the ~3% blend we saw at angle 192 (which produced 0.15-unit drift
    on magnitude-5 values).

    **Safe placement**: **post-attention, compiled-side only.**
    Reference math at high match-gain is exact; only the compiled
    piecewise-linear softmax drifts.  Expect this assert to fire
    through :func:`torchwright.debug.probe.check_asserts_on_compiled`,
    not during :func:`reference_eval`.
    """
    d_r = len(result)
    d_v = len(values)
    d_k = len(keys)
    if d_r != d_v:
        raise ValueError(
            f"assert_picked_from: result/values width mismatch "
            f"(result={d_r}, values={d_v})"
        )

    def predicate(x: torch.Tensor) -> tuple:
        res = x[:, :d_r]
        vals = x[:, d_r:d_r + d_v]
        keys_t = x[:, d_r + d_v:d_r + d_v + d_k]
        if d_k == 1:
            mask = keys_t.squeeze(-1) > 0.5
        else:
            mask = (keys_t > 0.5).any(dim=-1)
        valid_vals = vals[mask]
        if valid_vals.shape[0] == 0:
            return False, "no valid key positions"
        # (n_pos, 1, d) - (1, n_keys, d) → (n_pos, n_keys, d); max over d = L∞.
        dists = (res.unsqueeze(1) - valid_vals.unsqueeze(0)).abs().max(dim=-1).values
        min_dists, argmin_k = dists.min(dim=-1)
        bad = min_dists > atol
        if not bad.any():
            return True, ""
        q = bad.nonzero(as_tuple=False)[0].item()
        return False, (
            f"query row {q}: result doesn't match any value row "
            f"within atol={atol}; closest valid key #{argmin_k[q].item()} "
            f"at L∞ distance {min_dists[q].item():.4f}"
        )

    composite = Concatenate([result, values, keys])
    wrapped = Assert(
        composite, predicate,
        message=f"attention picked one (atol={atol})",
    )
    from torchwright.graph import Linear
    proj = torch.zeros(d_r + d_v + d_k, d_r)
    for i in range(d_r):
        proj[i, i] = 1.0
    return Linear(wrapped, proj, name="picked_from_result")
