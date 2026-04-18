"""Post-construction finalization pass for affine bound propagation.

``finalize(root)`` walks the graph topologically, builds the shared
``Basis`` from all reachable ``InputNode`` instances, and computes an
``AffineBound`` for every node. The affine-derived interval is used to
populate ``_value_type_affine`` with tightened ranges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Set

from torchwright.graph.affine_bound import AffineBound
from torchwright.graph.basis import Basis

if TYPE_CHECKING:
    from torchwright.graph.node import Node


def finalize(root: "Node") -> "Node":
    """Compute affine bounds for every node reachable from *root*.

    Idempotent: if every node already has ``_affine_bound is not None``,
    returns immediately.

    Returns the effective root — which may differ from *root* if root
    was a ``ConsumerPlaceholder`` that got materialized.
    """
    from torchwright.graph.misc import InputNode
    from torchwright.graph.placeholders import ConsumerPlaceholder

    all_nodes = _get_ancestor_nodes({root})
    if _is_finalized(all_nodes):
        return root

    input_nodes = sorted(
        [n for n in all_nodes if isinstance(n, InputNode)],
        key=lambda n: n.node_id,
    )
    basis = Basis.from_input_nodes(input_nodes)

    topo = _topo_sort(all_nodes)

    # Step 1: Compute affine bounds for non-placeholder nodes whose inputs
    # are all already computed (skip nodes downstream of placeholders)
    ph_set = {n for n in topo if isinstance(n, ConsumerPlaceholder)}
    for node in topo:
        if isinstance(node, ConsumerPlaceholder):
            continue
        if any(
            inp in ph_set or inp._affine_bound is None
            for inp in node.inputs
            if inp in all_nodes
        ):
            continue
        node._affine_bound = compute_affine_bound(node, basis)
        _set_affine_value_type(node)

    # Step 2: Materialize placeholders in topo order, rebind downstream.
    # After each materialization, recompute affine bounds for all reachable
    # nodes that don't have them yet.
    placeholders = [n for n in topo if isinstance(n, ConsumerPlaceholder)]
    effective_root = root
    remaining_phs = set(placeholders)
    for ph in placeholders:
        remaining_phs.discard(ph)
        output = ph.materialize()
        _rebind_downstream(ph, output, all_nodes)
        if ph is root:
            effective_root = output
        _compute_all_ready(effective_root, basis, remaining_phs)

    return effective_root


def _compute_all_ready(root: "Node", basis: Basis, skip: Set) -> None:
    """Compute affine bounds for all nodes reachable from root that are ready."""
    from torchwright.graph.placeholders import ConsumerPlaceholder

    all_nodes = _get_ancestor_nodes({root})
    topo = _topo_sort(all_nodes)
    for node in topo:
        if node._affine_bound is not None:
            continue
        if isinstance(node, ConsumerPlaceholder):
            continue
        if node in skip:
            continue
        if any(inp._affine_bound is None for inp in node.inputs):
            continue
        node._affine_bound = compute_affine_bound(node, basis)
        _set_affine_value_type(node)


def _rebind_downstream(old: "Node", new: "Node", all_nodes: Set["Node"]) -> None:
    """Rewire every consumer of *old* to point at *new* instead.

    Also updates the *all_nodes* set in place: removes *old*, adds *new*
    and all ancestors of *new* that aren't already present.
    """
    for node in all_nodes:
        for i, inp in enumerate(node.inputs):
            if inp is old:
                node.inputs[i] = new
    all_nodes.discard(old)
    new_ancestors = _get_ancestor_nodes({new})
    all_nodes.update(new_ancestors)


def _is_finalized(nodes: Set["Node"]) -> bool:
    return all(getattr(n, "_affine_bound", None) is not None for n in nodes)


def _set_affine_value_type(node: "Node") -> None:
    """Populate ``_value_type_affine`` from the affine bound's interval."""
    from dataclasses import replace

    ab: AffineBound = node._affine_bound  # type: ignore[assignment]
    affine_range = ab.to_scalar_range()
    eager = node._value_type_eager
    eager_range = eager.value_range
    tightened = eager_range.intersect(affine_range)
    node._value_type_affine = replace(eager, value_range=tightened)


def compute_affine_bound(node: "Node", basis: Basis) -> AffineBound:
    """Dispatch to the appropriate affine rule for *node*."""
    from torchwright.graph.misc import InputNode, LiteralValue, Add, Concatenate, Assert
    from torchwright.graph.linear import Linear
    from torchwright.graph.relu import ReLU
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.pos_encoding import PosEncoding
    from torchwright.graph.attn import Attn
    from torchwright.graph.placeholders import ConsumerPlaceholder

    if isinstance(node, InputNode):
        return AffineBound.identity(basis, node)

    if isinstance(node, LiteralValue):
        return AffineBound.constant(
            basis, node.value.to(dtype=__import__("torch").float64)
        )

    if isinstance(node, Linear):
        return _linear_rule(node, basis)

    if isinstance(node, Add):
        return _add_rule(node, basis)

    if isinstance(node, Concatenate):
        return _concat_rule(node, basis)

    if isinstance(node, ReLU):
        return _relu_rule(node, basis)

    if isinstance(node, Assert):
        return _assert_rule(node, basis)

    if isinstance(node, Attn):
        return _attn_rule(node, basis)

    if isinstance(node, Embedding):
        return _embedding_rule(node, basis)

    if isinstance(node, PosEncoding):
        return AffineBound.degenerate(basis, node.d_output, lo=-1.0, hi=1.0)

    r = node._value_type_eager.value_range
    return AffineBound.degenerate(basis, node.d_output, lo=r.lo, hi=r.hi)


def _linear_rule(node, basis: Basis) -> AffineBound:
    """y = x @ W + c: sign-split GEMM."""
    import torch

    inp_ab = node.inputs[0]._affine_bound
    W = node.output_matrix.to(torch.float64)
    c = node.output_bias.to(torch.float64)
    W_plus = torch.clamp(W, min=0)
    W_minus = torch.clamp(W, max=0)

    A_lo = W_plus.T @ inp_ab.A_lo + W_minus.T @ inp_ab.A_hi
    b_lo = W_plus.T @ inp_ab.b_lo + W_minus.T @ inp_ab.b_hi + c
    A_hi = W_plus.T @ inp_ab.A_hi + W_minus.T @ inp_ab.A_lo
    b_hi = W_plus.T @ inp_ab.b_hi + W_minus.T @ inp_ab.b_lo + c

    return AffineBound(A_lo=A_lo, A_hi=A_hi, b_lo=b_lo, b_hi=b_hi, basis=basis)


def _add_rule(node, basis: Basis) -> AffineBound:
    u = node.inputs[0]._affine_bound
    v = node.inputs[1]._affine_bound
    return AffineBound(
        A_lo=u.A_lo + v.A_lo,
        A_hi=u.A_hi + v.A_hi,
        b_lo=u.b_lo + v.b_lo,
        b_hi=u.b_hi + v.b_hi,
        basis=basis,
    )


def _concat_rule(node, basis: Basis) -> AffineBound:
    import torch

    parts_lo = [inp._affine_bound.A_lo for inp in node.inputs]
    parts_hi = [inp._affine_bound.A_hi for inp in node.inputs]
    parts_blo = [inp._affine_bound.b_lo for inp in node.inputs]
    parts_bhi = [inp._affine_bound.b_hi for inp in node.inputs]

    return AffineBound(
        A_lo=torch.cat(parts_lo, dim=0),
        A_hi=torch.cat(parts_hi, dim=0),
        b_lo=torch.cat(parts_blo, dim=0),
        b_hi=torch.cat(parts_bhi, dim=0),
        basis=basis,
    )


def _relu_rule(node, basis: Basis) -> AffineBound:
    """ReLU per-component case analysis using linear envelope."""
    import torch

    inp_ab = node.inputs[0]._affine_bound
    intervals = inp_ab.to_interval()
    d = node.d_output
    n = basis.n

    A_lo = torch.zeros(d, n, dtype=torch.float64)
    A_hi = torch.zeros(d, n, dtype=torch.float64)
    b_lo = torch.zeros(d, dtype=torch.float64)
    b_hi = torch.zeros(d, dtype=torch.float64)

    for i in range(d):
        l, h = intervals[i].lo, intervals[i].hi
        if l >= 0:
            A_lo[i] = inp_ab.A_lo[i]
            A_hi[i] = inp_ab.A_hi[i]
            b_lo[i] = inp_ab.b_lo[i]
            b_hi[i] = inp_ab.b_hi[i]
        elif h <= 0:
            pass  # zeros already
        else:
            # Upper bound: chord slope = h / (h - l)
            slope = h / (h - l)
            A_hi[i] = slope * inp_ab.A_hi[i]
            b_hi[i] = slope * (inp_ab.b_hi[i] - l)
            # Lower bound: alpha = 1.0 if h >= -l else 0.0
            alpha = 1.0 if h >= -l else 0.0
            A_lo[i] = alpha * inp_ab.A_lo[i]
            b_lo[i] = alpha * inp_ab.b_lo[i]

    return AffineBound(A_lo=A_lo, A_hi=A_hi, b_lo=b_lo, b_hi=b_hi, basis=basis)


def _assert_rule(node, basis: Basis) -> AffineBound:
    """Assert: pass through the wrapped node's coefficients unchanged."""
    return node.inputs[0]._affine_bound


def _attn_rule(node, basis: Basis) -> AffineBound:
    """Attn three-step degenerate rule.

    1. Propagate value_in's AffineBound through value_matrix (sign-split Linear rule).
    2. to_interval() → per-component [vproj_lo, vproj_hi].
    3. Sign-split [vproj_lo, vproj_hi] through output_matrix on scalar intervals
       → emit degenerate AffineBound (zero A matrices).
    """
    import torch

    value_ab = node.inputs[2]._affine_bound
    V = node.value_matrix.to(torch.float64)
    O = node.output_matrix.to(torch.float64)

    V_plus = torch.clamp(V, min=0)
    V_minus = torch.clamp(V, max=0)
    proj_A_lo = V_plus.T @ value_ab.A_lo + V_minus.T @ value_ab.A_hi
    proj_b_lo = V_plus.T @ value_ab.b_lo + V_minus.T @ value_ab.b_hi
    proj_A_hi = V_plus.T @ value_ab.A_hi + V_minus.T @ value_ab.A_lo
    proj_b_hi = V_plus.T @ value_ab.b_hi + V_minus.T @ value_ab.b_lo

    proj_ab = AffineBound(
        A_lo=proj_A_lo,
        A_hi=proj_A_hi,
        b_lo=proj_b_lo,
        b_hi=proj_b_hi,
        basis=basis,
    )
    proj_intervals = proj_ab.to_interval()

    d_v = len(proj_intervals)
    vlo = torch.tensor([iv.lo for iv in proj_intervals], dtype=torch.float64)
    vhi = torch.tensor([iv.hi for iv in proj_intervals], dtype=torch.float64)

    O_plus = torch.clamp(O, min=0)
    O_minus = torch.clamp(O, max=0)
    out_lo = O_plus.T @ vlo + O_minus.T @ vhi
    out_hi = O_plus.T @ vhi + O_minus.T @ vlo

    return AffineBound.degenerate(
        basis,
        node.d_output,
        lo=float(out_lo.min().item()),
        hi=float(out_hi.max().item()),
    )


def _embedding_rule(node, basis: Basis) -> AffineBound:
    """Degenerate with per-component min/max over table rows."""
    import torch

    t = node.table.to(torch.float64)
    if t.numel() == 0:
        return AffineBound.degenerate(basis, node.d_output)
    lo = float(t.min().item())
    hi = float(t.max().item())
    return AffineBound.degenerate(basis, node.d_output, lo=lo, hi=hi)


def _get_ancestor_nodes(start_nodes: Set["Node"]) -> Set["Node"]:
    result = set(start_nodes)
    queue = list(start_nodes)
    while queue:
        node = queue.pop()
        for inp in node.inputs:
            if inp not in result:
                result.add(inp)
                queue.append(inp)
    return result


def _topo_sort(nodes: Set["Node"]) -> "List[Node]":
    """Kahn's algorithm — returns nodes with inputs before dependents."""
    in_degree: Dict[int, int] = {}
    consumers: Dict[int, List] = defaultdict(list)
    for node in nodes:
        in_degree.setdefault(node.node_id, 0)
        for inp in set(node.inputs):
            if inp in nodes:
                in_degree[node.node_id] = in_degree.get(node.node_id, 0) + 1
                consumers[inp.node_id].append(node)

    queue = deque(
        sorted(
            [n for n in nodes if in_degree.get(n.node_id, 0) == 0],
            key=lambda n: n.node_id,
        )
    )
    result: List["Node"] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for consumer in consumers.get(node.node_id, []):
            in_degree[consumer.node_id] -= 1
            if in_degree[consumer.node_id] == 0:
                queue.append(consumer)
    return result
