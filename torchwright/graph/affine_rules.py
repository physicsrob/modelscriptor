"""Eager affine bound computation rules for each node type.

Called from ``Node.__init__`` to compute ``_affine_bound`` at
graph-construction time.  Each rule reads the already-set
``_affine_bound`` of its inputs (which are constructed first in
a dataflow graph) and produces the output bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchwright.graph.affine_bound import AffineBound

if TYPE_CHECKING:
    from torchwright.graph.node import Node


def compute_affine_bound(node: "Node") -> AffineBound:
    """Dispatch to the appropriate affine rule for *node*."""
    from torchwright.graph.misc import InputNode, LiteralValue, Add, Concatenate, Assert
    from torchwright.graph.linear import Linear
    from torchwright.graph.relu import ReLU
    from torchwright.graph.embedding import Embedding
    from torchwright.graph.pos_encoding import PosEncoding
    from torchwright.graph.attn import Attn

    if isinstance(node, InputNode):
        return AffineBound.identity(node)

    if isinstance(node, LiteralValue):
        import torch

        return AffineBound.constant(node.value.to(dtype=torch.float64))

    if isinstance(node, Linear):
        return _linear_rule(node)

    if isinstance(node, Add):
        return _add_rule(node)

    if isinstance(node, Concatenate):
        return _concat_rule(node)

    if isinstance(node, ReLU):
        return _relu_rule(node)

    if isinstance(node, Assert):
        return _assert_rule(node)

    if isinstance(node, Attn):
        return _attn_rule(node)

    if isinstance(node, Embedding):
        return _embedding_rule(node)

    if isinstance(node, PosEncoding):
        return AffineBound.degenerate(node.d_output, lo=-1.0, hi=1.0)

    r = node._value_type_eager.value_range
    return AffineBound.degenerate(node.d_output, lo=r.lo, hi=r.hi)


def _linear_rule(node) -> AffineBound:
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

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=inp_ab.columns,
        input_ranges=inp_ab.input_ranges,
    )


def _add_rule(node) -> AffineBound:
    u = node.inputs[0]._affine_bound
    v = node.inputs[1]._affine_bound
    if u.d_output != v.d_output:
        r = node._value_type_eager.value_range
        return AffineBound.degenerate(node.d_output, lo=r.lo, hi=r.hi)
    u, v = AffineBound.align(u, v)
    return AffineBound(
        A_lo=u.A_lo + v.A_lo,
        A_hi=u.A_hi + v.A_hi,
        b_lo=u.b_lo + v.b_lo,
        b_hi=u.b_hi + v.b_hi,
        columns=u.columns,
        input_ranges=u.input_ranges,
    )


def _concat_rule(node) -> AffineBound:
    import torch

    from torchwright.graph.affine_bound import _merge_layouts, _scatter

    bounds = [inp._affine_bound for inp in node.inputs]

    if len(bounds) == 1:
        return bounds[0]

    merged_columns, merged_ranges, n = _merge_layouts(*bounds)

    parts_lo, parts_hi, parts_blo, parts_bhi = [], [], [], []
    for b in bounds:
        scattered = _scatter(b, merged_columns, merged_ranges, n)
        parts_lo.append(scattered.A_lo)
        parts_hi.append(scattered.A_hi)
        parts_blo.append(scattered.b_lo)
        parts_bhi.append(scattered.b_hi)

    return AffineBound(
        A_lo=torch.cat(parts_lo, dim=0),
        A_hi=torch.cat(parts_hi, dim=0),
        b_lo=torch.cat(parts_blo, dim=0),
        b_hi=torch.cat(parts_bhi, dim=0),
        columns=merged_columns,
        input_ranges=merged_ranges,
    )


def _relu_rule(node) -> AffineBound:
    """ReLU per-component case analysis using linear envelope."""
    import torch

    inp_ab = node.inputs[0]._affine_bound
    intervals = inp_ab.to_interval()
    d = node.d_output
    n = inp_ab.n_cols

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
            pass
        else:
            slope = h / (h - l)
            A_hi[i] = slope * inp_ab.A_hi[i]
            b_hi[i] = slope * (inp_ab.b_hi[i] - l)
            alpha = 1.0 if h >= -l else 0.0
            A_lo[i] = alpha * inp_ab.A_lo[i]
            b_lo[i] = alpha * inp_ab.b_lo[i]

    return AffineBound(
        A_lo=A_lo,
        A_hi=A_hi,
        b_lo=b_lo,
        b_hi=b_hi,
        columns=inp_ab.columns,
        input_ranges=inp_ab.input_ranges,
    )


def _assert_rule(node) -> AffineBound:
    """Assert: pass through coefficients, optionally tighten input_ranges."""
    from torchwright.graph.misc import Assert, InputNode

    inp_ab = node.inputs[0]._affine_bound
    if node.claimed_type is not None:
        target = node.inputs[0]
        while isinstance(target, Assert):
            target = target.inputs[0]
        if isinstance(target, InputNode):
            claimed_range = node.claimed_type.value_range
            new_ranges = dict(inp_ab.input_ranges)
            if target.node_id in new_ranges:
                old_lo, old_hi = new_ranges[target.node_id]
                new_lo = max(old_lo, claimed_range.lo)
                new_hi = min(old_hi, claimed_range.hi)
                new_ranges[target.node_id] = (new_lo, new_hi)
                return AffineBound(
                    A_lo=inp_ab.A_lo,
                    A_hi=inp_ab.A_hi,
                    b_lo=inp_ab.b_lo,
                    b_hi=inp_ab.b_hi,
                    columns=inp_ab.columns,
                    input_ranges=new_ranges,
                )
    return inp_ab


def _attn_rule(node) -> AffineBound:
    """Attn three-step degenerate rule."""
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
        columns=value_ab.columns,
        input_ranges=value_ab.input_ranges,
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
        node.d_output,
        lo=float(out_lo.min().item()),
        hi=float(out_hi.max().item()),
    )


def _embedding_rule(node) -> AffineBound:
    """Degenerate with per-component min/max over table rows."""
    import torch

    t = node.table.to(torch.float64)
    if t.numel() == 0:
        return AffineBound.degenerate(node.d_output)
    lo = float(t.min().item())
    hi = float(t.max().item())
    return AffineBound.degenerate(node.d_output, lo=lo, hi=hi)
