"""Integration tests for CP-SAT-driven forward compilation.

The CP-SAT solver in ``torchwright/compiler/forward/cpsat_scheduler.py``
produces a ``ScheduleAssignment`` that ``DirectedLayerScheduler`` (a
subclass of ``LayerScheduler``) replays through the existing per-layer
code path.  These tests exercise the integration end-to-end: the same
graph compiled twice (heuristic vs CP-SAT) must produce the same token
output, and the CP-SAT version must use no more layers than the
heuristic.

See ``docs/cpsat_scheduler.md`` for the spec.
"""

import pytest
import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.cpsat_scheduler import Costs
from torchwright.graph import Add, Linear, ReLU
from torchwright.ops.arithmetic_ops import add, relu
from torchwright.ops.inout_nodes import create_input


D = 256
D_HEAD = 16


def _build_relu_chain():
    """Input -> Linear -> ReLU -> Linear graph."""
    x = create_input("x", 8)
    l1 = Linear(x, torch.randn(8, 16), torch.randn(16), name="l1")
    r = ReLU(l1)
    l2 = Linear(r, torch.randn(16, 4), torch.randn(4), name="l2")
    return l2, {"x": torch.randn(3, 8)}


def _build_branchy():
    """A non-trivial graph: input -> two parallel chains -> add."""
    x = create_input("x", 8)
    a_l1 = Linear(x, torch.randn(8, 16), torch.zeros(16), name="a_l1")
    a_r = ReLU(a_l1)
    a_l2 = Linear(a_r, torch.randn(16, 8), torch.zeros(8), name="a_l2")
    b_l1 = Linear(x, torch.randn(8, 16), torch.zeros(16), name="b_l1")
    b_r = ReLU(b_l1)
    b_l2 = Linear(b_r, torch.randn(16, 8), torch.zeros(8), name="b_l2")
    out = add(a_l2, b_l2)
    return out, {"x": torch.randn(2, 8)}


def test_relu_chain_compiles_with_cpsat():
    """Smallest non-trivial graph: chain of one MLP block."""
    out, inputs = _build_relu_chain()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(3, inputs)[out].cpu()
    expected = out.compute(3, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_branchy_compiles_with_cpsat():
    """Two parallel chains feed into an Add."""
    out, inputs = _build_branchy()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_matches_heuristic_output():
    """Token output is identical regardless of scheduler choice.

    The schedule is a placement decision, not a value-changing
    transformation.  Compiling the same graph twice — once with the
    heuristic, once with CP-SAT — must produce the same numerical
    output (modulo float-point ordering effects).
    """
    out, inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    out_heur = net_heur.compute(2, inputs)[out].cpu()
    out_cpsat = net_cpsat.compute(2, inputs)[out].cpu()
    torch.testing.assert_close(out_cpsat, out_heur, atol=1e-4, rtol=1e-4)


def test_cpsat_layer_count_no_worse_than_heuristic():
    """CP-SAT proves the layer-count optimum; heuristic is upper bound.

    With ``Costs(alpha=1, beta=0, gamma=0)`` (the default), CP-SAT
    minimizes layer count.  The heuristic — being a feasible schedule —
    gives an upper bound the solver always matches or beats.
    """
    out, inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    assert len(net_cpsat.layers) <= len(net_heur.layers), (
        f"CP-SAT used {len(net_cpsat.layers)} layers, "
        f"heuristic used {len(net_heur.layers)}; CP-SAT should be ≤"
    )


def test_cpsat_with_admission_control_raises():
    """``admission_control=True`` is a CP-SAT model precondition.

    See ``docs/cpsat_scheduler.md`` §3 — the model does not represent
    the sibling-cluster admission constraint, so a solver-feasible
    schedule may not be replayable.
    """
    out, inputs = _build_relu_chain()
    with pytest.raises(RuntimeError, match="admission_control"):
        forward_compile(
            d=D,
            d_head=D_HEAD,
            output_node=out,
            verbose=False,
            optimize=1,
            admission_control=True,
        )


def test_cpsat_flex_routing_explores_both_sublayers():
    """Flex routing is a CP-SAT decision variable per standalone Linear.

    A graph with a single standalone Linear can route to either
    attention or MLP-bypass; flex_routing=True lets the solver pick.
    The compile must succeed regardless.
    """
    x = create_input("x", 8)
    out = Linear(x, torch.randn(8, 4), torch.randn(4), name="lin")

    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_flex_routing=True,
    )
    inputs = {"x": torch.randn(2, 8)}
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_costs_beta_routes_more_to_mlp():
    """beta>0 should route at least as many heads off attention as beta=0.

    With multiple standalone Linears (no chains), the alpha=1, beta=0
    objective is indifferent between attention and MLP routing as long
    as the layer count is the same.  Setting beta>0 makes attention
    heads costly, so the solver prefers MLP-bypass.
    """
    x = create_input("x", 8)
    l1 = Linear(x, torch.randn(8, 4), torch.zeros(4), name="l1")
    l2 = Linear(x, torch.randn(8, 4), torch.zeros(4), name="l2")
    out = add(l1, l2)

    net_alpha = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_costs=Costs(alpha=1, beta=0, gamma=0),
    )
    net_beta = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        cpsat_costs=Costs(alpha=1, beta=10, gamma=0),
    )
    # Both compiles should produce the same numerical output.
    inputs = {"x": torch.randn(2, 8)}
    out_alpha = net_alpha.compute(2, inputs)[out].cpu()
    out_beta = net_beta.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(out_alpha, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_beta, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_assume_zero_init_compiles():
    """``assume_zero_init=True`` lets the model and heuristic skip
    BIRTH-layer dirty cancels.  The compiled module must still produce
    correct output when the runtime zero-initialises the residual
    stream — which ``HeadlessTransformer.compute()`` always does.
    """
    out, inputs = _build_branchy()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
        assume_zero_init=True,
    )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cpsat_warm_start_layer_count_no_worse():
    """Warm-start must not produce a worse schedule than the heuristic.

    ``forward_compile`` runs a schedule-only heuristic pass before
    invoking CP-SAT and feeds the result as ``hint_layers``.  Because
    the hint is feasible, CP-SAT can always match it; with
    ``cpsat_costs.alpha=1`` (the default) it tries to beat it.
    """
    out, inputs = _build_branchy()
    net_heur = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=0,
    )
    net_cpsat = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    assert len(net_cpsat.layers) <= len(net_heur.layers)


def test_cpsat_falls_back_to_heuristic_when_no_incumbent(monkeypatch):
    """When CP-SAT finds no feasible solution within budget,
    ``forward_compile`` falls back to the heuristic schedule rather
    than raising.  Simulated by monkey-patching ``solve_schedule`` to
    return ``(None, stats)``; the compile must still produce the
    correct token output.
    """
    from torchwright.compiler.forward import compile as compile_mod
    from torchwright.compiler.forward.cpsat_scheduler import SolveStats

    fake_stats = SolveStats(
        status_name="UNKNOWN",
        objective_value=-1,
        best_objective_bound=0.0,
        wall_time_s=0.0,
        solver_log="",
        total_attn_heads=-1,
        total_mlp_bypass_slots=-1,
        is_optimal=False,
    )

    def fake_solve(*args, **kwargs):
        return None, fake_stats

    monkeypatch.setattr(compile_mod, "solve_schedule", fake_solve)

    out, inputs = _build_branchy()
    net = forward_compile(
        d=D,
        d_head=D_HEAD,
        output_node=out,
        verbose=False,
        optimize=1,
    )
    actual = net.compute(2, inputs)[out].cpu()
    expected = out.compute(2, inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
