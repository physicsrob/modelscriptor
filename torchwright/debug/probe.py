"""Graph-vs-compiled divergence probe.

Runs a compiled ``HeadlessTransformer`` side-by-side with a direct,
recursive evaluation of its source graph (the oracle: ``node.compute``)
and reports the first graph node in topological order whose compiled
value disagrees with the reference beyond a numeric tolerance.

The probe relies on the per-sublayer column snapshots that
:func:`torchwright.compiler.forward.compile.forward_compile` writes into
``HeadlessTransformer.residual_assignment`` — one snapshot per
post-MLP sublayer state.  For each graph node the probe picks the
earliest state where the node is materialised and extracts its
compiled value from that sublayer's residual-stream tensor.

Scope and limits:

* Single-position (non-autoregressive) only.  Cross-position attention
  is evaluated by the oracle (``Attn.compute`` runs the full softmax
  matmul), so multi-position graphs do produce a correct oracle value,
  but a stateful decode-protocol bug — KV cache trimming, ``past_len``
  drift, etc. — would still hide behind the compiled module's
  ``forward()`` path used here.
* The oracle uses class-level monkey-patching of ``Node.compute`` to
  memoise each node's value.  The patches are restored in a ``finally``
  block; the probe is not thread-safe.
* Nodes whose columns never survive to the final ``out_state`` can
  still be checked as long as they appeared in one of the per-sublayer
  snapshots — this is what lets us localise a bug to the exact layer
  that broke it rather than only the top output.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from torchwright.compiler.export import CompiledHeadless, compile_headless
from torchwright.compiler.forward.graph_analysis import GraphAnalyzer
from torchwright.compiler.residual_assignment import (
    ResidualAssignment,
    ResidualStreamState,
)
from torchwright.compiler.transformer import HeadlessTransformer
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.graph import Concatenate, Node
from torchwright.graph.misc import Assert, InputNode, LiteralValue, Placeholder
from torchwright.graph.pos_encoding import PosEncoding


# ---------------------------------------------------------------------------
# Oracle: memoised recursive evaluator
# ---------------------------------------------------------------------------


def reference_eval(
    output_node: Node,
    input_values: Dict[str, torch.Tensor],
    n_pos: int,
) -> Dict[Node, torch.Tensor]:
    """Recursively evaluate the graph and return ``{Node: tensor}``.

    Walks the graph via ``node.compute`` — the same method the compiler
    already trusts as the semantic definition of each node.  Each node
    is computed exactly once via class-level ``compute`` monkey-patches
    that consult a shared cache: the patches intercept every recursive
    ``self.inputs[i].compute`` call, so the recursion collapses to
    O(n) torch ops over the graph's n nodes instead of the O(n²) the
    un-memoised recursion would pay.

    Args:
        output_node: the root of the subgraph to evaluate.
        input_values: ``{input_name: (n_pos, d_input) tensor}`` — one
            entry per :class:`InputNode` reachable from ``output_node``.
        n_pos: number of positions to evaluate at.

    Returns:
        A dict mapping every reachable node (including ``output_node``
        itself) to its oracle value as an ``(n_pos, node.d_output)``
        tensor.
    """
    cache: Dict[Node, torch.Tensor] = {}

    # Collect all node subclasses reachable from the output graph so we
    # only patch classes actually in use.  Walking by class lets us
    # restore every patch in a tight finally block even if compute()
    # raises mid-run.
    #
    # We walk via ``get_ancestor_nodes`` rather than ``GraphAnalyzer``
    # because ``GraphAnalyzer`` strips ``Assert`` nodes in-place — the
    # oracle pass must still see them so their predicates fire.
    all_nodes = get_ancestor_nodes({output_node})
    classes_in_graph = {type(n) for n in all_nodes}

    def _make_cached(orig_compute):
        def wrapped(self, n_pos_arg, input_values_arg):
            hit = cache.get(self)
            if hit is not None:
                return hit
            val = orig_compute(self, n_pos_arg, input_values_arg)
            cache[self] = val
            return val

        return wrapped

    patched: List[Tuple[type, Any]] = []
    try:
        for cls in classes_in_graph:
            if "compute" in cls.__dict__:
                orig = cls.__dict__["compute"]
                patched.append((cls, orig))
                cls.compute = _make_cached(orig)
        output_node.compute(n_pos, input_values)
    finally:
        for cls, orig in patched:
            cls.compute = orig

    return cache


# ---------------------------------------------------------------------------
# Probe report
# ---------------------------------------------------------------------------


@dataclass
class NodeDivergence:
    """Per-node divergence record."""

    node: Node
    state: ResidualStreamState
    max_abs_error: float
    compiled_mean: float
    oracle_mean: float
    compiled_min: float
    compiled_max: float
    oracle_min: float
    oracle_max: float

    def summary(self) -> str:
        return (
            f"{type(self.node).__name__}(id={self.node.node_id}, "
            f"name='{self.node.name}', d={self.node.d_output}) "
            f"at {self.state.name or f'state_{self.state.state_id}'}: "
            f"max_abs_err={self.max_abs_error:.4g} "
            f"(compiled mean={self.compiled_mean:.4g} "
            f"range=[{self.compiled_min:.4g}, {self.compiled_max:.4g}]; "
            f"oracle mean={self.oracle_mean:.4g} "
            f"range=[{self.oracle_min:.4g}, {self.oracle_max:.4g}])"
        )


@dataclass
class ProbeReport:
    """Structured result of a probe run."""

    #: Graph nodes ordered by topological rank, excluding nodes the
    #: probe cannot check (Concatenate groupings, nodes with no column
    #: assignment in any snapshot).
    nodes_checked: List[Node] = field(default_factory=list)

    #: Per-node divergence records, keyed by the node's topological
    #: index.  Every entry in ``nodes_checked`` has a record; the record
    #: is considered "divergent" if ``max_abs_error`` exceeds the
    #: probe's ``atol``.
    per_node: Dict[Node, NodeDivergence] = field(default_factory=dict)

    #: The first node in topological order whose compiled value
    #: exceeds ``atol``, or ``None`` if the probe found no divergence.
    first_divergent: Optional[NodeDivergence] = None

    #: Graph nodes the probe deliberately skipped, with a reason.
    skipped: Dict[Node, str] = field(default_factory=dict)

    #: Tolerance used for the "divergent" classification.
    atol: float = 1e-3

    def format_short(self, show_top_k: int = 10) -> str:
        lines = [
            f"ProbeReport: checked {len(self.nodes_checked)} nodes, "
            f"skipped {len(self.skipped)} (atol={self.atol:.2g})"
        ]
        if self.first_divergent is None:
            lines.append("  no divergence found")
            return "\n".join(lines)
        lines.append(f"  first divergent: {self.first_divergent.summary()}")
        ranked = sorted(
            self.per_node.values(), key=lambda r: -r.max_abs_error,
        )
        lines.append(f"  top-{show_top_k} by error magnitude:")
        for r in ranked[:show_top_k]:
            lines.append(f"    {r.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def _extract_compiled_value(
    node: Node,
    ra: ResidualAssignment,
    state: ResidualStreamState,
    res_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Pull a graph node's compiled value out of a post-sublayer residual stream.

    Returns ``None`` if the node is not materialised at ``state``.
    Concatenate nodes resolve to the concatenation of their children's
    columns via ``ResidualAssignment.get_node_indices``.
    """
    nodes_here = ra.get_nodes(state)
    if node not in nodes_here and not isinstance(node, Concatenate):
        return None
    try:
        cols = ra.get_node_indices(state, node)
    except KeyError:
        return None
    if not cols:
        return None
    return res_tensor[:, cols]


def _first_state_with(node: Node, ra: ResidualAssignment,
                      ordered_states: List[ResidualStreamState]
                      ) -> Optional[ResidualStreamState]:
    """Earliest sublayer state in which ``node`` is materialised."""
    if isinstance(node, Concatenate):
        # Concatenate is resolved transparently — pick the earliest
        # state where *all* of its leaves are present.
        children = [i for i in node.inputs]
        best: Optional[ResidualStreamState] = None
        for st in ordered_states:
            if all(
                _first_state_with(c, ra, [st]) is not None for c in children
            ):
                best = st
                break
        return best
    for st in ordered_states:
        if ra.has_node(st, node):
            return st
    return None


def probe_compiled(
    compiled: CompiledHeadless,
    output_node: Node,
    input_values: Dict[str, torch.Tensor],
    n_pos: int,
    atol: float = 1e-3,
) -> ProbeReport:
    """Run a divergence probe against an already-compiled module.

    Args:
        compiled: a :class:`CompiledHeadless` with per-sublayer
            snapshots on its underlying
            ``HeadlessTransformer.residual_assignment``.
        output_node: the same graph node that was passed to the
            compiler.  Used both as the oracle root and as the
            topological-order anchor.
        input_values: ``{input_name: (n_pos, d_input) tensor}``.
        n_pos: number of positions to run the probe on.
        atol: max absolute error below which a per-node comparison is
            treated as clean.  Raise for noisier op realisations.

    Returns:
        A populated :class:`ProbeReport`.
    """
    net: HeadlessTransformer = compiled._net
    ra = net.residual_assignment
    assert ra is not None, "compiled module has no residual_assignment"

    # Oracle first — cheap and deterministic.
    oracle = reference_eval(output_node, input_values, n_pos)

    # Run compiled forward with per-sublayer state capture.  The
    # compiled path needs the initial residual stream built the same
    # way CompiledHeadless.__call__ builds it.
    res_stream = compiled._build_res_stream(
        _inputs_from_dict(compiled, input_values, n_pos), past_len=0,
    )
    _, all_states = net.forward(res_stream, return_states=True)

    # Map from ResidualStreamState to (tensor, pretty_key).
    state_tensor: Dict[ResidualStreamState, Tuple[torch.Tensor, str]] = {}
    for key, (state, tensor) in all_states.items():
        state_tensor[state] = (tensor, key)

    # Ordered list of sublayer end-states in execution order.  We only
    # need states the compiler actually recorded assignments for — the
    # top-level in_state/out_state, plus every per-layer MLP out_state.
    ordered_states: List[ResidualStreamState] = []
    for i, layer in enumerate(net.layers):
        if layer.mlp.out_state in ra.mapping:
            ordered_states.append(layer.mlp.out_state)
    if net.layers[-1].mlp.out_state not in ordered_states:
        ordered_states.append(net.layers[-1].mlp.out_state)

    graph = GraphAnalyzer(output_node)
    report = ProbeReport(atol=atol)

    for node in graph.get_topological_order():
        if isinstance(node, (InputNode, LiteralValue, PosEncoding, Placeholder)):
            report.skipped[node] = "input/literal/pos_encoding/placeholder"
            continue
        if isinstance(node, Concatenate):
            report.skipped[node] = "concat grouping — leaves are checked individually"
            continue

        state = _first_state_with(node, ra, ordered_states)
        if state is None:
            report.skipped[node] = "no residual assignment found in any captured state"
            continue

        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            report.skipped[node] = (
                f"state {state.name or state.state_id} was assigned in "
                f"residual_assignment but not produced by forward(return_states=True)"
            )
            continue
        res_tensor, _ = tensor_pair

        compiled_val = _extract_compiled_value(node, ra, state, res_tensor)
        if compiled_val is None:
            report.skipped[node] = "no columns allocated at chosen state"
            continue

        oracle_val = oracle.get(node)
        if oracle_val is None:
            report.skipped[node] = "oracle did not reach this node"
            continue

        if compiled_val.shape != oracle_val.shape:
            report.skipped[node] = (
                f"shape mismatch compiled={tuple(compiled_val.shape)} "
                f"oracle={tuple(oracle_val.shape)}"
            )
            continue

        diff = (compiled_val.detach().cpu() - oracle_val.detach().cpu()).abs()
        max_err = float(diff.max().item())
        rec = NodeDivergence(
            node=node,
            state=state,
            max_abs_error=max_err,
            compiled_mean=float(compiled_val.mean().item()),
            oracle_mean=float(oracle_val.mean().item()),
            compiled_min=float(compiled_val.min().item()),
            compiled_max=float(compiled_val.max().item()),
            oracle_min=float(oracle_val.min().item()),
            oracle_max=float(oracle_val.max().item()),
        )
        report.nodes_checked.append(node)
        report.per_node[node] = rec
        if report.first_divergent is None and max_err > atol:
            report.first_divergent = rec

    return report


def _inputs_from_dict(
    compiled: CompiledHeadless,
    input_values: Dict[str, torch.Tensor],
    n_pos: int,
) -> torch.Tensor:
    """Pack an input-name → tensor dict into the flat row-tensor layout
    the compiled module's ``_build_res_stream`` expects.
    """
    d_input = max(start + width for _, start, width in compiled._input_specs)
    out = torch.zeros(n_pos, d_input)
    for name, start, width in compiled._input_specs:
        if name not in input_values:
            raise ValueError(f"probe missing input '{name}'")
        out[:, start : start + width] = input_values[name]
    return out


def probe_graph(
    output_node: Node,
    pos_encoding: PosEncoding,
    input_values: Dict[str, torch.Tensor],
    n_pos: int,
    d: int = 1024,
    d_head: int = 16,
    d_hidden: Optional[int] = None,
    max_layers: int = 200,
    atol: float = 1e-3,
    verbose: bool = False,
) -> ProbeReport:
    """Compile a graph and run the probe in one call.

    Convenience wrapper around ``compile_headless`` + ``probe_compiled``
    for the usual "I have a graph and I want to know where it breaks"
    workflow.  The compilation uses the same defaults as
    :func:`torchwright.compiler.forward.compile.forward_compile`.
    """
    compiled = compile_headless(
        output_node,
        pos_encoding,
        d=d,
        d_head=d_head,
        max_layers=max_layers,
        verbose=verbose,
        d_hidden=d_hidden,
    )
    return probe_compiled(
        compiled, output_node, input_values, n_pos, atol=atol,
    )


# ---------------------------------------------------------------------------
# Compiled-side Assert checks
# ---------------------------------------------------------------------------


def check_asserts_on_compiled(
    compiled: CompiledHeadless,
    asserts: List[Assert],
    input_values: Dict[str, torch.Tensor],
    n_pos: int,
) -> None:
    """Run each Assert's predicate against the compiled transformer's residual stream.

    Complements the reference-eval check (which runs predicates as
    ``Assert.compute`` is called during the oracle walk).  Here we run
    the same predicates against the *compiled* values of each Assert's
    input node — catching invariants that reference math satisfies but
    compiled approximations violate.

    ``asserts`` must have been collected via
    ``torchwright.graph.asserts.collect_asserts(output_node)`` **before**
    ``compile_headless`` was called, since compilation strips Asserts
    from the graph.

    Raises ``AssertionError`` on the first violation, with the same
    annotation-tagged message format as ``Assert.compute``.  Asserts
    whose input nodes have no residual assignment (e.g. pure-literal
    subgraphs) are silently skipped — they have no compiled value to
    check.
    """
    if not asserts:
        return

    net: HeadlessTransformer = compiled._net
    ra = net.residual_assignment
    assert ra is not None, "compiled module has no residual_assignment"

    res_stream = compiled._build_res_stream(
        _inputs_from_dict(compiled, input_values, n_pos), past_len=0,
    )
    _, all_states = net.forward(res_stream, return_states=True)

    state_tensor: Dict[ResidualStreamState, Tuple[torch.Tensor, str]] = {}
    for key, (state, tensor) in all_states.items():
        state_tensor[state] = (tensor, key)

    ordered_states: List[ResidualStreamState] = []
    for layer in net.layers:
        if layer.mlp.out_state in ra.mapping:
            ordered_states.append(layer.mlp.out_state)
    if net.layers[-1].mlp.out_state not in ordered_states:
        ordered_states.append(net.layers[-1].mlp.out_state)

    for assert_node in asserts:
        # If an assert wraps another assert (e.g. a user's outer
        # ``assert_in_range(y, ...)`` wrapping an op that internally
        # asserted its output range), the direct ``.inputs[0]`` is the
        # inner Assert node — stripped at compile time with no residual
        # state.  Unwrap the chain so we look up the innermost non-Assert
        # target, which does have a state.
        target = assert_node.inputs[0]
        while isinstance(target, Assert):
            target = target.inputs[0]
        state = _first_state_with(target, ra, ordered_states)
        if state is None:
            continue  # no residual assignment — can't check on compiled.
        tensor_pair = state_tensor.get(state)
        if tensor_pair is None:
            continue
        res_tensor, _ = tensor_pair
        compiled_val = _extract_compiled_value(target, ra, state, res_tensor)
        if compiled_val is None:
            continue
        assert_node._check(compiled_val)
