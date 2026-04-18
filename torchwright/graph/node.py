import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Dict, Optional, Set

import torch

from torchwright.graph.value_type import NodeValueType

global_node_id = 0


def _verify_tensor_against_value_type(node: "Node", tensor: torch.Tensor) -> None:
    """Assert the actual tensor conforms to the node's declared value_type.

    Used by the optional runtime verifier (``TW_VERIFY_VALUE_TYPES``).
    All violations raise ``AssertionError``.
    """
    vt = node.value_type
    if tensor.numel() == 0:
        return
    t = tensor.detach()
    name = f"{node.node_type()}(id={node.node_id}, name='{node.name}')"

    r = vt.value_range
    actual_lo = float(t.min().item())
    actual_hi = float(t.max().item())
    tol = 1e-4
    if actual_lo < r.lo - tol or actual_hi > r.hi + tol:
        raise AssertionError(
            f"{name}: value_range mismatch — declared {r}, "
            f"observed [{actual_lo}, {actual_hi}]"
        )
    if vt.is_integer:
        diff = (t - t.round()).abs().max().item()
        if diff > tol:
            raise AssertionError(
                f"{name}: declared is_integer but observed max deviation {diff}"
            )
    if vt.is_binary:
        if not torch.all((t.round() == 0) | (t.round() == 1)).item():
            raise AssertionError(
                f"{name}: declared is_binary but values outside {{0,1}}"
            )
    if vt.is_sign:
        if not torch.all((t.round() == -1) | (t.round() == 1)).item():
            raise AssertionError(
                f"{name}: declared is_sign but values outside {{-1,+1}}"
            )
    if vt.is_one_hot:
        sums = t.round().sum(dim=-1)
        if not torch.all(sums == 1).item():
            raise AssertionError(
                f"{name}: declared is_one_hot but per-row sum not equal to 1"
            )


_current_annotation: ContextVar[Optional[str]] = ContextVar(
    "current_annotation",
    default=None,
)


@contextmanager
def annotate(label: str):
    """Tag all nodes created inside this block with a hierarchical label.

    Nesting builds a ``/``-separated path::

        with annotate("render"):
            with annotate("texture"):
                # nodes get annotation = "render/texture"
    """
    current = _current_annotation.get()
    new = f"{current}/{label}" if current else label
    token = _current_annotation.set(new)
    try:
        yield
    finally:
        _current_annotation.reset(token)


class Node:
    """Base class for all computation graph nodes.

    Each node represents one operation in a dataflow graph that will be
    compiled into transformer weights. Nodes are connected by their
    ``inputs`` list (upstream dependencies).

    Attributes:
        d_output: Width of this node's output vector.
        inputs: Upstream nodes whose outputs feed into this node.
        node_id: Auto-incremented unique identifier.
        name: Optional human-readable label (for debugging / repr).
        annotation: Hierarchical label set by the ``annotate`` context manager.
    """

    inputs: List["Node"]
    d_output: int
    node_id: int
    name: str
    annotation: Optional[str]
    scheduling_predecessors: Set["Node"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original = cls.__dict__.get("compute")
        if original is None or getattr(original, "_tw_verified", False):
            return

        def wrapped(self, n_pos, input_values, *args, **kw):
            result = original(self, n_pos, input_values, *args, **kw)
            if os.environ.get("TW_VERIFY_VALUE_TYPES") and isinstance(
                result, torch.Tensor
            ):
                _verify_tensor_against_value_type(self, result)
            return result

        wrapped.__name__ = original.__name__
        wrapped.__qualname__ = original.__qualname__
        wrapped.__doc__ = original.__doc__
        wrapped._tw_verified = True
        cls.compute = wrapped

    def __init__(self, d_output: int, inputs: List["Node"], name: str = ""):
        global global_node_id
        self.d_output = d_output
        self.inputs = inputs
        self.node_id = global_node_id
        self.name = name
        self.annotation = _current_annotation.get()
        # Scheduling-only predecessors (not data inputs).  Populated by
        # ``torchwright.graph.scheduling_hints.sequential_scope`` and
        # similar helpers; honored by ``GraphAnalyzer.is_ready`` so the
        # node isn't scheduled until every listed predecessor is in
        # ``computed_nodes``.  Empty by default.
        self.scheduling_predecessors: Set["Node"] = set()
        global_node_id += 1
        self._value_type_eager = self.compute_value_type()
        from torchwright.graph.affine_rules import compute_affine_bound

        self._affine_bound = compute_affine_bound(self)
        affine_range = self._affine_bound.to_scalar_range()
        eager_range = self._value_type_eager.value_range
        tightened = eager_range.intersect(affine_range)
        if tightened != eager_range:
            from dataclasses import replace

            self._value_type_eager = replace(
                self._value_type_eager, value_range=tightened
            )

    @property
    def value_type(self) -> NodeValueType:
        return self._value_type_eager

    @property
    def _value_type(self) -> NodeValueType:
        return self._value_type_eager

    @_value_type.setter
    def _value_type(self, val: NodeValueType) -> None:
        self._value_type_eager = val

    @property
    def affine_bound(self):
        return self._affine_bound

    def compute_value_type(self) -> NodeValueType:
        """Return the static value-type of this node's output.

        Default: ``unknown()`` (fail-closed). Subclasses override to
        propagate properties from their inputs or declare constants.
        Called eagerly from ``__init__`` so rule errors surface at graph
        build time.
        """
        return NodeValueType.unknown()

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self):
        return self.d_output

    def replace_input(self, old_input: "Node", new_input: "Node"):
        for i, _ in enumerate(self.inputs):
            if self.inputs[i] == old_input:
                self.inputs[i] = new_input

    def node_type(self):
        return type(self).__name__

    def __repr__(self):
        type_name = self.node_type()
        if len(self.inputs) == 0:
            return f"{type_name}(id={self.node_id}, name='{self.name}', d={len(self)})"
        elif len(self.inputs) == 1:
            inp = self.inputs[0]
            inp_type_name = inp.node_type()
            return f"{type_name}(id={self.node_id}, name='{self.name}', inp={inp_type_name}(id={inp.node_id}, name='{inp.name}', d={len(inp)}), d={len(self)})"
        else:
            inp_strings = []
            for i, inp in enumerate(self.inputs):
                inp_type_name = inp.node_type()
                inp_strings.append(
                    f"inp{i}={inp_type_name}(id={inp.node_id}, name='{inp.name}', d={len(inp)})"
                )
            inp_str = ", ".join(inp_strings)
            return f"{type_name}(id={self.node_id}, name='{self.name}', {inp_str}, d={len(self)})"

    def num_params(self):
        return 0

    def __eq__(self, other):
        return self.node_id == other.node_id

    def __hash__(self):
        return self.node_id
