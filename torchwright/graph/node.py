from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Dict, Optional

import torch

from torchwright.graph.value_type import NodeValueType

global_node_id = 0

_current_annotation: ContextVar[Optional[str]] = ContextVar(
    "current_annotation", default=None,
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

    def __init__(self, d_output: int, inputs: List["Node"], name: str = ""):
        global global_node_id
        self.d_output = d_output
        self.inputs = inputs
        self.node_id = global_node_id
        self.name = name
        self.annotation = _current_annotation.get()
        global_node_id += 1
        self._value_type = self.compute_value_type()

    @property
    def value_type(self) -> NodeValueType:
        return self._value_type

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
