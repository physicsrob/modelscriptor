"""Contract helpers for primitives that require specific ``NodeValueType``
properties on their inputs.

Each ``require_*`` raises ``TypeError`` at graph-construction time with a
message naming the caller, the offending node's op type, and the
inferred ``NodeValueType``. This pushes contract violations to the point
where they're introduced rather than surfacing as numerical weirdness
later.
"""

from __future__ import annotations

from torchwright.graph.node import Node


def _fmt(node: Node, caller: str, expected: str) -> str:
    return (
        f"{caller}: node {node.node_type()}(id={node.node_id}, name='{node.name}') "
        f"must be {expected}; got value_type={node.value_type}"
    )


def require_integer(node: Node, caller: str) -> None:
    if not node.value_type.is_integer:
        raise TypeError(_fmt(node, caller, "integer-valued"))


def require_binary(node: Node, caller: str) -> None:
    if not node.value_type.is_binary:
        raise TypeError(_fmt(node, caller, "binary (elements in {0, 1})"))


def require_sign(node: Node, caller: str) -> None:
    if not node.value_type.is_sign:
        raise TypeError(_fmt(node, caller, "sign-valued (elements in {-1, +1})"))


def require_one_hot(node: Node, caller: str) -> None:
    if not node.value_type.is_one_hot:
        raise TypeError(_fmt(node, caller, "one-hot"))
