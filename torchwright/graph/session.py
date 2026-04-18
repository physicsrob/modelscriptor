"""Graph session management for affine bound propagation.

A *graph session* scopes the set of ``InputNode`` instances that form
the basis for affine bounds.  Call ``fresh_graph_session()`` as a
context manager to create a new session; ``InputNode.__init__`` auto-
registers into the current session.

Construction outside any explicit session uses an implicit module-level
session (back-compat for REPL / simple test use).  Nested ``with`` is
an error.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from torchwright.graph.misc import InputNode


class ValueTypeNotFinalized(RuntimeError):
    """Raised when accessing affine bounds before ``finalize()``."""


class GraphFrozen(RuntimeError):
    """Raised when mutating a session after ``finalize()``."""


class GraphSession:
    """Holds state for one graph-construction session."""

    def __init__(self) -> None:
        self.input_nodes: List["InputNode"] = []
        self.frozen: bool = False

    def register_input(self, node: "InputNode") -> None:
        if self.frozen:
            raise GraphFrozen(
                "Cannot create new InputNode after finalize(). "
                "Start a fresh_graph_session() for a new graph."
            )
        self.input_nodes.append(node)

    def freeze(self) -> None:
        self.frozen = True


_implicit_session = GraphSession()

_current_session: ContextVar[Optional[GraphSession]] = ContextVar(
    "_current_session",
    default=None,
)


def current_session() -> GraphSession:
    """Return the active session, falling back to the implicit one."""
    s = _current_session.get()
    if s is not None:
        return s
    return _implicit_session


@contextmanager
def fresh_graph_session():
    """Context manager that creates a new, isolated graph session.

    All ``InputNode`` instances created inside the block register into
    this session.  Nested ``with fresh_graph_session():`` is an error.
    """
    if _current_session.get() is not None:
        raise RuntimeError("Nested fresh_graph_session() is not allowed")
    session = GraphSession()
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)
