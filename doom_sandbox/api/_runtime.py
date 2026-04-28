"""Internal: shared state used by primitives that must run at module load.

Module-level-only primitives (`pwl_def`, `constant`, and any future
ones) check `_FORWARD_RUNNING` to refuse construction during `forward()`.
The framework runtime sets this flag while a forward pass is in flight.

Not part of the agent-facing API — never re-exported through
`doom_sandbox.api`.
"""

from __future__ import annotations


# Set to True by the framework while forward() is running. While True,
# module-level-only constructors raise on instantiation. The sandbox
# is single-threaded — concurrent forward() calls in the same process
# are not supported.
_FORWARD_RUNNING: bool = False


# Monotonic counter assigned to each PWLDef / PWLDef2D at construction
# time. Used as a stable per-PWL identity for noise seeding so that
# identical inputs to the same PWLDef produce identical outputs within
# a process. Not stable across processes (depends on import order).
_PWL_ID_COUNTER: int = 0


def next_pwl_id() -> int:
    """Return the next PWL construction-order ID and advance the counter."""
    global _PWL_ID_COUNTER
    _PWL_ID_COUNTER += 1
    return _PWL_ID_COUNTER
