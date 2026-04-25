from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SchedulingPolicy:
    """Controls how the forward compiler routes ops to attention vs MLP.

    The compiler maps graph nodes to two sublayer types: attention heads
    (cross-position communication via Q/K/V/O matrices) and MLP slots
    (position-local transforms via Linear1/ReLU/Linear2).  Many ops —
    standalone Linears, add_into, compute_add — are position-local
    and can use either mechanism.  This policy controls which they use.
    """

    # Whether position-local compute ops (standalone Linears, add_into,
    # compute_add) use attention heads or MLP bypass.
    #   "always": use attention heads (legacy behavior)
    #   "never":  use MLP bypass, defer to next layer if MLP full
    local_in_attention: Literal["always", "never"] = "never"

    # Residual stream occupancy fraction above which the scheduler switches
    # its sort order from "longest critical path first" to "free columns
    # first."  Under pressure, freeing residual columns (cancels and ops
    # whose output reclaims more columns than it allocates) gets prioritized
    # over the deepest-chain-first heuristic, to keep the residual stream
    # from filling up.
    pressure_threshold: float = 0.75


LEGACY_POLICY = SchedulingPolicy(
    local_in_attention="always",
)
