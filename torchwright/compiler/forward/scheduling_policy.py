from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SchedulingPolicy:
    """Controls how the forward compiler routes ops to attention vs MLP.

    The compiler maps graph nodes to two sublayer types: attention heads
    (cross-position communication via Q/K/V/O matrices) and MLP slots
    (position-local transforms via Linear1/ReLU/Linear2).  Many ops —
    standalone Linears, add_into, compute_add, cancel — are position-local
    and can use either mechanism.  This policy controls which they use.
    """

    # Whether position-local compute ops (standalone Linears, add_into,
    # compute_add) use attention heads or MLP bypass.
    #   "always": use attention heads (legacy behavior)
    #   "never":  use MLP bypass, defer to next layer if MLP full
    #   "when_mlp_full": prefer MLP bypass, fall back to attention if full
    local_in_attention: Literal["always", "never", "when_mlp_full"] = "never"

    # Whether cancel ops (column zeroing) use attention heads or MLP bypass.
    # Cancel is cheap in attention (batched, ~1 head per d_head columns)
    # but expensive in MLP (2 bypass slots per column).  Deferring cancel
    # increases residual stream pressure in subsequent layers.
    #   "always": use attention heads (recommended)
    #   "never":  use MLP bypass, defer to next layer if MLP full
    #   "when_mlp_full": prefer MLP bypass, fall back to attention if full
    cancel_in_attention: Literal["always", "never", "when_mlp_full"] = "always"

    # MLP slot priority under normal conditions.  Controls scheduling order
    # when multiple op types compete for the d_hidden slot pool.
    #   "mandatory_first": ops requiring MLP (chains, standalone ReLUs)
    #       before ops preferring MLP (bypass).  Never starves nonlinear ops.
    #   "cleanup_first": cancel/freeing ops first, then mandatory, then
    #       bypass.  Minimizes residual stream pressure.
    #   "by_depth": all ops by critical path depth regardless of type.
    #       Optimal for minimizing layer count in theory.
    mlp_priority: Literal["mandatory_first", "cleanup_first", "by_depth"] = (
        "mandatory_first"
    )

    # MLP slot priority when residual stream occupancy exceeds
    # pressure_threshold.  Freeing columns becomes more urgent.
    mlp_priority_pressure: Literal["mandatory_first", "cleanup_first", "by_depth"] = (
        "cleanup_first"
    )

    # Residual stream occupancy fraction above which the scheduler
    # switches from mlp_priority to mlp_priority_pressure.
    pressure_threshold: float = 0.75


LEGACY_POLICY = SchedulingPolicy(
    local_in_attention="always",
    cancel_in_attention="always",
)
