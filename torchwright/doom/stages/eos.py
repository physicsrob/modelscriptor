"""EOS stage: collision resolution + resolved-state broadcast.

At the single EOS token the graph:

1. Aggregates the per-WALL collision flags via ``attend_mean_where`` —
   if *any* wall was hit along the full or axis-only rays, the averaged
   flag exceeds a small threshold.
2. Resolves the new player position using axis-separated wall sliding:
   the player moves on an axis only if neither the full ray nor that
   axis's lone ray hit anything.
3. Broadcasts the resolved ``(x, y, angle)`` triple to every position
   via another ``attend_mean_where``, so SORTED tokens can read the
   post-collision player state without the host feeding it.
"""

from dataclasses import dataclass

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add,
    bool_to_01,
    compare,
    negate,
)
from torchwright.ops.attention_ops import attend_mean_where
from torchwright.ops.logic_ops import bool_any_true
from torchwright.ops.map_select import select

from torchwright.doom.graph_utils import extract_from
from torchwright.doom.stages.wall import CollisionFlags

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class EosInputs:
    # Token-type flags.
    is_wall: Node  # 1.0 at WALL positions (for collision aggregation)
    is_eos: Node  # 1.0 at the single EOS position (for state broadcast)

    # Per-wall collision flags from the WALL stage.
    collision: CollisionFlags

    # Host-fed pre-update player position.
    player_x: Node
    player_y: Node

    # Broadcast values from the INPUT stage.
    vel_dx: Node
    vel_dy: Node
    new_angle: Node

    pos_encoding: PosEncoding


@dataclass
class EosOutputs:
    # Resolved position at the EOS token — emitted as part of the EOS output
    # seed for the autoregressive sort loop.
    resolved_x: Node
    resolved_y: Node

    # Post-broadcast state available at every position for the SORTED stage.
    px: Node
    py: Node
    angle: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_eos(inputs: EosInputs) -> EosOutputs:
    with annotate("eos/collision_resolve"):
        resolved_x, resolved_y = _resolve_collision(inputs)

    with annotate("eos/attention"):
        px, py, angle = _broadcast_resolved_state(
            inputs.pos_encoding,
            inputs.is_eos,
            resolved_x,
            resolved_y,
            inputs.new_angle,
        )

    return EosOutputs(
        resolved_x=resolved_x,
        resolved_y=resolved_y,
        px=px,
        py=py,
        angle=angle,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _resolve_collision(inputs: EosInputs):
    """Axis-separated wall sliding from per-WALL hit flags.

    Aggregates ``hit_*`` across WALL positions via a mean; any mean above
    ``1/max_walls`` implies at least one hit.  The resolved position on
    each axis uses the velocity component if *neither* the full ray nor
    that axis's lone ray hit a wall; otherwise the player stays put on
    that axis.
    """
    hit_full_01 = bool_to_01(inputs.collision.hit_full)
    hit_x_01 = bool_to_01(inputs.collision.hit_x)
    hit_y_01 = bool_to_01(inputs.collision.hit_y)

    resolve_attn = attend_mean_where(
        inputs.pos_encoding,
        validity=inputs.is_wall,
        value=Concatenate([hit_full_01, hit_x_01, hit_y_01]),
    )
    avg_hf = extract_from(resolve_attn, 3, 0, 1, "avg_hf")
    avg_hx = extract_from(resolve_attn, 3, 1, 1, "avg_hx")
    avg_hy = extract_from(resolve_attn, 3, 2, 1, "avg_hy")

    any_hit_full = compare(avg_hf, 0.05)
    any_hit_x = compare(avg_hx, 0.05)
    any_hit_y = compare(avg_hy, 0.05)

    use_new_x = bool_any_true([negate(any_hit_full), negate(any_hit_x)])
    use_new_y = bool_any_true([negate(any_hit_full), negate(any_hit_y)])

    new_x = add(inputs.player_x, inputs.vel_dx)
    new_y = add(inputs.player_y, inputs.vel_dy)
    resolved_x = select(use_new_x, new_x, inputs.player_x)
    resolved_y = select(use_new_y, new_y, inputs.player_y)
    return resolved_x, resolved_y


def _broadcast_resolved_state(
    pos_encoding: PosEncoding,
    is_eos: Node,
    resolved_x: Node,
    resolved_y: Node,
    new_angle: Node,
):
    """Copy the EOS-position resolved (x, y, angle) triple to all positions.

    Since exactly one position has is_eos=1, the attention's mean is the
    value at that single position.
    """
    eos_state_attn = attend_mean_where(
        pos_encoding,
        validity=is_eos,
        value=Concatenate([resolved_x, resolved_y, new_angle]),
    )
    px = extract_from(eos_state_attn, 3, 0, 1, "eos_px")
    py = extract_from(eos_state_attn, 3, 1, 1, "eos_py")
    angle = extract_from(eos_state_attn, 3, 2, 1, "eos_angle")
    return px, py, angle
