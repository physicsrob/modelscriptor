"""EOS stage: collision resolution.

At the single EOS token the graph:

1. Aggregates the per-WALL collision flags via ``attend_mean_where`` —
   if *any* wall was hit along the full or axis-only rays, the averaged
   flag exceeds a small threshold.
2. Resolves the new player position using axis-separated wall sliding:
   the player moves on an axis only if neither the full ray nor that
   axis's lone ray hit anything.

The resolved ``(x, y)`` is emitted as overflow output.  The host reads
it and feeds it to the PLAYER tokens, which broadcast the resolved
state to all positions.
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
class EosToken:
    """Host-fed fields at the single EOS position."""

    player_x: Node  # pre-collision-resolution
    player_y: Node  # pre-collision-resolution


@dataclass
class EosKVInput:
    """Values arriving from other stages via attention."""

    collision: CollisionFlags  # from WallKVOutput
    vel_dx: Node  # from InputKVOutput
    vel_dy: Node  # from InputKVOutput


@dataclass
class EosTokenOutput:
    """Overflow outputs the host reads from the EOS position."""

    resolved_x: Node
    resolved_y: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_eos(
    token: EosToken,
    kv: EosKVInput,
    *,
    is_wall: Node,
    pos_encoding: PosEncoding,
) -> EosTokenOutput:
    with annotate("eos/collision_resolve"):
        resolved_x, resolved_y = _resolve_collision(token, kv, is_wall, pos_encoding)

    return EosTokenOutput(
        resolved_x=resolved_x,
        resolved_y=resolved_y,
    )


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------


def _resolve_collision(
    token: EosToken, kv: EosKVInput, is_wall: Node, pos_encoding: PosEncoding
):
    """Axis-separated wall sliding from per-WALL hit flags.

    Aggregates ``hit_*`` across WALL positions via a mean; any mean above
    ``1/max_walls`` implies at least one hit.  The resolved position on
    each axis uses the velocity component if *neither* the full ray nor
    that axis's lone ray hit a wall; otherwise the player stays put on
    that axis.
    """
    hit_full_01 = bool_to_01(kv.collision.hit_full)
    hit_x_01 = bool_to_01(kv.collision.hit_x)
    hit_y_01 = bool_to_01(kv.collision.hit_y)

    resolve_attn = attend_mean_where(
        pos_encoding,
        validity=is_wall,
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

    new_x = add(token.player_x, kv.vel_dx)
    new_y = add(token.player_y, kv.vel_dy)
    resolved_x = select(use_new_x, new_x, token.player_x)
    resolved_y = select(use_new_y, new_y, token.player_y)
    return resolved_x, resolved_y
