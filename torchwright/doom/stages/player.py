"""PLAYER stage: discrete player state → float broadcast.

Three token types emitted after EOS, before SORTED:

    PLAYER_X     Broadcasts the resolved x position to all positions.
    PLAYER_Y     Broadcasts the resolved y position to all positions.
    PLAYER_ANGLE Looks up cos(θ)/sin(θ) and broadcasts them to all
                 positions.

The host feeds the resolved player state (from EOS overflow outputs)
as the ``player_x``, ``player_y``, ``player_angle`` inputs at these
positions.  The broadcasts land in the KV cache so downstream tokens
(SORTED, RENDER) can read them via attention.
"""

from dataclasses import dataclass

from torchwright.graph import Concatenate, Node, annotate
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.attention_ops import attend_mean_where

from torchwright.doom.graph_utils import extract_from
from torchwright.doom.renderer import trig_lookup

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass
class PlayerToken:
    player_x: Node      # host-fed at PLAYER_X position
    player_y: Node      # host-fed at PLAYER_Y position
    player_angle: Node   # host-fed at PLAYER_ANGLE position


@dataclass
class PlayerKVOutput:
    px: Node
    py: Node
    cos_theta: Node
    sin_theta: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_player(
    token: PlayerToken,
    *,
    is_player_x: Node,
    is_player_y: Node,
    is_player_angle: Node,
    pos_encoding: PosEncoding,
) -> PlayerKVOutput:
    with annotate("player/broadcast"):
        px = attend_mean_where(
            pos_encoding,
            validity=is_player_x,
            value=token.player_x,
        )

        py = attend_mean_where(
            pos_encoding,
            validity=is_player_y,
            value=token.player_y,
        )

        cos_theta, sin_theta = trig_lookup(token.player_angle)
        trig_attn = attend_mean_where(
            pos_encoding,
            validity=is_player_angle,
            value=Concatenate([cos_theta, sin_theta]),
        )
        player_cos = extract_from(trig_attn, 2, 0, 1, "player_cos")
        player_sin = extract_from(trig_attn, 2, 1, 1, "player_sin")

    return PlayerKVOutput(
        px=px,
        py=py,
        cos_theta=player_cos,
        sin_theta=player_sin,
    )
