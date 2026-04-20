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
class PlayerInputs:
    player_x: Node
    player_y: Node
    player_angle: Node
    is_player_x: Node
    is_player_y: Node
    is_player_angle: Node
    pos_encoding: PosEncoding


@dataclass
class PlayerOutputs:
    px: Node
    py: Node
    cos_theta: Node
    sin_theta: Node


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_player(inputs: PlayerInputs) -> PlayerOutputs:
    with annotate("player/broadcast"):
        px = attend_mean_where(
            inputs.pos_encoding,
            validity=inputs.is_player_x,
            value=inputs.player_x,
        )

        py = attend_mean_where(
            inputs.pos_encoding,
            validity=inputs.is_player_y,
            value=inputs.player_y,
        )

        cos_theta, sin_theta = trig_lookup(inputs.player_angle)
        trig_attn = attend_mean_where(
            inputs.pos_encoding,
            validity=inputs.is_player_angle,
            value=Concatenate([cos_theta, sin_theta]),
        )
        player_cos = extract_from(trig_attn, 2, 0, 1, "player_cos")
        player_sin = extract_from(trig_attn, 2, 1, 1, "player_sin")

    return PlayerOutputs(
        px=px,
        py=py,
        cos_theta=player_cos,
        sin_theta=player_sin,
    )
