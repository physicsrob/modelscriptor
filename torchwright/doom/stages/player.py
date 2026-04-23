"""PLAYER stage: discrete player state → float broadcast.

Three token types emitted after EOS, before SORTED:

    PLAYER_X     Broadcasts the pre-collision x position to all positions.
    PLAYER_Y     Broadcasts the pre-collision y position to all positions.
    PLAYER_ANGLE Looks up cos(θ)/sin(θ) for the post-turn angle and
                 broadcasts them to all positions.

The host feeds the pre-collision player state as the ``player_x`` and
``player_y`` inputs.  The ``player_angle`` input at PLAYER_ANGLE is
ignored by this stage — cos/sin are computed from INPUT's post-turn
``new_angle`` broadcast so the host never does angle arithmetic.
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
    player_x: Node  # host-fed at PLAYER_X position (pre-collision)
    player_y: Node  # host-fed at PLAYER_Y position (pre-collision)


@dataclass
class PlayerKVInput:
    new_angle: Node  # from INPUT's broadcast — post-turn angle


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
    kv: PlayerKVInput,
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

        # cos/sin of the post-turn angle.  Sourced from INPUT's
        # ``new_angle`` broadcast so the host doesn't need to compute
        # (state.angle + turn_delta) % 256 itself.  The value is
        # identical at every position (INPUT already broadcast it), so
        # the PLAYER_ANGLE re-broadcast is semantically a no-op;
        # keeping the ``attend_mean_where`` hop preserves the stage
        # boundary and costs only one extra attention layer.
        cos_theta, sin_theta = trig_lookup(kv.new_angle)
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
