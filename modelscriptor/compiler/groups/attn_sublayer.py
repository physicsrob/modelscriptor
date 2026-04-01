from typing import Optional

import torch

from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.feature_assignment import ResidualStreamState
from modelscriptor.graph import PosEncoding


class AttnSubLayer:
    """Attention sublayer: multi-head attention + residual skip connection.

    Forward: out = attn(inp) + inp
    """

    def __init__(
        self,
        d: int,
        d_head: int,
        pos_encoding: Optional[PosEncoding] = None,
    ):
        self.d = d
        self.in_state = ResidualStreamState(name="AttnSubLayer In State")
        self.out_state = ResidualStreamState(name="AttnSubLayer Out State")
        self.attn = AttnLayerComponent(d, d_head, pos_encoding, name="attn")

    def forward(self, inp: torch.Tensor, return_states=False):
        states = {}

        x = self.attn.forward(inp)
        states["attn_out_state"] = (self.attn.out_state, x)
        x = x + inp
        states["skip_out_state"] = (self.out_state, x)
        if return_states:
            return x, states
        else:
            return x

    def num_params(self):
        return self.attn.num_params()

    def resize(self, new_d):
        self.d = new_d
        self.attn.resize(new_d)
