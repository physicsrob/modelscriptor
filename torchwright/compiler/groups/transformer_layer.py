from typing import Optional

from torchwright.compiler.groups.attn_sublayer import AttnSubLayer
from torchwright.compiler.groups.mlp_sublayer import MLPSubLayer
from torchwright.graph import PosEncoding


class TransformerLayer:
    attn: AttnSubLayer
    mlp: MLPSubLayer

    def __init__(
        self,
        d: int,
        d_head: int,
        pos_encoding: Optional[PosEncoding] = None,
        d_hidden: Optional[int] = None,
    ):
        self.attn = AttnSubLayer(d, d_head, pos_encoding)
        self.mlp = MLPSubLayer(d, d_hidden)

    def to(self, device):
        self.attn.to(device)
        self.mlp.to(device)
        return self

    def num_params(self):
        return self.attn.num_params() + self.mlp.num_params()
