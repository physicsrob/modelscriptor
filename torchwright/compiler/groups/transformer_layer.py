from typing import Optional

from torchwright.compiler.groups.attn_sublayer import AttnSubLayer
from torchwright.compiler.groups.ffn_sublayer import FFNSubLayer
from torchwright.graph import PosEncoding


class TransformerLayer:
    attn: AttnSubLayer
    ffn: FFNSubLayer

    def __init__(
        self,
        d: int,
        d_head: int,
        pos_encoding: Optional[PosEncoding] = None,
    ):
        self.attn = AttnSubLayer(d, d_head, pos_encoding)
        self.ffn = FFNSubLayer(d)

    def to(self, device):
        self.attn.to(device)
        self.ffn.to(device)
        return self

    def num_params(self):
        return self.attn.num_params() + self.ffn.num_params()
