from typing import Optional

from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.graph import PosEncoding


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

    def num_params(self):
        return self.attn.num_params() + self.ffn.num_params()
