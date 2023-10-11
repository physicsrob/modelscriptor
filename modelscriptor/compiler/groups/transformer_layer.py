from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer


class TransformerLayer:
    attn: AttnSubLayer
    ffn: FFNSubLayer

    def __init__(self, d: int, d_head: int):
        self.attn = AttnSubLayer(d, d_head)
        self.ffn = FFNSubLayer(d)

    def num_params(self):
        return self.attn.num_params() + self.ffn.num_params()
