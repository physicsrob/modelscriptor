from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer


class TransformerLayer:
    attn: AttnSubLayer
    ffn: FFNSubLayer

    def __init__(self, d: int):
        self.attn = AttnSubLayer(d)
        self.ffn = FFNSubLayer(d)
