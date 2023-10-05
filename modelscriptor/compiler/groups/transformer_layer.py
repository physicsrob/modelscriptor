from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.group import Group


class TransformerLayer(Group):
    attn: AttnSubLayer
    ffn: FFNSubLayer
