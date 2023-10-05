from modelscriptor.compiler.components.attn import AttnLayerComponent
from modelscriptor.compiler.components.skip import SkipLayerComponent
from modelscriptor.compiler.groups.group import Group


class AttnSubLayer(Group):
    attn: AttnLayerComponent
    skip: SkipLayerComponent
