from typing import List

from modelscriptor.compiler.components.component import Component
from modelscriptor.compiler.components.linear import LinearLayerComponent
from modelscriptor.compiler.components.output import OutputLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node


class SimpleNetwork:
    layers: List[LinearLayerComponent]
    output_layer: OutputLayer
    d: int

    def __init__(self, d: int, output_node: Node):
        self.d = d
        self.output_layer = OutputLayer(d, output_node)

    def get_prev_layer(self) -> Component:
        if len(self.layers) > 1:
            return self.layers[1]
        else:
            return self.output_layer

    def add_layer(self) -> LinearLayerComponent:
        layer = LinearLayerComponent(self.d)
        self.layers = [layer] + self.layers
        layer.in_state.connect(self.get_prev_layer().out_state)
        return layer


class MedSimpleNetwork:
    layers: List[FFNSubLayer]
    output_layer: OutputLayer
    d: int

    def __init__(self, d: int, output_node: Node):
        self.d = d
        self.output_layer = OutputLayer(d, output_node)

    def get_prev_layer_in_state(self) -> ResState:
        if len(self.layers) > 1:
            return self.layers[1].in_state
        else:
            return self.output_layer.in_state

    def add_layer(self) -> FFNSubLayer:
        layer = FFNSubLayer(self.d)
        self.layers = [layer] + self.layers
        layer.out_state.connect(self.get_prev_layer_in_state())
        return layer
