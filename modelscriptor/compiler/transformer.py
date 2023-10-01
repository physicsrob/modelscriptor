from typing import List, Set, Dict

import torch

from modelscriptor.compiler.components.component import Component
from modelscriptor.compiler.components.linear import LinearLayerComponent
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.group import Group
from modelscriptor.compiler.res_state import ResState
from modelscriptor.graph import Node, Constant, InputNode


class FFNNetwork:
    layers: List[FFNSubLayer]
    d: int

    def __init__(self, d: int):
        self.d = d
        self.layers = []

    def add_layer(self) -> FFNSubLayer:
        layer = FFNSubLayer(self.d)
        self.layers = [layer] + self.layers
        return layer

    def print(self):
        print("FFNNetwork")
        print(f"{len(self.layers)} layers.")
        for layer in self.layers:
            layer.print()
        print()

    def get_input_nodes(self) -> Set[Node]:
        return self.layers[0].in_state.nodes

    def get_output_nodes(self) -> Set[Node]:
        return self.layers[-1].out_state.nodes

    def get_input_res_stream(self, n_pos: int, input_values: Dict[str, torch.Tensor]):
        in_state = self.layers[0].in_state
        res_stream = torch.zeros((n_pos, self.d))

        for node, indices in in_state.node_to_indices.items():
            if isinstance(node, Constant):
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = node.value[i]
            elif isinstance(node, InputNode):
                assert node.name in input_values
                value = input_values[node.name]
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = value[:, i]
            else:
                assert False, "Unsupported node type"
        return res_stream

    def forward(self, inp: torch.Tensor):
        res = inp
        for layer in self.layers:
            res = layer.forward(res)
        return res

    def compute(
        self, n_pos: int, input_values: Dict[str, torch.Tensor]
    ) -> Dict[Node, torch.Tensor]:
        res = self.forward(self.get_input_res_stream(n_pos, input_values))
        result = {}
        out_state = self.layers[-1].out_state

        for out_node, out_indices in out_state.node_to_indices.items():
            result[out_node] = res[:, out_indices]
        return result
