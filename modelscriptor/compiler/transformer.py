from typing import List, Set, Dict

import torch

from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
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

    def get_input_res_stream(self, n_pos: int, input_values: Dict[str, torch.Tensor]):
        in_state = self.layers[0].in_state
        res_stream = torch.zeros((n_pos, self.d))

        for node in in_state.get_distinct_nodes():
            indices = in_state.get_node_indices(node)
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

        for node in out_state.get_distinct_nodes():
            indices = out_state.get_node_indices(node)
            result[node] = res[:, indices]
        return result

    def compute_layer_output(
        self, n_pos: int, input_values: Dict[str, torch.Tensor], layer_n: int
    ) -> Dict[Node, torch.Tensor]:
        res = self.get_input_res_stream(n_pos, input_values)

        for layer in self.layers[: layer_n + 1]:
            res, states = layer.forward(res, return_states=True)

        result = {}
        for state_name, (res_state, x) in states.items():
            for node in res_state.get_distinct_nodes():
                indices = res_state.get_node_indices(node)
                if node in result:
                    if not torch.allclose(result[node], x[:, indices]):
                        print(
                            f"Node {node} changed value from {result[node]} to {x[:, indices]}"
                        )
                        print(f"indices: {indices}")
                        assert False
                else:
                    result[node] = x[:, indices]
        return result
