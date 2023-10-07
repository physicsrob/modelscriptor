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

    def forward(self, inp: torch.Tensor, return_states=False):
        """
        Forward pass through the model.

        Parameters
        ----------
        inp : torch.Tensor
            The input tensor.
        return_states : bool, optional
            If True, returns the intermediate states from each layer.
            Default is False.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor and dict
            The output tensor if `return_states=False`.
            A tuple of the output tensor and a dictionary containing intermediate states
            if `return_states=True`. The dictionary keys are the names of the layers or
            operations, and the values are tuples containing the state object and the
            output tensor of that layer.
        """
        res = inp
        all_states = {}  # Dictionary to collect all states
        for i, layer in enumerate(self.layers):
            if return_states:
                res, states = layer.forward(res, return_states=True)
                # Prefix the keys in the states dict and update all_states
                prefixed_states = {
                    f"layer_{i}_{key}": value for key, value in states.items()
                }
                all_states.update(prefixed_states)
            else:
                res = layer.forward(res)
        if return_states:
            return res, all_states
        else:
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
