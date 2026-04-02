from typing import List, Dict, Optional

import torch

from modelscriptor.compiler.feature_assignment import FeatureAssignment
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.graph import (
    Node,
    Constant,
    InputNode,
    PosEncoding,
    Concatenate,
    Embedding,
)


class HeadlessTransformer:
    layers: List[TransformerLayer]
    d: int
    d_head: int
    pos_encoding: Optional[PosEncoding]
    feature_assignment: Optional[FeatureAssignment]

    def __init__(
        self,
        d: int,
        d_head: int,
        pos_encoding: Optional[PosEncoding] = None,
    ):
        self.d = d
        self.d_head = d_head
        self.pos_encoding = pos_encoding
        self.layers = []
        self.feature_assignment = None

    def add_layer(self, end: bool = False) -> TransformerLayer:
        layer = TransformerLayer(self.d, self.d_head, self.pos_encoding)
        if not end:
            self.layers = [layer] + self.layers
        else:
            self.layers.append(layer)
        return layer

    def get_input_res_stream(
        self,
        n_pos: int,
        input_values: Dict[str, torch.Tensor],
    ):
        assert self.feature_assignment
        in_state = self.layers[0].attn.in_state
        res_stream = torch.zeros((n_pos, self.d))

        for node in self.feature_assignment.get_nodes(in_state):
            indices = self.feature_assignment.get_node_indices(in_state, node)
            if isinstance(node, Constant):
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = node.value[i]
            elif isinstance(node, InputNode):
                assert node.name in input_values
                value = input_values[node.name]
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = value[:, i]
            elif isinstance(node, PosEncoding):
                encoding = node.get_pos_encoding(n_pos)
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = encoding[:, i]
            elif isinstance(node, Concatenate):
                # Noop — children are guaranteed to be in the state individually.
                pass
            elif isinstance(node, Embedding):
                embedding_output = node.compute(n_pos, input_values)
                for i, idx in enumerate(indices):
                    res_stream[:, idx] = embedding_output[:, i]
            else:
                assert False, "Unsupported node type"
        return res_stream

    def forward(self, inp: torch.Tensor, return_states=False):
        res = inp
        all_states = {}
        for i, layer in enumerate(self.layers):
            for sublayer, sublayer_name in [(layer.attn, "attn"), (layer.ffn, "ffn")]:
                if return_states:
                    res, states = sublayer.forward(res, return_states=True)
                    prefixed_states = {
                        f"layer_{i}_{sublayer_name}_{key}": value
                        for key, value in states.items()
                    }
                    all_states.update(prefixed_states)
                else:
                    res = sublayer.forward(res)
        if return_states:
            return res, all_states
        else:
            return res

    def compute(
        self, n_pos: int, input_values: Dict[str, torch.Tensor]
    ) -> Dict[Node, torch.Tensor]:
        assert self.feature_assignment

        res = self.forward(self.get_input_res_stream(n_pos, input_values))
        result = {}
        out_state = self.layers[-1].ffn.out_state

        for node in self.feature_assignment.get_nodes(out_state):
            indices = self.feature_assignment.get_node_indices(out_state, node)
            result[node] = res[:, indices]
        return result
