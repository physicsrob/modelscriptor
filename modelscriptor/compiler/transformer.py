from typing import List, Set, Dict, Union, Optional

import torch

from modelscriptor.compiler.components.embedding import EmbeddingLayerComponent
from modelscriptor.compiler.groups.attn_sublayer import AttnSubLayer
from modelscriptor.compiler.groups.ffn_sublayer import FFNSubLayer
from modelscriptor.compiler.groups.transformer_layer import TransformerLayer
from modelscriptor.graph import Node, Constant, InputNode, PosEncoding


class HeadlessTransformer:
    layers: List[TransformerLayer]
    d: int
    d_head: int
    pos_encoding: Optional[PosEncoding]

    def __init__(self, d: int, d_head: int, pos_encoding: Optional[PosEncoding] = None):
        self.d = d
        self.d_head = d_head
        self.pos_encoding = pos_encoding
        self.layers = []

    def add_layer(self) -> TransformerLayer:
        layer = TransformerLayer(self.d, self.d_head, self.pos_encoding)
        self.layers = [layer] + self.layers
        return layer

    def get_input_res_stream(self, n_pos: int, input_values: Dict[str, torch.Tensor]):
        in_state = self.layers[0].attn.in_state
        res_stream = torch.zeros((n_pos, self.d))

        for node in in_state.get_nodes():
            indices = in_state.get_node_indices(node)
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
            for sublayer, sublayer_name in [(layer.attn, "attn"), (layer.ffn, "ffn")]:
                if return_states:
                    res, states = sublayer.forward(res, return_states=True)
                    # Prefix the keys in the states dict and update all_states
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
        res = self.forward(self.get_input_res_stream(n_pos, input_values))
        result = {}
        out_state = self.layers[-1].ffn.out_state

        for node in out_state.get_nodes():
            indices = out_state.get_node_indices(node)
            result[node] = res[:, indices]
        return result


class Transformer:
    embed: EmbeddingLayerComponent
    headless_net: HeadlessTransformer

    def __init__(self, headless_net: HeadlessTransformer, max_vocab: int = 1000):
        self.headless_net = headless_net
        self.embed = EmbeddingLayerComponent(headless_net.d, max_vocab)

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
        List[str]
        """
        x = self.embed.forward(inp)
        x = self.headless_net.forward(x, return_states)
        return self.embed.deembed_forward(x)

    def compute(self, n_pos: int, inp: List[str]) -> List[str]:
        raise NotImplementedError("TODO")
