from typing import Set, Dict, List, NamedTuple

import torch

from modelscriptor.compiler.plan.placement import CompileStrategy
from modelscriptor.compiler.plan.placement_search import (
    get_sequential_placement_strategies,
)
from modelscriptor.graph import Node, Concatenate, Linear


# Definitions:
# LayerComponent: One sub-sub-layer -- a linear transformation, relu, attention layer, or skip connection.
# NetworkComponent:  A single linear transformation all the way up to multiple layers.
# Node: One node in the computation graph.


class NetworkComponent:
    d: int
    in_nodes: Set[Node]
    in_nodes_to_indices: Dict[Node, torch.Tensor]
    out_nodes: Set[Node]
    out_nodes_to_indices: Dict[Node, torch.Tensor]

    def __init__(self, d):
        self.d = d
        self.in_nodes = set()
        self.in_nodes_to_indices = set()
        self.out_nodes = set()
        self.out_nodes_to_indices = set()

    def add_node(self, node: Node, indices: torch.Tensor):
        self.nodes.add(node)
        self.node_to_indices[node] = indices

    def allocate_node(self, node: Node, is_input: bool = False):
        # Allocate a new node.
        if is_input:
            assert node not in self.in_nodes
        else:
            assert node not in self.out_nodes
        ...

    def connect(self, output: "NetworkComponent"):
        # Assuming output is a network component that has been fully specified, connect the output
        # of this node to the input of output.
        self.out_nodes = output.in_nodes.copy()
        self.out_nodes_to_indices = output.in_nodes_to_indices.copy()


class NodeComponentStrategy:
    # Represents the way a node passes through and is transformed by one component
    in_nodes: Set[Node]  # Input nodes used for this computation
    out_node: Node  # Output node fhor this computation
    points: int


class LinearNodeComponentStrategy(NodeComponentStrategy):
    output_matrix: torch.Tensor  # d_input x d_output (both len(out_node))
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        in_node: Node,
        out_node: Node,
        output_matrix: torch.Tensor,
        output_bias: torch.Tensor,
        points: int,
    ):
        self.in_nodes = {in_node}
        self.out_node = out_node
        self.output_matrix = output_matrix
        self.output_bias = output_bias
        self.points = points


class LinearLayerComponent(NetworkComponent):
    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(self, d: int):
        super().__init__(d)
        self.output_matrix = torch.zeros(d, d)
        self.output_bias = torch.zeros(d)

    def get_strategies(self, node: Node) -> List[NodeComponentStrategy]:
        strategies = []

        # Always have the pass-through option.
        strategies.append(
            LinearNodeComponentStrategy(
                in_node=node,
                out_node=node,
                output_matrix=torch.eye(len(node)),
                output_bias=torch.zeros(len(node)),
                points=0,
            )
        )

        # If the node is linear, we can compile it!
        if isinstance(node, Linear):
            strategies.append(
                LinearNodeComponentStrategy(
                    in_node=node.inputs[0],
                    out_node=node,
                    output_matrix=node.output_matrix,
                    output_bias=node.output_bias,
                    points=1,
                )
            )
        return strategies

    def apply_strategy(self, strategy: NodeComponentStrategy):
        assert isinstance(strategy, LinearNodeComponentStrategy)
        assert (
            strategy.out_node in self.out_nodes
        ), "Strategy applied before output allocated"
        in_node = next(iter(strategy.in_nodes))
        self.allocate_node(in_node, is_input=True)

        # Copy the matrix
        in_indices = self.in_nodes_to_indices[in_node]
        out_indices = self.out_nodes_to_indices[strategy.out_node]

        for i, in_idx in enumerate(in_indices):
            for j, out_idx in enumerate(out_indices):
                self.output_matrix[in_idx, out_idx] = strategy.output_matrix[i, j]

        for j, out_idx in enumerate(out_indices):
            self.output_bias[out_idx] = strategy.output_bias[j]


class ReLULayerComponent(NetworkComponent):
    ...


class AttnLayerComponent(NetworkComponent):
    ...


class SkipLayerComponent(NetworkComponent):
    ...


class AttnSubLayer(NetworkComponent):
    attn: AttnLayerComponent
    skip: SkipLayerComponent


class FFNSubLayer(NetworkComponent):
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent
    skip: SkipLayerComponent


class TransformerLayer(NetworkComponent):
    attn: AttnSubLayer
    ffn: FFNSubLayer


class TransformerNetwork(NetworkComponent):
    layers: List[TransformerLayer]

    def get_input_nodes(self) -> Set[Node]:
        return self.layers[0].in_nodes

    def get_output_nodes(self) -> Set[Node]:
        return self.layers[-1].out_nodes

    skip: SkipLayerComponent


class FFNSubLayer(NetworkComponent):
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent
    skip: SkipLayerComponent


class TransformerLayer(NetworkComponent):
    attn: AttnSubLayer
    ffn: FFNSubLayer


class OutputLayer(NetworkComponent):
    def __init__(self, d: int, output_node: Node):
        super().__init__(d)
        self.allocate_node(output_node)
        self.in_nodes = self.out_nodes
        self.in_nodes_to_indices = self.out_nodes_to_indices


def compile(output_node: Node) -> List[TransformerLayer]:
    ...


class SimpleNetwork(NetworkComponent):
    layers: List[LinearLayerComponent]
    output_layer: OutputLayer

    def __init__(self, d: int, output_node: Node):
        super().__init__(d)
        self.output_layer = OutputLayer(d, output_node)

    def get_prev_layer(self) -> NetworkComponent:
        if len(self.layers) > 1:
            return self.layers[1]
        else:
            return self.output_layer

    def add_layer(self) -> LinearLayerComponent:
        layer = LinearLayerComponent(self.d)
        self.layers = [layer] + self.layers
        layer.connect(self.get_prev_layer())
        return layer


def compile_simple(d: int, output_node: Node) -> SimpleNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = SimpleNetwork(d, output_node)
    while True:
        layer = net.add_layer()
        for node in layer.out_nodes:
            strategy = sorted(
                layer.get_strategies(output_node), key=lambda s: -s.points
            )[0]
            layer.apply_strategy(strategy)
        # Woohoo -- layer is done!
        ...


class FFNSubLayerNoSkip(NetworkComponent):
    linear1: LinearLayerComponent
    relu: ReLULayerComponent
    linear2: LinearLayerComponent

    def get_strategies(self, node: Node) -> List[CompileStrategy]:
        return get_sequential_placement_strategies(
            output_nodes={node},
            layer_components=[self.linear1, self.relu, self.linear2],
        )

    def apply_strategy(self, strategy: CompileStrategy):
        # Strategy contains nodes at each layer.
        # allocate them all.
        self.linear2.out_nodes = self.out_nodes
        self.linear2.out_nodes_to_indices = self.out_nodes_to_indices

        ...
        # APPLY STRATEGY
        ...

        self.in_nodes = self.linear1.in_nodes
        self.in_nodes_to_indices = self.linear1.in_nodes_to_indices


class MedSimpleNetwork(NetworkComponent):
    layers: List[FFNSubLayerNoSkip]
    output_layer: OutputLayer

    def __init__(self, d: int, output_node: Node):
        super().__init__(d)
        self.output_layer = OutputLayer(d, output_node)

    def get_prev_layer(self) -> NetworkComponent:
        if len(self.layers) > 1:
            return self.layers[1]
        else:
            return self.output_layer

    def add_layer(self) -> FFNSubLayerNoSkip:
        layer = FFNSubLayerNoSkip(self.d)
        self.layers = [layer] + self.layers
        layer.connect(self.get_prev_layer())
        return layer


def compile_med(d: int, output_node: Node) -> MedSimpleNetwork:
    # Start with the first layer and try to compile as much as possible.

    net = MedSimpleNetwork(d, output_node)
    while True:
        layer = net.add_layer()
        for node in layer.out_nodes:
            strategy = sorted(
                layer.get_strategies(output_node), key=lambda s: -s.points
            )[0]
            layer.apply_strategy(strategy)
        # Woohoo -- layer is done!
        ...
