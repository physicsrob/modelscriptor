from enum import Enum
from typing import List, Tuple, Set

from modelscriptor.compiler.plan.layer_component import LayerComponent
from modelscriptor.graph import Node


class NodePlacementType(Enum):
    pass_through = "pass_through"
    compile = "compile"


class NodeLayerComponentStrategy:
    # Represents the way a node passes through and is transformed by one layercomponent.
    in_nodes: Set[Node]  # Input nodes used for this computation
    out_node: Node  # The node we're considering, and is represented in the output.
    ...


class CompileStrategy:
    node_placements: List[Tuple[LayerComponent, Node, NodePlacementType]]

    def __init__(self):
        self.node_placements = []

    def place_node(
        self,
        layer_component: LayerComponent,
        node: Node,
        placement_type: NodePlacementType,
    ):
        self.node_placements.append((layer_component, node, placement_type))

    def get_score(self):
        return sum(
            1 if o == NodePlacementType.compile else 0
            for l, t, o in self.node_placements
        )

    #
    # def get_layer_node_strategy(self, layer: Layer, node: Node) -> LayerNodeOption:
    #     ...

    def get_layer_inputs(self, layer: LayerComponent) -> Set[Node]:
        return {
            inp
            for layer, node, opt in self.node_placements
            for inp in node.inputs
            if opt == NodePlacementType.compile
        } | {
            node
            for layer, node, opt in self.node_placements
            if opt == NodePlacementType.pass_through
        }

    def get_layer_outputs(self, layer: LayerComponent) -> Set[Node]:
        return {node for layer, node, opt in self.node_placements}

    @classmethod
    def merge(cls, strategy_list: List["CompileStrategy"]):
        result = cls()
        for s in strategy_list:
            result.node_placements += s.node_placements
        return result
