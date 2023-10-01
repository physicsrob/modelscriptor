from itertools import product
from typing import Set, List, Tuple

from modelscriptor.compiler.components.component import Component, NodeComponentStrategy
from modelscriptor.graph import Node


class GroupStrategy:
    sub_strategies: List[Tuple[Component, Node, NodeComponentStrategy]]

    def __init__(self):
        self.sub_strategies = []

    def get_score(self):
        return sum(s.points for c, n, s in self.sub_strategies)

    def place_node(
        self, layer_component: Component, node: Node, strategy: NodeComponentStrategy
    ):
        self.sub_strategies.append((layer_component, node, strategy))

    def get_component_strategies(
        self, component: Component
    ) -> List[NodeComponentStrategy]:
        return [s for c, n, s in self.sub_strategies if c == component]

    def get_component_inputs(self, component: Component) -> Set[Node]:
        strategies = self.get_component_strategies(component)
        input_nodes = set()
        for s in strategies:
            input_nodes |= s.in_nodes
        return input_nodes

    @classmethod
    def from_node_strategies(
        cls, component: Component, strategies: List[NodeComponentStrategy]
    ):
        ret = cls()
        for s in strategies:
            ret.place_node(component, s.out_node, s)
        return ret

    @classmethod
    def merge(cls, strategy_list: List["GroupStrategy"]):
        result = cls()
        for s in strategy_list:
            result.sub_strategies += s.sub_strategies
        return result


def get_sequential_placement_strategies(
    output_nodes: Set[Node],
    layer_components: List[Component],
    strategy_count: int = 3,
) -> List[GroupStrategy]:
    # Given a sequence of components, each one connected to the next, this function searches
    # for placement strategies which result in output_nodes being accessible on the last layer.

    # Get all strategies for output_node on layers[-1]
    # (Remember we always work from the output towards the input)
    current_component = layer_components[-1]

    # Each entry in node_strategies is a list of strategies for the corresponding node.
    nodes_strategies: List[List[NodeComponentStrategy]] = [
        current_component.get_strategies(node) for node in output_nodes
    ]

    # Combined strategy is the outer product of the strategies for each current output node
    # in the current layer.
    group_strategies = [
        GroupStrategy.from_node_strategies(current_component, strategy_list)
        for strategy_list in product(*nodes_strategies)
    ]

    # Sort the strategies and only keep the top-k
    group_strategies.sort(key=lambda s: -s.get_score())
    group_strategies = group_strategies[:strategy_count]

    if len(layer_components) == 1:
        return group_strategies

    result_strategies = []
    for s in group_strategies:
        # Now let's process all strategies that start with s
        out_nodes = s.get_component_inputs(current_component)
        s_strategies = get_sequential_placement_strategies(
            out_nodes, layer_components[:-1]
        )
        for s2 in s_strategies:
            result_strategies.append(GroupStrategy.merge([s, s2]))

    result_strategies.sort(key=lambda s: -s.get_score())
    result_strategies = result_strategies[:strategy_count]
    return result_strategies


def get_res_sequential_placement_strategies(
    output_nodes: Set[Node],
    layer_components: List[Component],
    strategy_count: int = 3,
) -> List[GroupStrategy]:
    # Assumes a network substructure which is:
    # [Input] -> [LayerComponent 1] -> ... -> [LayerComponent N] -> [Add LayerComponent N w/ Input]
    # This could be the attention sub-layer of a transformer (with one layer_component, the
    # attention head), or a ffn sub-layer of a transformer (with layer_components being Linear ->
    # ReLU -> Linear)
    ...
