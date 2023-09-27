from itertools import product
from typing import Set, List

from modelscriptor.compiler.plan.layer_component import LayerComponent
from modelscriptor.compiler.plan.placement import CompileStrategy
from modelscriptor.graph import Node


class SequentialLayerComponents:
    layer_components: List[LayerComponent]

    def __init__(self, layer_components: List[LayerComponent]):
        self.layer_components = layer_components

    def get_placement_strategies(self, output_node: Node) -> List[CompileStrategy]:
        return get_sequential_placement_strategies({output_node}, self.layer_components)


def get_sequential_placement_strategies(
    output_nodes: Set[Node],
    layer_components: List[LayerComponent],
    strategy_count: int = 3,
) -> List[CompileStrategy]:
    # Given a sequence of layercomponents, each one connected to the next, this function searches
    # for placement strategies which result in output_nodes being accessible on the last layer.

    # Get all strategies for output_node on layers[-1]
    # (Remember we always work from the output towards the input)
    current_layer = layer_components[-1]

    # Each entry in node_strategies is a list of strategies for the corresponding node.
    nodes_strategies = [current_layer.get_strategies(node) for node in output_nodes]

    # Combined strategy is the outer product of the strategies for each current output node
    # in the current layer.
    combined_strategies = [
        CompileStrategy.merge(strategy_list)
        for strategy_list in product(*nodes_strategies)
    ]

    # Sort the strategies and only keep the top-k
    combined_strategies.sort(key=lambda s: -s.get_score())
    combined_strategies = combined_strategies[:strategy_count]

    if len(layer_components) == 1:
        return combined_strategies

    result_strategies = []
    for s in combined_strategies:
        # Now let's process all strategies that start with s
        out_nodes = s.get_layer_inputs(current_layer)
        s_strategies = get_sequential_placement_strategies(
            out_nodes, layer_components[:-1]
        )
        for s2 in s_strategies:
            result_strategies.append(CompileStrategy.merge([s, s2]))

    result_strategies.sort(key=lambda s: -s.get_score())
    result_strategies = result_strategies[:strategy_count]
    return result_strategies


def get_res_sequential_placement_strategies(
    output_nodes: Set[Node],
    layer_components: List[LayerComponent],
    strategy_count: int = 3,
) -> List[CompileStrategy]:
    # Assumes a network substructure which is:
    # [Input] -> [LayerComponent 1] -> ... -> [LayerComponent N] -> [Add LayerComponent N w/ Input]
    # This could be the attention sub-layer of a transformer (with one layer_component, the
    # attention head), or a ffn sub-layer of a transformer (with layer_components being Linear ->
    # ReLU -> Linear)
    ...
