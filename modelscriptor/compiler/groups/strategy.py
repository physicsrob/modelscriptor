from collections import defaultdict
from itertools import product
from typing import Set, List, Tuple, Dict

from modelscriptor.compiler.components.component import Component, NodeComponentStrategy
from modelscriptor.compiler.components.skip import SkipNodeComponentStrategy
from modelscriptor.graph import Node, Add, Concatenate
from modelscriptor.graph.misc import Placeholder


def _get_ancestor_nodes(start_nodes: Set[Node]) -> Set[Node]:
    # Find all ancestors
    result = set()

    for node in start_nodes:
        result.add(node)
        if node.inputs:
            result |= _get_ancestor_nodes(set(node.inputs))
    return result


class GroupStrategy:
    sub_strategies: List[Tuple[Component, Node, NodeComponentStrategy]]
    dependent_nodes: Set[Node]

    def __init__(self):
        self.sub_strategies = []
        self.dependent_nodes = set()

    def place_node(
        self, layer_component: Component, node: Node, strategy: NodeComponentStrategy
    ):
        self.sub_strategies.append((layer_component, node, strategy))
        self.dependent_nodes |= set(strategy.in_nodes)
        if isinstance(strategy, SkipNodeComponentStrategy):
            self.dependent_nodes.add(strategy.skip_node)

    def get_component_strategies(
        self, component: Component
    ) -> List[NodeComponentStrategy]:
        return [s for c, n, s in self.sub_strategies if c == component]

    def get_represented_nodes(self) -> Set[Node]:
        result = set()
        for c, n, s in self.sub_strategies:
            result.add(n)
            result |= set(n.inputs)
        return result
        # return {n for c, n, s in self.sub_strategies}

    def get_compilable_input_nodes(self, component: Component) -> Set[Node]:
        strategies = self.get_component_strategies(component)
        input_nodes = set()
        for s in strategies:
            for n in s.in_nodes:
                if isinstance(n, Concatenate):
                    inputs = n.simplify_inputs()
                    input_nodes.update(set(inputs))
                elif isinstance(n, Placeholder):
                    pass
                else:
                    input_nodes.add(n)
        return input_nodes

    def get_score(self):
        dependenct_ancestors = _get_ancestor_nodes(self.dependent_nodes)
        return len(dependenct_ancestors)

    def print(self, layer_components: List[Component], layer_names: List[str]):
        print("Group Strategy")
        for layer, name in zip(layer_components, layer_names):
            for s in self.get_component_strategies(layer):
                print(f"- {name}: {repr(layer)} {repr(s)}")


def get_strategies_for_component(
    output_nodes: Set[Node], layer_component: Component, strategy_count: int = 10
):
    nodes_strategies: List[List[NodeComponentStrategy]] = [
        layer_component.get_strategies(node) for node in output_nodes
    ]

    # Take the outer product of the strategies for each current output node
    group_strategies = []
    for strategy_list in product(*nodes_strategies):
        # Inlining 'from_node_strategies' logic
        ret = GroupStrategy()
        for s in strategy_list:
            ret.place_node(layer_component, s.out_node, s)
        group_strategies.append(ret)

    # Sort the strategies and only keep the top-k
    group_strategies.sort(key=lambda s: s.get_score())
    return group_strategies[:strategy_count]


def get_combined_strategies(node_to_strategies: Dict[Node, List[GroupStrategy]]):
    # Given a dictionary of node -> strategies, this function returns a list of strategies
    # which are the combination of all strategies for each node.
    # For example, if node1 has strategies [s1, s2] and node2 has strategies [s3, s4], this
    # function will return [s1 + s3, s1 + s4, s2 + s3, s2 + s4]
    # for k, v in node_to_strategies:
    #     print(k, v)

    combined_strategies = []

    for strategies in product(*node_to_strategies.values()):
        combined = GroupStrategy()
        for strategy in strategies:
            combined.sub_strategies += strategy.sub_strategies
            combined.dependent_nodes |= strategy.dependent_nodes
        combined_strategies.append(combined)
    combined_strategies.sort(key=lambda s: s.get_score())
    return combined_strategies


def get_sequential_placement_strategies(
    output_nodes: Set[Node],
    layer_components: List[Component],
    strategy_count: int = 10,
) -> List[GroupStrategy]:
    # Given a sequence of components, each one connected to the next, this function searches
    # for placement strategies which result in output_nodes being accessible on the last layer.

    # Get all strategies for output_node on layers[-1]
    # (Remember we always work from the output towards the input)
    current_component = layer_components[-1]
    group_strategies = get_strategies_for_component(
        output_nodes, current_component, strategy_count
    )
    if len(layer_components) == 1:
        return group_strategies

    result_strategies = []
    for s in group_strategies:
        # Now let's process all strategies that start with s
        out_nodes = s.get_compilable_input_nodes(current_component)
        s_strategies = get_sequential_placement_strategies(
            out_nodes, layer_components[:-1]
        )
        for s2 in s_strategies:
            strategy = GroupStrategy()
            # We assume that the dependencies of s are satisfied by s2
            # except for any skip nodes which will still have the skip dependencies.
            strategy.sub_strategies = s.sub_strategies + s2.sub_strategies
            strategy.dependent_nodes = s2.dependent_nodes

            # If any of the strategies in layer1_strategy are skip strategies, we need to add the skip node
            # to the dependent nodes.
            for c, n, s2 in s.sub_strategies:
                if isinstance(s2, SkipNodeComponentStrategy):
                    strategy.dependent_nodes.add(s2.skip_node)
            result_strategies.append(strategy)

    result_strategies.sort(key=lambda s: s.get_score())
    result_strategies = result_strategies[:strategy_count]
    return result_strategies
