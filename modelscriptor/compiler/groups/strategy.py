from itertools import product
from typing import Set, List, Tuple

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
        represented_nodes = self.get_represented_nodes()
        needed_nodes = _get_ancestor_nodes(self.get_represented_nodes())
        input_nodes = {n for n in needed_nodes if not len(n.inputs)}
        print(f"{represented_nodes=}")
        print(f"{needed_nodes=}")
        print(f"{input_nodes=}")
        print(f"{self.dependent_nodes=}")
        print(f"missing nodes: {needed_nodes.difference(represented_nodes)}")

        dependenct_ancestors = _get_ancestor_nodes(self.dependent_nodes)
        return len(dependenct_ancestors)

    def print(self, layer_components: List[Component], layer_names: List[str]):
        print("Group Strategy")
        for layer, name in zip(layer_components, layer_names):
            for s in self.get_component_strategies(layer):
                print(f"- {name}: {repr(layer)} {repr(s)}")

    @classmethod
    def from_node_strategies(
        cls, component: Component, strategies: List[NodeComponentStrategy]
    ):
        ret = cls()
        for s in strategies:
            ret.place_node(component, s.out_node, s)
            ret.dependent_nodes |= set(s.in_nodes)
            if isinstance(s, SkipNodeComponentStrategy):
                ret.dependent_nodes.add(s.skip_node)
        return ret

    @classmethod
    def merge(cls, strategy1: "GroupStrategy", strategy2: "GroupStrategy"):
        # strategy1 is for the component towards the output.
        # We assume that the dependencies of strategy1 are satisfied by strategy2.

        result = cls()
        result.sub_strategies = strategy1.sub_strategies + strategy2.sub_strategies
        result.dependent_nodes = strategy2.dependent_nodes
        # If any of the strategies in strategy1 are skip strategies, we need to add the skip node
        # to the dependent nodes.
        for c, n, s in strategy1.sub_strategies:
            if isinstance(s, SkipNodeComponentStrategy):
                result.dependent_nodes.add(s.skip_node)

        return result


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
    # print(
    #     "get_sequential_placement_strategies called for: ",
    #     output_nodes,
    #     "top layer:",
    #     current_component,
    # )

    # Each entry in node_strategies is a list of strategies for the corresponding node.
    nodes_strategies: List[List[NodeComponentStrategy]] = [
        current_component.get_strategies(node) for node in output_nodes
    ]
    if len(output_nodes) == 1:
        node = next(iter(output_nodes))
        print(f"{node=} {current_component=} {nodes_strategies=}")
    else:
        print(f"{output_nodes=} {current_component=} {nodes_strategies=}")

    # if "PosEnc" in str(output_nodes):
    #     import pdb
    #
    #     pdb.set_trace()

    # Combined strategy is the outer product of the strategies for each current output node
    # in the current layer.
    group_strategies = [
        GroupStrategy.from_node_strategies(current_component, list(strategy_list))
        for strategy_list in product(*nodes_strategies)
    ]

    # Sort the strategies and only keep the top-k
    group_strategies.sort(key=lambda s: s.get_score())
    group_strategies = group_strategies[:strategy_count]

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
            result_strategies.append(GroupStrategy.merge(s, s2))

    result_strategies.sort(key=lambda s: s.get_score())
    result_strategies = result_strategies[:strategy_count]
    return result_strategies


# def get_res_sequential_placement_strategies(
#     output_nodes: Set[Node],
#     layer_components: List[Component],
#     strategy_count: int = 3,
# ) -> List[GroupStrategy]:
#     # Assumes a network substructure which is:
#     # [Input] -> [LayerComponent 1] -> ... -> [LayerComponent N] -> [Add LayerComponent N w/ Input]
#     # This could be the attention sub-layer of a transformer (with one layer_component, the
#     # attention head), or a ffn sub-layer of a transformer (with layer_components being Linear ->
#     # ReLU -> Linear)
#     if len(output_nodes) > 1:
#         assert False  # NEED TO IMPLEMENT
#
#     output_node = next(iter(output_nodes))
#
#     # If output_node is add (with inputs addend1, addend2):
#     if isinstance(output_node, Add):
#         # Options:
#         #   Add(addend1 on input, addend2 on linear2)
#         #   Add(addend2 on input, addend1 on linear2)
#         strategy1 = GroupStrategy()
#         strategy1.place_node()
#     else:
#         # Options:
#         #   Add(Constant(0) on input, output_node on linear2)
#         #   Add(Constant(0) on linear2, output_node on input)
#         zeros = create_constant(torch.zeros(len(output_node)))
#         add_node = Add(zeros, output_node)
#         strategy1 = get_sequential_placement_strategies(
#             {output_node}, [self.linear1, self.relu, self.linear2]
#         )
#
#     return get_sequential_placement_strategies(
#         {output_node}, [self.linear1, self.relu, self.linear2]
#     )
