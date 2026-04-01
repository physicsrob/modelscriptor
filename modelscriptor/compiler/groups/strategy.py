from itertools import product
from typing import Set, List, Tuple, Dict, Any

from modelscriptor.compiler.components.component import Component, NodeComponentStrategy
from modelscriptor.compiler.components.skip import (
    SkipNodeComponentStrategy,
    SkipLayerComponent,
)
from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    simplify_nodes,
    solve,
)
from modelscriptor.compiler.utils import get_ancestor_nodes
from modelscriptor.graph import Node, Concatenate, PosEncoding, Embedding
from modelscriptor.graph.misc import Placeholder, InputNode, Constant


def _is_input_type(node: Node):
    # Although a Constant is compilable, we don't consider it as such for calculating the score
    # of a strategy.  This encourages the constants to be compiled away.
    return (
        isinstance(node, InputNode)
        or isinstance(node, PosEncoding)
        or isinstance(node, Embedding)
    )


class GroupStrategy:
    sub_strategies: List[Tuple[Component, Node, NodeComponentStrategy]]
    components: List[Component]
    component_names: List[str]
    dependent_nodes: Set[Node]

    def __init__(self, components: List[Component], component_names: List[str]):
        self.sub_strategies = []
        self.components = components
        self.component_names = component_names
        self.dependent_nodes = set()

    def place_node(
        self, layer_component: Component, node: Node, strategy: NodeComponentStrategy
    ):
        # Only called for single layer groupstrategies
        self.sub_strategies.append((layer_component, node, strategy))
        self.dependent_nodes |= set(strategy.in_nodes)
        if isinstance(strategy, SkipNodeComponentStrategy):
            self.dependent_nodes.add(strategy.skip_node)

    def get_component_strategies(
        self, component: Component
    ) -> List[NodeComponentStrategy]:
        return [s for c, n, s in self.sub_strategies if c == component]

    def get_constraints_for_strategy(self):
        constraints = FeatureAssignmentConstraints()
        for c, n, s in self.sub_strategies:
            constraints.update(c.get_constraints_for_strategy(s))
        return constraints

    def get_compilable_input_nodes(self, include_skip: bool = False) -> Set[Node]:
        strategies = self.get_component_strategies(self.components[0])

        input_nodes = set()
        for s in strategies:
            input_nodes.update(s.in_nodes)

        if include_skip:
            for c, n, s in self.sub_strategies:
                if isinstance(s, SkipNodeComponentStrategy):
                    input_nodes.add(s.skip_node)

        return set(simplify_nodes(list(input_nodes)))

    def get_score(self):
        dependenct_ancestors = get_ancestor_nodes(self.dependent_nodes)
        return len([node for node in dependenct_ancestors if not _is_input_type(node)])

    def print(self, prefix: str = ""):
        print(f"{prefix}Group Strategy")
        for layer, name in zip(self.components, self.component_names):
            for s in self.get_component_strategies(layer):
                print(f"{prefix}- {name}: {repr(layer)} {repr(s)}")


def get_strategies_for_component(
    output_nodes: Set[Node],
    component: Component,
    component_name: str,
    strategy_count: int = 10,
):
    nodes_strategies: List[List[NodeComponentStrategy]] = [
        component.get_strategies(node) for node in output_nodes
    ]

    # Take the outer product of the strategies for each current output node
    group_strategies = []
    for strategy_list in product(*nodes_strategies):
        # Inlining 'from_node_strategies' logic
        ret = GroupStrategy([component], [component_name])
        for s in strategy_list:
            ret.place_node(component, s.out_node, s)
        group_strategies.append(ret)

    # Sort the strategies and only keep the top-k
    group_strategies.sort(key=lambda s: s.get_score())

    return group_strategies[:strategy_count]


def get_combined_strategies(
    node_to_strategies: Dict[Node, List[GroupStrategy]],
    existing_constraints: FeatureAssignmentConstraints,
    max_strategies: int = 2,
) -> List[GroupStrategy]:
    # Given a dictionary of node -> strategies, this function returns a list of strategies
    # which are the combination of all strategies for each node.
    # For example, if node1 has strategies [s1, s2] and node2 has strategies [s3, s4], this
    # function will return [s1 + s3, s1 + s4, s2 + s3, s2 + s4]
    node = next(iter(node_to_strategies.keys()))
    if not solve(existing_constraints):
        print("Could not solve")
        print()

    if len(node_to_strategies) == 1:
        strategies = []
        for s in node_to_strategies[node]:
            constraint = s.get_constraints_for_strategy()
            constraint.update(existing_constraints)
            if solve(constraint):
                strategies.append(s)
        if not len(strategies):
            print("Filtered out all strategies due to existing constraint")
        return strategies
    else:
        strategies = node_to_strategies[node]
        strategies.sort(key=lambda s: s.get_score())
        strategies = strategies[:max_strategies]

        other_node_to_strategies = {
            n: s for n, s in node_to_strategies.items() if n != node
        }
        other_strategies = get_combined_strategies(
            other_node_to_strategies, existing_constraints, max_strategies
        )

        # Calculate product
        combined_strategies = []
        for s1 in strategies:
            for s2 in other_strategies:
                combined = GroupStrategy(s1.components, s1.component_names)
                combined.sub_strategies = s1.sub_strategies + s2.sub_strategies
                combined.dependent_nodes = s1.dependent_nodes | s2.dependent_nodes
                constraint = combined.get_constraints_for_strategy()
                constraint.update(existing_constraints)
                solvable = solve(constraint)
                if solvable:
                    combined_strategies.append(combined)
        if not len(combined_strategies):
            print()

        combined_strategies.sort(key=lambda s: s.get_score())
        return combined_strategies[:max_strategies]


def get_sequential_placement_strategies(
    output_nodes: Set[Node],
    components: List[Component],
    component_names: List[str],
    strategy_count: int = 10,
    depth: int = 0,
    verbose: bool = False,
) -> List[GroupStrategy]:
    # Given a sequence of components, each one connected to the next, this function searches
    # for placement strategies which result in output_nodes being accessible on the last layer.

    if verbose and depth == 0:
        print(f"Finding strategies for {output_nodes}")
        print(f"on layers: {component_names}")
        print()

    # Get all strategies for output_node on layers[-1]
    # (Remember we always work from the output towards the input)
    current_component = components[-1]
    current_component_name = component_names[-1]
    group_strategies = get_strategies_for_component(
        output_nodes, current_component, current_component_name, strategy_count
    )

    if verbose:
        if depth:
            prefix = f"(depth {depth})   "
        else:
            prefix = ""

        print(
            f"{prefix}# Strategies for {current_component_name}.  {len(group_strategies)} strategies."
        )
        for i, strategy in enumerate(group_strategies):
            print(f"{prefix}Strategy {i}:")
            strategy.print(prefix)
            print()
        print(f"{prefix}End strategies.")

    if len(components) == 1:
        return group_strategies

    result_strategies = []
    for i, s in enumerate(group_strategies):
        # Now let's process all strategies that start with s
        out_nodes = s.get_compilable_input_nodes()
        if verbose and depth == 0:
            print(f"## Strategy {i} has nodes {out_nodes}")

        s_strategies = get_sequential_placement_strategies(
            out_nodes, components[:-1], component_names[:-1], depth=depth + 1
        )
        if verbose and depth == 0:
            print(f"Found {len(s_strategies)} strategies for remaining components")
            for n, s_ in enumerate(s_strategies):
                print(f"Remaining strategy {n} for primary strategy {i}")
                s_.print()
            print()

        for s2 in s_strategies:
            strategy = GroupStrategy(components, component_names)
            # We assume that the dependencies of s are satisfied by s2
            # except for any skip nodes which will still have the skip dependencies.
            strategy.sub_strategies = s.sub_strategies.copy() + s2.sub_strategies.copy()
            strategy.dependent_nodes = s2.dependent_nodes

            # If any of the strategies in layer1_strategy are skip strategies, we need to add the skip node
            # to the dependent nodes.
            for c, n, s_ in s.sub_strategies:
                if isinstance(s_, SkipNodeComponentStrategy):
                    strategy.dependent_nodes.add(s_.skip_node)

            result_strategies.append(strategy)

    result_strategies.sort(key=lambda s: s.get_score())
    result_strategies = result_strategies[:strategy_count]
    return result_strategies
