from typing import Set, Dict, List, NamedTuple, Tuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
    ResidualStreamState,
)
from modelscriptor.compiler.groups.strategy import (
    GroupStrategy,
    get_combined_strategies,
)
from modelscriptor.graph import Node, Concatenate, Linear


class Group:
    d: int
    in_state: ResidualStreamState
    out_state: ResidualStreamState

    def __init__(self, d, name: str = ""):
        self.d = d
        self.in_state = ResidualStreamState(name=f"{name} Group In State")
        self.out_state = ResidualStreamState(name=f"{name} Group Out State")

    @abstractmethod
    def forward(self, inp: torch.Tensor, return_states=False):
        ...

    @abstractmethod
    def get_strategies_for_node(self, node: Node) -> List[GroupStrategy]:
        ...

    def get_strategies(self, nodes: Set[Node]) -> List[GroupStrategy]:
        node_to_strategies = {}
        for node in nodes:
            node_strategies = self.get_strategies_for_node(node)
            node_strategies.sort(key=lambda s: s.get_score())

            # For any given set of input/output nodes, we'll only keep one strategy.
            existings_keys = set()
            filtered_strategies = list()
            for strategy in node_strategies:
                key = tuple(
                    sorted(
                        strategy.get_compilable_input_nodes(include_skip=True),
                        key=lambda n: n.node_id,
                    )
                )
                if key in existings_keys:
                    continue
                existings_keys.add(key)
                filtered_strategies.append(strategy)

            # Filter out duplicate strategies
            node_to_strategies[node] = filtered_strategies

        return get_combined_strategies(node_to_strategies)

    @abstractmethod
    def get_constraints(self, strategy: GroupStrategy) -> FeatureAssignmentConstraints:
        ...

    @abstractmethod
    def apply_strategy(
        self, feature_assignment: FeatureAssignment, strategy: GroupStrategy
    ):
        ...

    @abstractmethod
    def resize(self, new_d):
        ...
