from typing import Set, Dict, List, NamedTuple, Tuple
from abc import ABC, abstractmethod

import torch

from modelscriptor.compiler.feature_assignment import (
    FeatureAssignmentConstraints,
    FeatureAssignment,
    ResidualStreamState,
    solve,
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

    def get_strategies(
        self, nodes: Set[Node], existing_constraints: FeatureAssignmentConstraints
    ) -> List[GroupStrategy]:
        node_to_strategies = {
            node: self.get_strategies_for_node(node) for node in nodes
        }
        # For any given set of input/output nodes, we'll only keep one strategy.
        strategies = get_combined_strategies(node_to_strategies, existing_constraints)
        result = []
        for s in strategies:
            constraint = self.get_constraints(s)
            constraint.update(existing_constraints)
            if solve(constraint):
                result.append(s)
        return result

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
