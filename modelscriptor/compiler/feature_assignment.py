from modelscriptor.graph import Node, Concatenate

from ortools.sat.python import cp_model  # type: ignore
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional

global_state_id = 0


class ResidualStreamState:
    state_id: int

    def __init__(self):
        global global_state_id
        self.state_id = global_state_id
        global_state_id += 1


class FeatureAssignment:
    mapping: Dict[ResidualStreamState, Dict[Node, List[int]]]

    def __init__(self):
        self.mapping = {}

    def assign(self, state: ResidualStreamState, node: Node, indices: List[int]):
        if state not in self.mapping:
            self.mapping[state] = {}
        self.mapping[state][node] = indices

    def duplicate_state(self, src: ResidualStreamState, dst: ResidualStreamState):
        self.mapping[dst] = self.mapping[src]

    def has_node(self, state: ResidualStreamState, node: Node) -> bool:
        return node in self.mapping[state]

    def get_node_indices(self, state: ResidualStreamState, node: Node) -> List[int]:
        return self.mapping[state][node]


def simplify_nodes(nodes: List[Node]) -> List[Node]:
    # Simplify all concatenations
    simplified_other_nodes = []
    for n in nodes:
        if isinstance(n, Concatenate):
            simplified_other_nodes += simplify_nodes(n.inputs)
        else:
            simplified_other_nodes.append(n)
    return simplified_other_nodes


class FeatureAssignmentConstraints:
    _state_to_nodes: Dict[ResidualStreamState, Set[Node]]
    _shared_feature_constraints: List[
        Tuple[ResidualStreamState, List[Node], ResidualStreamState, List[Node]]
    ]
    _equivalent: List[Tuple[ResidualStreamState, ResidualStreamState]]

    def __init__(self):
        self._state_to_nodes = defaultdict(set)
        self._states = set()
        self._shared_feature_constraints = []
        self._equivalent = []

    def add_node_to_state(self, node: Node, state: ResidualStreamState):
        self._state_to_nodes[state].add(node)
        if isinstance(node, Concatenate):
            simplified = simplify_nodes([node])
            self._state_to_nodes[state] |= set(simplified)
            self._shared_feature_constraints.append((state, [node], state, simplified))

    def add_shared_features_constraint(
        self,
        state1: ResidualStreamState,
        nodes1: Union[List[Node], Node],
        state2: ResidualStreamState,
        nodes2: Union[List[Node], Node],
    ):
        if not isinstance(nodes1, list):
            nodes1 = [nodes1]
        if not isinstance(nodes2, list):
            nodes2 = [nodes2]

        nodes1 = simplify_nodes(nodes1)
        nodes2 = simplify_nodes(nodes2)
        len1 = sum(len(n) for n in nodes1)
        len2 = sum(len(n) for n in nodes2)
        assert len1 == len2, f"Shared constraint of unequal lengths {len1} and {len2}"

        # Adds a constraint that when you concatenate the features from nodes1 in state1 together you will get the same features
        # as when you concatenate the features from nodes2 in state2.
        self._shared_feature_constraints.append((state1, nodes1, state2, nodes2))

    def add_equivalency(self, state1: ResidualStreamState, state2: ResidualStreamState):
        # Adds a constraint that specifies that state1 and state2 represent the same hting.
        # In other words, they share the same nodes, and all nodes have the same indices.
        self._equivalent.append((state1, state2))

    def get_simplified_state_mapping(
        self,
    ) -> Dict[ResidualStreamState, ResidualStreamState]:
        states = {k for k in self._state_to_nodes.keys()}
        states |= {s for s1, s2 in self._equivalent for s in (s1, s2)}

        # Build simplification of equivalents
        simplification = {state: state for state in states}

        for state1, state2 in self._equivalent:
            root_state1 = simplification[state1]
            root_state2 = simplification[state2]

            if root_state1 != root_state2:
                for state, root_state in simplification.items():
                    if root_state == root_state2:
                        simplification[state] = root_state1
        return simplification

    def get_consolidated_constraints(self):
        simplification = self.get_simplified_state_mapping()

        for state1, state2 in self._equivalent:
            root_state1 = simplification[state1]
            root_state2 = simplification[state2]

            if root_state1 != root_state2:
                for state, root_state in simplification.items():
                    if root_state == root_state2:
                        simplification[state] = root_state1

        state_to_nodes = {
            simplification[state]: node for state, node in self._state_to_nodes.items()
        }
        shared_feature_constraints = [
            (simplification[state1], nodes1, simplification[state2], nodes2)
            for state1, nodes1, state2, nodes2 in self._shared_feature_constraints
        ]
        return {
            "state_to_nodes": state_to_nodes,
            "shared_feature_constraints": shared_feature_constraints,
        }

    def check_solution(self, solution: FeatureAssignment) -> bool:
        # Check that each feature is used by at most one node within each state
        for state, node_to_features in solution.mapping.items():
            all_features = []
            for node, features in node_to_features.items():
                if not isinstance(node, Concatenate):
                    all_features.extend(features)

            if len(all_features) != len(set(all_features)):
                breakpoint()
                return False  #  Duplicate feature assignments in state

        # Check shared_feature_constraints
        for state1, nodes1, state2, nodes2 in self._shared_feature_constraints:
            features1 = []
            features2 = []

            for node in nodes1:
                if not solution.has_node(state1, node):
                    breakpoint()
                    return False
                features1.extend(solution.get_node_indices(state1, node))

            for node in nodes2:
                if not solution.has_node(state2, node):
                    breakpoint()
                    return False
                features2.extend(solution.get_node_indices(state2, node))

            if features1 != features2:
                breakpoint()
                return False

        # Check state equivalency
        for state1, state2 in self._equivalent:
            if solution.mapping[state1] != solution.mapping[state2]:
                breakpoint()
                return False

        return True

    @classmethod
    def merge(cls, constraints: List["FeatureAssignmentConstraints"]):
        result = cls()
        for c in constraints:
            result._state_to_nodes.update(c._state_to_nodes)
            result._shared_feature_constraints.extend(c._shared_feature_constraints)
            result._equivalent.extend(c._equivalent)
        return result


def solve(
    constraints: FeatureAssignmentConstraints, max_d: int = 1000
) -> Optional[FeatureAssignment]:
    model = cp_model.CpModel()
    node_to_interval_var = {}
    end_feature_indices = []
    consolidated_constriants = constraints.get_consolidated_constraints()
    state_to_nodes = consolidated_constriants["state_to_nodes"]
    shared_feature_constraints = consolidated_constriants["shared_feature_constraints"]

    # Step 1: Define Interval Variables
    for state, nodes in state_to_nodes.items():
        for node in nodes:
            start_var = model.NewIntVar(
                0, max_d, f"start_{state.state_id}_{node.node_id}"
            )
            end_var = model.NewIntVar(0, max_d, f"end_{state.state_id}_{node.node_id}")
            size_var = model.NewIntVar(
                len(node), len(node), f"size_{state.state_id}_{node.node_id}"
            )
            interval_var = model.NewIntervalVar(
                start_var,
                size_var,
                end_var,
                f"interval_{state.state_id}_{node.node_id}",
            )
            node_to_interval_var[(state, node)] = interval_var
            end_feature_indices.append(end_var)

    # Step 2: Non-Overlapping Constraints
    for state, nodes in state_to_nodes.items():
        intervals_in_state = [
            node_to_interval_var[(state, node)]
            for node in nodes
            if not isinstance(node, Concatenate)
        ]
        model.AddNoOverlap(intervals_in_state)

    # Step 3: Shared Feature Constraints
    for state1, nodes1, state2, nodes2 in shared_feature_constraints:
        first_node1 = nodes1[0]
        first_node2 = nodes2[0]

        # Enforce that the starting feature index for the first node in nodes1 is the same as the first node in nodes2
        model.Add(
            node_to_interval_var[(state1, first_node1)].StartExpr()
            == node_to_interval_var[(state2, first_node2)].StartExpr()
        )

        # Ensure nodes within each group are contiguous
        last_end_var1 = node_to_interval_var[(state1, first_node1)].EndExpr()
        last_end_var2 = node_to_interval_var[(state2, first_node2)].EndExpr()

        for next_node1 in nodes1[1:]:
            next_start_var1 = node_to_interval_var[(state1, next_node1)].StartExpr()
            model.Add(next_start_var1 == last_end_var1)
            last_end_var1 = node_to_interval_var[(state1, next_node1)].EndExpr()

        for next_node2 in nodes2[1:]:
            next_start_var2 = node_to_interval_var[(state2, next_node2)].StartExpr()
            model.Add(next_start_var2 == last_end_var2)
            last_end_var2 = node_to_interval_var[(state2, next_node2)].EndExpr()

    # Step 5: Define Objective
    max_end_var = model.NewIntVar(0, 1000, "")  # Adjust the upper limit as needed
    for state, nodes in constraints._state_to_nodes.items():
        for node in nodes:
            model.Add(max_end_var >= node_to_interval_var[(state, node)].EndExpr())
    model.Minimize(max_end_var)

    # Step 6: Solve the Model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Step 7: Populate Results
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        result = FeatureAssignment()
        for state, nodes in state_to_nodes.items():
            for node in nodes:
                interval_var = node_to_interval_var[(state, node)]
                start_feature_index = solver.Value(interval_var.StartExpr())
                indices = list(
                    range(start_feature_index, start_feature_index + len(node))
                )
                result.assign(state, node, indices)

        # Copy the solution to equivalent states.
        simplification = constraints.get_simplified_state_mapping()
        for state1, state2 in simplification.items():
            if state1 != state2:
                result.duplicate_state(state2, state1)

        return result
    else:
        return None
