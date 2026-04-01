from modelscriptor.compiler.feature_assignment import (
    solve,
    FeatureAssignmentConstraints,
    ResidualStreamState,
)
from modelscriptor.graph import Node
from modelscriptor.modelscript.arithmetic_ops import concat


def test_single_node_shared_features():
    """Two nodes in different states must occupy the same columns (skip connection)."""
    constraints = FeatureAssignmentConstraints()
    node1 = Node(5, [])
    node2 = Node(5, [])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()

    constraints.add_node_to_state(node1, state1)
    constraints.add_node_to_state(node2, state2)
    constraints.add_shared_features_constraint(state1, [node1], state2, [node2])

    solution = solve(constraints)
    assert solution
    assert constraints.check_solution(solution)


def test_non_overlapping_with_equivalency():
    """Multiple nodes in equivalent states must not overlap."""
    constraints = FeatureAssignmentConstraints()
    node1 = Node(4, [])
    node2 = Node(4, [])
    node3 = Node(4, [])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()
    state3 = ResidualStreamState()

    constraints.add_node_to_state(node1, state1)
    constraints.add_node_to_state(node2, state1)
    constraints.add_node_to_state(node3, state1)
    constraints.add_equivalency(state1, state2)
    constraints.add_equivalency(state2, state3)

    solution = solve(constraints)
    assert solution
    assert constraints.check_solution(solution)

    # Verify non-overlapping
    indices1 = solution.get_node_indices(state1, node1)
    indices2 = solution.get_node_indices(state1, node2)
    indices3 = solution.get_node_indices(state1, node3)
    assert not set(indices1) & set(indices2)
    assert not set(indices1) & set(indices3)
    assert not set(indices2) & set(indices3)


def test_many_nodes_non_overlapping():
    """Many nodes in the same state must all get non-overlapping assignments."""
    constraints = FeatureAssignmentConstraints()
    nodes = [Node(2, []) for _ in range(10)]
    state = ResidualStreamState()

    for node in nodes:
        constraints.add_node_to_state(node, state)

    solution = solve(constraints)
    assert solution
    assert constraints.check_solution(solution)

    # Verify all non-overlapping
    all_indices = [set(solution.get_node_indices(state, n)) for n in nodes]
    for i in range(len(all_indices)):
        for j in range(i + 1, len(all_indices)):
            assert not all_indices[i] & all_indices[j]


def test_concat_adds_children_individually():
    """Concatenate nodes should add their children to the state, not themselves."""
    constraints = FeatureAssignmentConstraints()
    node1 = Node(1, [])
    node2 = Node(2, [])
    node3 = Node(3, [])
    node4 = Node(4, [])
    cat = concat([node1, node2, node3, node4])
    state = ResidualStreamState()

    constraints.add_node_to_state(cat, state)

    solution = solve(constraints)
    assert solution
    assert constraints.check_solution(solution)

    # All children should be individually assigned
    for node in [node1, node2, node3, node4]:
        assert solution.has_node(state, node)


def test_skip_constraint_with_concat_input():
    """Simulates a skip connection scenario: add = in_node + skip_node,
    where the constraint is single-node shared features (as the real compiler produces)."""
    constraints = FeatureAssignmentConstraints()
    node1 = Node(1, [])
    node2 = Node(1, [])
    node3 = Node(2, [])
    lnode3 = Node(2, [])
    rlnode3 = Node(2, [])
    lrlnode3 = Node(2, [])
    add = Node(2, [])
    cat = concat([node1, node2])

    # State_in: has node1, node2 (from cat), and node3
    state_in = ResidualStreamState()
    constraints.add_node_to_state(cat, state_in)  # adds node1, node2 individually
    constraints.add_node_to_state(node3, state_in)

    # State_linear1:
    state_linear1 = ResidualStreamState()
    constraints.add_node_to_state(lnode3, state_linear1)

    # State_relu:
    state_relu = ResidualStreamState()
    constraints.add_node_to_state(rlnode3, state_relu)
    constraints.add_shared_features_constraint(
        state_relu, rlnode3, state_linear1, lnode3
    )

    # state_linear2:
    state_linear2 = ResidualStreamState()
    constraints.add_node_to_state(lrlnode3, state_linear2)

    # state_skip: add occupies same columns as lrlnode3 (skip constraint)
    state_skip = ResidualStreamState()
    constraints.add_node_to_state(add, state_skip)
    constraints.add_shared_features_constraint(state_skip, add, state_linear2, lrlnode3)

    solution = solve(constraints)
    assert solution
    assert constraints.check_solution(solution)
