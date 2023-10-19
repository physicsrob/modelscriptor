from modelscriptor.compiler.feature_assignment import (
    solve,
    FeatureAssignmentConstraints,
    ResidualStreamState,
)
from modelscriptor.graph import Node
from modelscriptor.modelscript.arithmetic_ops import concat


def test1():
    constraints = FeatureAssignmentConstraints()
    node1 = Node(5, [])
    node2 = Node(5, [])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()
    state3 = ResidualStreamState()

    # constraints.add_node_to_state(node1, state1)
    constraints.add_node_to_state(node1, state2)
    constraints.add_node_to_state(node2, state3)
    constraints.add_shared_features_constraint(state2, [node1], state3, [node2])

    solution = solve(constraints)
    print(solution)
    assert constraints.check_solution(solution)


def test2():
    constraints = FeatureAssignmentConstraints()
    node1 = Node(8, [])
    node2 = Node(4, [])
    node3 = Node(4, [])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()
    state3 = ResidualStreamState()
    state4 = ResidualStreamState()

    constraints.add_node_to_state(node1, state1)
    constraints.add_node_to_state(node2, state2)
    constraints.add_node_to_state(node3, state2)
    constraints.add_shared_features_constraint(state1, [node1], state2, [node2, node3])
    constraints.add_equivalency(state2, state3)
    constraints.add_equivalency(state3, state4)

    solution = solve(constraints)
    assert solution
    print(solution)
    assert constraints.check_solution(solution)


def test3():
    constraints = FeatureAssignmentConstraints()
    node1 = Node(8, [])
    node2 = Node(4, [])
    node3 = Node(4, [])
    node4 = Node(2, [])
    node5 = Node(2, [])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()

    constraints.add_node_to_state(node1, state1)
    constraints.add_node_to_state(node2, state2)
    constraints.add_node_to_state(node4, state2)
    constraints.add_node_to_state(node3, state2)
    constraints.add_node_to_state(node5, state2)
    constraints.add_shared_features_constraint(state1, [node1], state2, [node2, node3])

    solution = solve(constraints)
    assert solution
    print(solution)
    assert constraints.check_solution(solution)


def test4():
    constraints = FeatureAssignmentConstraints()
    node1 = Node(1, [])
    node2 = Node(2, [])
    node3 = Node(3, [])
    node4 = Node(4, [])
    cat = concat([node1, node2, node3, node4])
    state1 = ResidualStreamState()
    state2 = ResidualStreamState()

    constraints.add_node_to_state(cat, state1)
    constraints.add_node_to_state(node1, state2)
    # constraints.add_shared_features_constraint(state1, [node1], state2, [node2, node3])

    solution = solve(constraints)
    assert solution
    print(solution)
    assert constraints.check_solution(solution)


def test5():
    # state_in: Node1, Node2, Concat(Node1, Node2), Node 3
    # state_linear1: Linear(Node3)
    # state_relu: ReluLinear(Node3)
    # state_linear2: LinearReluLinear(Node3)
    # state_skip: Add(Concat(Node1, Node2), LinearReluLinear(Node3))

    constraints = FeatureAssignmentConstraints()
    node1 = Node(1, [])
    node2 = Node(1, [])
    node3 = Node(2, [])
    lnode3 = Node(2, [])
    rlnode3 = Node(2, [])
    lrlnode3 = Node(2, [])
    add = Node(2, [])
    cat = concat([node1, node2])

    # State_in:
    state_in = ResidualStreamState()
    constraints.add_node_to_state(cat, state_in)
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

    # state_linear3:
    state_linear2 = ResidualStreamState()
    constraints.add_node_to_state(lrlnode3, state_linear2)

    # state_skip
    state_skip = ResidualStreamState()
    constraints.add_node_to_state(add, state_skip)
    constraints.add_shared_features_constraint(state_skip, add, state_linear2, lrlnode3)
    constraints.add_shared_features_constraint(state_skip, add, state_in, cat)

    solution = solve(constraints)
    assert solution
    print(solution)
    assert constraints.check_solution(solution)
