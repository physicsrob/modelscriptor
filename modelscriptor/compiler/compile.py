from enum import Enum
from typing import Dict, Set, Optional, List

import torch

from modelscriptor.graph import Node, Linear, Add, Attn, Constant
from modelscriptor.graph.relu import ReLU


class FCNPlan:
    # Plan for FCN, not counting skip.
    in_nodes: Set[Node]
    linear1_output_nodes: Set[Node]
    relu_output_nodes: Set[Node]
    linear2_output_nodes: Set[Node]

    def __init__(self):
        self.in_nodes = set()
        self.linear1_output_nodes = set()
        self.relu_output_nodes = set()
        self.linear2_output_nodes = set()
        self.res_output_nodes = set()


class FCNSkipPlan:
    in_nodes: Set[Node]
    fcn_plan: FCNPlan
    res_output_nodes: Set[Node]

    def __init__(self):
        self.in_nodes = set()
        self.res_output_nodes = set()
        self.fcn_plan = FCNPlan()


class Plan:
    layer_plans: List[FCNPlan]


def plan_fcn_layer(fcn_output_node: Node) -> Optional[FCNPlan]:
    # Fri 12:35pm -- I think this function actually works.
    plan = FCNPlan()

    current_node = fcn_output_node

    # Compile linear2 layer
    plan.linear2_output_nodes.add(current_node)
    if isinstance(current_node, Linear):
        current_node = current_node.inputs[0]
    # Otherwise identity matrix will be added, we don't go up the graph.

    # Compile relu layer
    if not isinstance(current_node, ReLU):
        return None

    plan.relu_output_nodes.add(current_node)
    current_node = current_node.inputs[0]

    # Compile linear1 layer
    plan.linear1_output_nodes.add(current_node)
    if isinstance(current_node, Linear):
        current_node = current_node.inputs[0]
    # Otherwise identity matrix will be added, we don't go up the graph.

    # Calculate in layer
    plan.in_nodes.add(current_node)
    return plan


def plan_fcn_skip_layer(output_node: Node) -> FCNSkipPlan:
    plan = FCNSkipPlan()

    if isinstance(output_node, Add):
        fcn_plan0 = plan_fcn_layer(output_node.inputs[0])

        if fcn_plan0:
            plan.in_nodes = fcn_plan0.in_nodes | {output_node.inputs[1]}
            plan.fcn_plan = fcn_plan0
            plan.res_output_nodes = output_node
            return plan

        fcn_plan1 = plan_fcn_layer(output_node.inputs[1])
        if fcn_plan0:
            plan.in_nodes = fcn_plan1.in_nodes | {output_node.inputs[0]}
            plan.fcn_plan = fcn_plan1
            plan.res_output_nodes = output_node
            return plan

        # Neither plan worked, which means we need to skip both branches of the add.
        plan.in_nodes.add(output_node)
        plan.res_output_nodes.add(output_node)
    else:
        # output_node is not an addition.  Try to compile it to the FCN
        fcn_plan = plan_fcn_layer(output_node)
        if fcn_plan:
            # We planned it
            plan.in_nodes = fcn_plan.in_nodes
            plan.fcn_plan = fcn_plan
            plan.res_output_nodes = output_node  # We'll add an Add(0) at compile time.
        else:
            # We can't plan anything!
            plan.in_nodes = {output_node}
            plan.res_output_nodes = {output_node}
            return plan


def compile(output_node: Node):
    # Strategy:  Compile from the bottom-up (output first)
    ...
    current_layer = FCNPlan()  ### FIXME
    node = output_node

    current_sub_layer = "res"
    if isinstance(node, Add):
        # We'll use the residual to perform the add.
        # TODO -- add a heuristic to better decide which branch of the add goes where
        current_layer.res_output_nodes.add(node)
        current_layer.in_nodes.add(node.inputs[0])
        current_layer.linear2_output_nodes.add(node.inputs[1])
    else:
        # To traverse the add layer, we need to insert an Add(0, node)
        zero_constant = Constant(torch.zeros(len(node)))
        current_layer.res_output_nodes.add(Add(zero_constant, node))
        current_layer.linear2_output_nodes.add(node)

    current_sub_layer = "linear2"
    current_nodes = current_layer.linear2_output_nodes
    for node in current_nodes:
        if isinstance(node, Linear):
            current_layer.relu_output_nodes.add(node.inputs[0])
        else:
            current_layer.relu_output_nodes.add(node)

    current_sub_layer = "relu"
    current_nodes = current_layer.relu_output_nodes
    for node in current_nodes:
        if isinstance(node, ReLU):
            current_layer.linear1_output_nodes.add(node.inputs[0])
        else:
            assert False  # Not compilable.
