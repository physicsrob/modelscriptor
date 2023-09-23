import torch

from modelscriptor.compiler.layer_plan import FFNPlan
from modelscriptor.graph import Linear, ReLU
from modelscriptor.modelscript.inout_nodes import create_input


def test_net():
    input_node = create_input("test", 10)

    linear1 = Linear(input_node, torch.zeros(10, 10), torch.zeros(10))
    relu_out = ReLU(linear1)
    linear2 = Linear(relu_out, torch.zeros(10, 10), torch.zeros(10))

    plan = FFNPlan()
    plan.add_output(linear2)
    plan.add_output(relu_out)
    plan.print()


def test_net2():
    input_node = create_input("test", 10)

    linear1 = Linear(input_node, torch.zeros(10, 10), torch.zeros(10))
    relu_out = ReLU(linear1)

    plan = FFNPlan()
    plan.add_output(relu_out)
    plan.print()
