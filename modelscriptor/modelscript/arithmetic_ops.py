from typing import List

from modelscriptor.graph import Node, Add, Concatenate, Linear
import torch

from modelscriptor.graph.relu import ReLU
from modelscriptor.modelscript.linear_relu_linear import linear_relu_linear

from modelscriptor.modelscript.const import step_sharpness, big_offset


def add_scalar(inp: Node, scalar: float) -> Node:
    """
    Adds a scalar value to each entry of the input node.

    Args:
        inp (Node): Node whose values will have the scalar added.
        scalar (float): Scalar value to add.

    Returns:
        Node: Output node with scalar added to each entry.
    """
    return Add(
        inp,
        linear_relu_linear(
            input_node=inp,
            input_proj=torch.tensor([0.0] * len(inp)),
            input_bias=torch.zeros(1),
            output_proj=torch.tensor([0.0] * len(inp)),
            output_bias=torch.tensor([scalar] * len(inp)),
            name="add_scalar_ffn",
        ),
        name="add_scalar_add",
    )


def add(inp1: Node, inp2: Node) -> Node:
    """
    Performs element-wise addition of two input nodes.

    Args:
        inp1 (Node): First node for addition.
        inp2 (Node): Second node for addition.

    Returns:
        Node: Node resulting from element-wise addition.
    """
    return Add(inp1, inp2)


def compare(
    inp: Node, thresh: float, true_level: float = 1.0, false_level: float = -1.0
) -> Node:
    """
    Compare input with threshold and return boolean valued node (1.0 for true, -1.0 for false)

    Args:
        inp: Node to compare. Must be length 1.
        thresh: Threshold to use.
        true_level: Value to return if inp is greater than thresh.
        false_level: Value to return if inp is less than thresh.


    Returns:
        Node: Node with a value of true_level if inp is greater than thresh, false_level otherwise.
    """

    # We need 2 FFN entries, we'll use the equation:
    # y= (true_level-false_level) * [
    #   max(step_sharpness*x - step_sharpness*thresh, 0) - max(step_sharpness*x - step_sharpness*thresh - 1, 0)
    # ] + false_level

    d_input = len(inp)

    input_proj = torch.tensor([[step_sharpness], [step_sharpness]])
    input_bias = torch.tensor([-step_sharpness * thresh, -step_sharpness * thresh - 1.0])
    output_proj = torch.tensor([[true_level - false_level], [false_level - true_level]])
    output_bias = false_level * torch.ones(1)

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def concat(inp_list: List[Node]) -> Node:
    """
    Concatenates all the Nodes in inp_list.

    Args:
        inp_list (List[Node]): List of nodes to concatenate

    Returns:
        Node: Node resulting from concatenation
    """
    return Concatenate(inp_list)


def add_scaled_nodes(scale1: float, inp1: Node, scale2: float, inp2: Node) -> Node:
    """
    Computes the linear combination of two nodes using specified coefficients.

    Args:
        scale1 (float): Coefficient for the first node.
        inp1 (Node): First node.
        scale2 (float): Coefficient for the second node.
        inp2 (Node): Second node.

    Returns:
        Node: Node resulting from the linear combination of input nodes.
    """
    assert len(inp1) == len(inp2)
    d = len(inp1)

    concat = Concatenate([inp1, inp2])
    M = torch.zeros(len(concat), d)
    for i in range(d):
        M[i, i] = scale1
        M[d + i, i] = scale2

    return Linear(concat, M)


def sum_nodes(inp_list: List[Node]) -> Node:
    """
    Computes the sum of all input nodes.

    Args:
        inp_list (List[Node]): List of nodes to be summed.

    Returns:
        Node: Node with the summed value of input nodes.
    """
    d_values = {len(node) for node in inp_list}
    assert len(d_values) == 1
    d = d_values.pop()
    x = Concatenate(inp_list)
    output_matrix = torch.zeros(len(x), d)
    for i in range(len(x)):
        output_matrix[i, i % d] = 1.0

    return Linear(input_node=x, output_matrix=output_matrix)


def relu(inp: Node) -> Node:
    """
    Applies the Rectified Linear Unit (ReLU) function to the input node.

    Args:
        inp (Node): Node to apply ReLU.

    Returns:
        Node: Node with ReLU applied.
    """
    return ReLU(inp)


def relu_add(inp1: Node, inp2: Node) -> Node:
    """
    Applies the ReLU function to both input nodes and then adds them together.

    Args:
        inp1 (Node): First node for ReLU and addition.
        inp2 (Node): Second node for ReLU and addition.

    Returns:
        Node: Node resulting from ReLU application and addition.
    """
    # Rectifies val1 and val2 and then adds them together.
    # Equivalent to torch.clamp(val1, min=0) + torch.clamp(val2, min=0)
    assert len(inp1) == len(inp2)
    x = Concatenate([inp1, inp2])

    input_proj = torch.eye(len(x))
    input_bias = torch.zeros(len(x))
    output_proj = torch.zeros((len(x), len(inp1)))
    output_bias = torch.zeros(len(inp1))

    for i in range(len(inp1)):
        output_proj[i, i] = 1.0
        output_proj[len(inp1) + i, i] = 1.0

    return linear_relu_linear(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
    )


def negate(inp: Node) -> Node:
    """
    Negates the input node (multiplies by -1).

    Args:
        inp (Node): Node to negate.

    Returns:
        Node: Node with negated values.
    """
    d = len(inp)
    return Linear(inp, -torch.eye(d), name="negate")


def subtract(inp1: Node, inp2: Node) -> Node:
    """
    Subtracts inp2 from inp1 element-wise.

    Args:
        inp1 (Node): Node to subtract from.
        inp2 (Node): Node to subtract.

    Returns:
        Node: Node resulting from inp1 - inp2.
    """
    return add(inp1, negate(inp2))


def multiply_scalar(inp: Node, scalar: float) -> Node:
    """
    Multiplies each entry of the input node by a scalar.

    Args:
        inp (Node): Node to scale.
        scalar (float): Scalar multiplier.

    Returns:
        Node: Node with scaled values.
    """
    d = len(inp)
    return Linear(inp, scalar * torch.eye(d), name="multiply_scalar")


def thermometer_square(inp: Node, max_value: int) -> Node:
    """Compute x² for a non-negative integer x in [0, max_value].

    Uses the odd-number identity: x² = 1 + 3 + 5 + ... + (2x-1).
    Place a detector at each integer k = 1..max_value. Each detector
    contributes (2k-1) when x >= k, so the sum is x².

        x=0 → no detectors fire                → 0
        x=3 → detectors 1,2,3 fire: 1+3+5      → 9
        x=5 → detectors 1..5 fire: 1+3+5+7+9   → 25

    Same paired-ReLU step as thermometer_floor_div in adder_v2, but
    the output weight is (2k-1) instead of 1.0.

    Only exact for non-negative integers. For non-integers, the step
    functions fire based on floor(x), giving floor(x)² instead of x².

    Args:
        inp: 1D scalar node with integer value in [0, max_value].
        max_value: Upper bound on input (determines number of detectors).

    Returns:
        1D scalar node containing x².
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value >= 1, "max_value must be at least 1"

    d_int = 2 * max_value  # Two ReLU units per threshold detector
    input_proj = torch.zeros(d_int, 1)
    input_bias = torch.zeros(d_int)
    output_proj = torch.zeros(d_int, 1)

    for k in range(1, max_value + 1):
        threshold = k - 0.5  # Half-integer: 0.5, 1.5, 2.5, ...
        row = 2 * (k - 1)

        # Paired ReLU: ReLU(s*x - s*threshold) - ReLU(s*x - s*threshold - 1)
        input_proj[row, 0] = step_sharpness
        input_proj[row + 1, 0] = step_sharpness
        input_bias[row] = -step_sharpness * threshold
        input_bias[row + 1] = -step_sharpness * threshold - 1.0

        # Odd-number weight: detector k contributes (2k-1) instead of 1.0
        weight = 2.0 * k - 1.0
        output_proj[row, 0] = weight
        output_proj[row + 1, 0] = -weight

    return linear_relu_linear(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=torch.zeros(1),
        name="thermometer_square",
    )


def multiply_integers(inp1: Node, inp2: Node, max_value: int) -> Node:
    """Multiply two non-negative integer scalars using the polarization identity.

    a * b = ((a+b)² - (a-b)²) / 4

    Both inputs must be 1D scalar nodes with integer values in [0, max_value].
    This identity is exact for all reals, but thermometer_square is only exact
    for non-negative integers, so the inputs must be integers.

    Implementation:
        s = a + b                          range [0, 2*max_value], Linear (free)
        d = a - b                          range [-max_value, max_value], Linear (free)
        |d| = ReLU(d) + ReLU(-d)           1 FFN layer (relu_add)
        s² = thermometer_square(s)         1 FFN layer
        |d|² = thermometer_square(|d|)     1 FFN layer
        result = (s² - |d|²) / 4          Linear (free)

    Total cost: 3 FFN layers.

    Args:
        inp1: 1D scalar node, integer value in [0, max_value].
        inp2: 1D scalar node, integer value in [0, max_value].
        max_value: Upper bound on each input.

    Returns:
        1D scalar node containing inp1 * inp2.
    """
    assert len(inp1) == 1, "Input must be a 1D scalar node"
    assert len(inp2) == 1, "Input must be a 1D scalar node"

    s = add(inp1, inp2)  # a+b
    d = subtract(inp1, inp2)  # a-b (may be negative)
    abs_d = relu_add(d, negate(d))  # |a-b| = ReLU(d) + ReLU(-d)

    sq_sum = thermometer_square(s, 2 * max_value)  # (a+b)²
    sq_diff = thermometer_square(abs_d, max_value)  # (a-b)²

    # a*b = ((a+b)² - (a-b)²) / 4
    return add_scaled_nodes(0.25, sq_sum, -0.25, sq_diff)
