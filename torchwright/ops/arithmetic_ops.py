from typing import List

from torchwright.graph import Node, Add, Concatenate, Linear
import torch

from torchwright.graph.relu import ReLU
from torchwright.ops.linear_relu_linear import linear_relu_linear

from torchwright.ops.const import step_sharpness, big_offset


def add_const(inp: Node, scalar: float) -> Node:
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
            name="add_const_ffn",
        ),
        name="add_const_add",
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


def multiply_const(inp: Node, scalar: float) -> Node:
    """
    Multiplies each entry of the input node by a scalar.

    Args:
        inp (Node): Node to scale.
        scalar (float): Scalar multiplier.

    Returns:
        Node: Node with scaled values.
    """
    d = len(inp)
    return Linear(inp, scalar * torch.eye(d), name="multiply_const")


def thermometer_floor_div(inp: Node, divisor: int, max_value: int) -> Node:
    """Compute floor(inp / divisor) using a piecewise-linear staircase.

    Places a steep ramp at each multiple of the divisor.  Half-integer
    thresholds (9.5 not 10.0) ensure clean separation for integer inputs.

        x = 35, divisor = 10  →  output = 3 = floor(35/10)

    Implemented via :func:`piecewise_linear` with a staircase whose
    transition width is ``1 / step_sharpness``.

    Args:
        inp: 1D scalar node with integer value in [0, max_value].
        divisor: The divisor for floor division.
        max_value: Upper bound on input (determines number of steps).

    Returns:
        1D scalar node containing floor(inp / divisor).
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = max_value // divisor
    if n == 0:
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([0.0]))

    # Build staircase breakpoints: each step is a steep ramp of width
    # 1/step_sharpness centred on the half-integer threshold.
    eps = 1.0 / step_sharpness
    breakpoints = [0.0 - eps]  # clamp region before first step
    values = [0.0]
    for k in range(1, n + 1):
        threshold = k * divisor - 0.5  # Half-integer: 9.5, 19.5, ...
        breakpoints.extend([threshold - eps / 2, threshold + eps / 2])
        values.extend([float(k - 1), float(k)])
    # Final clamp point beyond last step
    breakpoints.append(max_value + eps)
    values.append(float(n))

    return piecewise_linear(
        inp, breakpoints, values, input_scale=step_sharpness,
        name="thermometer_floor_div",
    )


def piecewise_linear(
    inp: Node,
    breakpoints: List[float],
    values: List[float],
    clamp: bool = True,
    d_max: int = 1024,
    input_scale: float = 1.0,
    name: str = "piecewise_linear",
) -> Node:
    """Evaluate a piecewise-linear function defined by (x, y) breakpoints.

    Linearly interpolates between consecutive breakpoints.  By default the
    output is clamped to ``values[0]`` / ``values[-1]`` outside the
    breakpoint range.  Set ``clamp=False`` to extrapolate using the first
    and last segment slopes instead.

    Implemented as a sum of ReLU slope-changes::

        f(x) = y_0 + Σ delta_m_i · ReLU(x - x_i)

    where ``delta_m_i`` is the change in slope at breakpoint ``x_i``.
    Uses one ReLU unit per slope change, so segments with equal slopes
    are free.

    Args:
        inp: 1D scalar node.
        breakpoints: Strictly ascending x-coordinates (length n >= 2).
        values: Corresponding y-values (same length as breakpoints).
        clamp: If True (default), hold constant outside the range.
            If False, extrapolate linearly.
        d_max: Maximum ReLU units per FFN layer (chunks beyond this).
        input_scale: Multiplier for the first-layer weights.  Each ReLU
            ``delta · ReLU(x - b)`` is rewritten as
            ``(delta/s) · ReLU(s·x - s·b)`` so the intermediate
            activations are amplified while the mathematical result is
            unchanged.  This matters when breakpoints are large: the
            bias ``-b`` may not be exact in float32, but ``-s·b`` can
            be (e.g. ``-999.45`` rounds, but ``-9994.5`` is exact).
            Use ``step_sharpness`` for step-function staircases where
            precision compounds across chained operations.
        name: Debug label prefix.

    Returns:
        1D scalar node containing f(inp).
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    n = len(breakpoints)
    assert n == len(values) and n >= 2, "Need >= 2 breakpoints with matching values"
    assert all(
        breakpoints[i] < breakpoints[i + 1] for i in range(n - 1)
    ), "Breakpoints must be strictly ascending"

    slopes = [
        (values[i + 1] - values[i]) / (breakpoints[i + 1] - breakpoints[i])
        for i in range(n - 1)
    ]

    # Build (input_weight, threshold, output_weight) triples for each ReLU.
    relus: list = []

    # Always start with the clamped representation: slope starts at 0
    # before x_0, changes at each breakpoint, cancelled to 0 after x_{n-1}.
    prev_slope = 0.0
    for i in range(n - 1):
        delta = slopes[i] - prev_slope
        if abs(delta) > 1e-12:
            relus.append((1.0, breakpoints[i], delta))
        prev_slope = slopes[i]
    if abs(prev_slope) > 1e-12:
        relus.append((1.0, breakpoints[-1], -prev_slope))

    if not clamp:
        # Add tail ReLUs to restore linear extrapolation beyond the range.
        # Left: ReLU(-(x - x_0)) active when x < x_0, adds slope m_0.
        if abs(slopes[0]) > 1e-12:
            relus.append((-1.0, breakpoints[0], -slopes[0]))
        # Right: ReLU(x - x_{n-1}) active when x > x_{n-1}, adds slope m_{n-2}.
        if abs(slopes[-1]) > 1e-12:
            relus.append((1.0, breakpoints[-1], slopes[-1]))

    if len(relus) == 0:
        # Constant function
        from torchwright.ops.inout_nodes import create_literal_value

        return create_literal_value(torch.tensor([values[0]]))

    chunks = []
    for chunk_start in range(0, len(relus), d_max):
        chunk = relus[chunk_start : chunk_start + d_max]
        d = len(chunk)

        s = input_scale
        input_proj = torch.tensor([[r[0] * s] for r in chunk])  # (d, 1)
        input_bias = torch.tensor([-r[0] * s * r[1] for r in chunk])  # (d,)
        output_proj = torch.tensor([[r[2] / s] for r in chunk])  # (d, 1)

        # Only the first chunk carries the output bias (y_0).
        ob = torch.tensor([values[0]]) if chunk_start == 0 else torch.zeros(1)

        chunks.append(
            linear_relu_linear(
                input_node=inp,
                input_proj=input_proj,
                input_bias=input_bias,
                output_proj=output_proj,
                output_bias=ob,
                name=f"{name}_{chunk_start}_{chunk_start + d}",
            )
        )

    if len(chunks) == 1:
        return chunks[0]
    return sum_nodes(chunks)


def square(inp: Node, max_value: float, step: float = 1.0, d_max: int = 1024) -> Node:
    """Compute x² via piecewise-linear interpolation.

    Exact when x is a multiple of ``step``. Between grid points the
    result is a piecewise-linear interpolation (always an underestimate
    of x²).

    Implemented via :func:`piecewise_linear` with breakpoints at
    multiples of ``step`` from 0 to ``max_value``.

    Args:
        inp: 1D scalar node with value in [0, max_value].
        max_value: Upper bound on input.
        step: Grid spacing. Exact for multiples of step. Smaller values
            give better accuracy for non-grid inputs at the cost of more
            breakpoints.
        d_max: Maximum ReLU units per FFN layer. When more breakpoints
            are needed, they are split into chunks of this size.

    Returns:
        1D scalar node containing x² (exact at grid points).
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    assert max_value > 0, "max_value must be positive"

    breakpoints = []
    vals = []
    x = 0.0
    while x <= max_value + step / 2.0:
        breakpoints.append(x)
        vals.append(x * x)
        x += step

    return piecewise_linear(inp, breakpoints, vals, d_max=d_max, name="square")


def multiply_integers(inp1: Node, inp2: Node, max_value: int) -> Node:
    """Multiply two non-negative integer scalars using the polarization identity.

    a * b = ((a+b)² - (a-b)²) / 4

    Both inputs must be 1D scalar nodes with integer values in [0, max_value].
    The polarization identity is exact for all reals, but ``square`` is only
    exact at grid points (integers by default), so the inputs must be integers.

    Implementation:
        s = a + b                          range [0, 2*max_value], Linear (free)
        d = a - b                          range [-max_value, max_value], Linear (free)
        |d| = ReLU(d) + ReLU(-d)           1 FFN layer (relu_add)
        s² = square(s)                     1+ FFN layers
        |d|² = square(|d|)                 1+ FFN layers
        result = (s² - |d|²) / 4          Linear (free)

    Total cost: 3+ FFN layers.

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

    sq_sum = square(s, 2 * max_value)  # (a+b)²
    sq_diff = square(abs_d, max_value)  # (a-b)²

    # a*b = ((a+b)² - (a-b)²) / 4
    return add_scaled_nodes(0.25, sq_sum, -0.25, sq_diff)
