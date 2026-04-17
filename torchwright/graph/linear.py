from typing import List, Dict, Optional
from torchwright.graph import Node
from torchwright.graph.value_type import (
    NodeValueType,
    is_integer_tensor,
    linear_output_range,
)

import torch


class Linear(Node):
    """Affine transform: ``y = x @ output_matrix + output_bias``.

    The compiler may realise this as either an MLP slice or an attention
    head attending to the current position, depending on context.

    Attributes:
        output_matrix: Weight matrix, shape ``(d_input, d_output)``.
        output_bias: Bias vector, shape ``(d_output,)``.
    """

    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        input_node: Node,
        output_matrix: torch.Tensor,
        output_bias: Optional[torch.Tensor] = None,
        name: str = "",
    ):
        # output_matrix shape (d_input, d_output)
        self.d_input = output_matrix.shape[0]
        self.d_output = output_matrix.shape[1]
        assert len(input_node) == self.d_input
        self.output_matrix = output_matrix

        if output_bias is None:
            self.output_bias = torch.zeros(self.d_output)
        else:
            assert len(output_bias) == self.d_output
            self.output_bias = output_bias

        super().__init__(self.d_output, [input_node], name=name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        value_in = self.inputs[0].compute(n_pos, input_values)

        assert value_in.shape == (n_pos, self.d_input)
        return torch.matmul(value_in, self.output_matrix) + self.output_bias

    def compute_value_type(self) -> NodeValueType:
        from torchwright.graph.misc import Concatenate
        from torchwright.graph.value_type import Range

        inp = self.inputs[0]
        inp_t = inp.value_type
        weights_int = is_integer_tensor(self.output_matrix)
        bias_int = is_integer_tensor(self.output_bias)
        # Preserve the input's guarantee level when weights and bias are integer.
        is_int = inp_t.is_integer if (weights_int and bias_int) else False

        # If the input is a Concatenate, each child slab has its own range;
        # using the Concatenate's scalar summary (union across all children)
        # hugely over-estimates the output range when the children have very
        # different ranges (e.g. ``[cond (-1,1), inp (-30,30)]``).
        # Do interval arithmetic slab-by-slab for a much tighter bound.
        if isinstance(inp, Concatenate) and inp.inputs:
            row = 0
            per_col_mins = None
            per_col_maxs = None
            for child in inp.flatten_inputs():
                child_range = child.value_type.value_range
                child_rows = len(child)
                child_matrix = self.output_matrix[row : row + child_rows]
                row += child_rows
                if not child_range.is_finite():
                    # Bail to the scalar summary path — this child dominates.
                    per_col_mins = None
                    break
                lo_prod = child_range.lo * child_matrix
                hi_prod = child_range.hi * child_matrix
                child_mins = torch.minimum(lo_prod, hi_prod).sum(dim=0)
                child_maxs = torch.maximum(lo_prod, hi_prod).sum(dim=0)
                if per_col_mins is None:
                    per_col_mins = child_mins
                    per_col_maxs = child_maxs
                else:
                    per_col_mins = per_col_mins + child_mins
                    per_col_maxs = per_col_maxs + child_maxs
            if per_col_mins is not None:
                per_col_mins = per_col_mins + self.output_bias
                per_col_maxs = per_col_maxs + self.output_bias
                out_range = Range(
                    float(per_col_mins.min().item()),
                    float(per_col_maxs.max().item()),
                )
                return NodeValueType(value_range=out_range, is_integer=is_int)

        out_range = linear_output_range(
            inp_t.value_range, self.output_matrix, self.output_bias
        )
        return NodeValueType(value_range=out_range, is_integer=is_int)

    def num_params(self):
        return self.d_input * self.d_output + self.d_output
