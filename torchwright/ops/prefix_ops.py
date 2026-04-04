"""Parallel prefix reductions across sequence positions.

These operations compute running aggregates (sum, AND) over a token
sequence using the Hillis-Steele parallel prefix algorithm.  Each stage
doubles the reach via ``attend_to_offset`` with power-of-2 offsets.

Early positions where the offset would land before BOS are handled by
extracting an approximate position index from the positional encoding
and gating the out-of-bounds result to the identity element (0 for sum,
True for AND).
"""

import torch

from torchwright.graph import Node
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import add, compare
from torchwright.ops.inout_nodes import create_constant
from torchwright.ops.logic_ops import bool_all_true, cond_gate
from torchwright.ops.map_select import select


def prefix_sum(pos_encoding: PosEncoding, value: Node, n_stages: int) -> Node:
    """Inclusive prefix sum of a 1D scalar value across positions.

    After ``n_stages`` stages, position *i* holds the sum of ``value``
    at positions 0 through *i*.  Handles up to ``2**n_stages`` positions.

    Args:
        pos_encoding: Positional encoding for attention operations.
        value: 1D scalar node to sum (one value per position).
        n_stages: Number of doubling stages.  Must satisfy
            ``2**n_stages >= max sequence length``.
    """
    position = pos_encoding.get_position_scalar()
    partial = value
    for k in range(n_stages):
        offset_value = pos_encoding.attend_to_offset(partial, delta_pos=-(2**k))
        is_in_bounds = compare(position, 2**k - 0.5)
        gated = cond_gate(is_in_bounds, offset_value)
        partial = add(partial, gated)
    return partial


def prefix_and(pos_encoding: PosEncoding, flag: Node, n_stages: int) -> Node:
    """Inclusive prefix AND of a boolean flag across positions.

    After ``n_stages`` stages, position *i* is True (1.0) only if
    ``flag`` is True at every position 0 through *i*.

    Args:
        pos_encoding: Positional encoding for attention operations.
        flag: 1D boolean node (1.0 = True, -1.0 = False).
        n_stages: Number of doubling stages.
    """
    position = pos_encoding.get_position_scalar()
    result = flag
    for k in range(n_stages):
        offset_flag = pos_encoding.attend_to_offset(result, delta_pos=-(2**k))
        is_in_bounds = compare(position, 2**k - 0.5)
        # OOB → True (1.0) so AND is unchanged
        safe_flag = select(
            is_in_bounds, offset_flag, create_constant(torch.tensor([1.0]))
        )
        result = bool_all_true([result, safe_flag])
    return result
