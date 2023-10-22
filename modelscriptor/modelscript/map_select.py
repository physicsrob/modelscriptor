from modelscriptor.graph import Node, Concatenate
from typing import List, Tuple, Dict
import torch

from modelscriptor.modelscript.const import big_offset, turn_on_speed
from modelscriptor.modelscript.logic_ops import cond_add_vector
from modelscriptor.modelscript.ffn_layer import ffn_layer


def map_to_table(
    inp: Node, key_to_value: Dict[torch.Tensor, torch.Tensor], default: torch.Tensor
) -> Node:
    """
    Maps the value of the input node to a lookup table.

    Args:
        inp (Node): Node whose values will be looked up.
        key_to_value (Dict[torch.Tensor, torch.Tensor]): Lookup table mapping from keys to values.
        default (torch.Tensor): Default tensor to return if the input value doesn't exist in the table.

    Returns:
        Node: Output node with mapped values.
    """
    d_keys = {len(x) for x in key_to_value.keys()}
    d_values = {len(x) for x in key_to_value.values()}
    assert len(d_keys) == 1
    assert len(d_values) == 1
    d_key = d_keys.pop()
    d_value = d_values.pop()
    assert len(inp) == d_key
    assert len(default) == d_value

    d_int = len(key_to_value)
    # We'll use 1 FFN entry per item in the table, and an overall output bias of the default value
    # So roughly speaking:
    # input_proj will be (d_int x d_key), where input_proj[i, :] = table.keys()[i]
    # input_bias will be (d_int), where input_bias[i] = 1.0/turn_on_speed - (table.keys()[i] @ table.keys()[i])
    # output_proj will be (d_int, d_value), where output_proj[i, :] = turn_on_speed * (table.values()[i] - default)
    # output_bias will be (d_value), equal to default

    input_proj = torch.zeros(d_int, d_key)
    input_bias = torch.zeros(d_int)
    output_proj = torch.zeros(d_int, d_value)

    for i, (key, value) in enumerate(key_to_value.items()):
        input_proj[i, :] = key
        input_bias[i] = 1.0 / turn_on_speed - (key @ key)
        output_proj[i, :] = turn_on_speed * (value - default)

    return ffn_layer(
        input_node=inp,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=default,
    )


def select_from_list(
    cond_value_list: List[Tuple[Node, Node]], default: torch.Tensor
) -> Node:
    """
    Uses a list of conditions to determine which value to select from a table.

    Args:
        cond_value_list (List[Tuple[Node, Node]]): List of tuples where each tuple consists of a condition and its
            corresponding value.
        default (torch.Tensor): Default tensor to return if none of the conditions in the table are met.

    Returns:
        Node: Output node with the selected value based on the conditions.
    """
    raise NotImplementedError()


def select(cond: Node, true_node: Node, false_node: Node) -> Node:
    """
    Outputs one of two nodes based on a boolean condition.

    Args:
        cond (Node): Condition node that outputs either true or false.
        true_node (Node): Node to be outputted if the condition is true.
        false_node (Node): Node to be outputted if the condition is false.

    Returns:
        Node: Either true_node or false_node based on the condition.
    """
    assert len(cond) == 1  # Condition must be length 1
    assert len(true_node) == len(false_node)

    # Strategy:
    # - Concatenate(true_node; false_node)
    # - Add [offset ... offset;  -offset ... -offset] if cond is true
    # - Add [-offset ... -offset;  offset ... offset] if cond is false
    # - Rectify
    # - Merge the two halves by summing
    # - Subtract offset
    true_offset = torch.tensor(
        ([big_offset] * len(true_node)) + ([-big_offset] * len(true_node))
    )
    false_offset = torch.tensor(
        ([-big_offset] * len(true_node)) + ([big_offset] * len(true_node))
    )
    x: Node = Concatenate([true_node, false_node])
    x = cond_add_vector(cond, x, true_offset, false_offset)

    # Splits the input node into two equal-length vectors, apply ReLU to each,
    # and sum them with offset.
    input_proj = torch.eye(len(x))
    input_bias = torch.zeros(len(x))
    output_proj = torch.zeros((len(x), len(true_node)))
    output_bias = torch.tensor([-big_offset] * len(true_node))

    for i in range(len(true_node)):
        output_proj[i, i] = 1.0
        output_proj[len(true_node) + i, i] = 1.0

    return ffn_layer(
        input_node=x,
        input_proj=input_proj,
        input_bias=input_bias,
        output_proj=output_proj,
        output_bias=output_bias,
        name="select",
    )
