"""Balanced parentheses checker.

Parses a variable-length sequence of '(' and ')' tokens terminated by
'\\n', and outputs 'Y' if the parentheses are balanced, 'N' otherwise.

A sequence is balanced when:
  1. The total number of '(' equals the total number of ')'.
  2. At no point does the running count of ')' exceed '('.

Uses a parallel prefix sum (Hillis-Steele) to compute nesting depth at
every position, and a parallel prefix AND to verify no intermediate
position underflows below zero.

    Input:  <bos> ( ( ) ) \n
    Output: Y

    Input:  <bos> ) ( \n
    Output: N  (underflow at position 1)
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import add_scaled_nodes, compare, negate
from torchwright.ops.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import bool_all_true, equals_vector
from torchwright.ops.map_select import select
from torchwright.ops.prefix_ops import prefix_and, prefix_sum
from torchwright.ops.sequence_ops import output_sequence


D_MODEL = 1024


def create_network_parts(
    n_stages: int = 5,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the balanced-parentheses checker graph.

    Args:
        n_stages: Prefix-sum doubling stages (handles up to 2**n_stages positions).
    """
    vocab = list("()YN ") + ["\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()
    embed = embedding.get_embedding

    # --- Token classification ---
    is_open = equals_vector(embedding, embed("("))
    is_close = equals_vector(embedding, embed(")"))
    # +1 for '(', -1 for ')', 0 for anything else
    contribution = add_scaled_nodes(0.5, is_open, -0.5, is_close)

    # --- Running depth via parallel prefix sum ---
    depth = prefix_sum(pos_encoding, contribution, n_stages)

    # --- Underflow detection via parallel prefix AND ---
    not_underflowed = compare(depth, -0.5)  # True if depth >= 0
    all_valid = prefix_and(pos_encoding, not_underflowed, n_stages)

    # --- Check depth == 0 ---
    depth_ge_zero = compare(depth, -0.5)
    depth_le_zero = compare(negate(depth), -0.5)
    depth_is_zero = bool_all_true([depth_ge_zero, depth_le_zero])

    # --- Balanced = depth is zero AND no underflow anywhere ---
    is_balanced = bool_all_true([depth_is_zero, all_valid])

    # --- Output at trigger ---
    is_trigger = equals_vector(embedding, embed("\n"))
    result = select(
        is_balanced,
        create_constant(embed("Y")),
        create_constant(embed("N")),
    )
    output_node = output_sequence(
        pos_encoding,
        is_trigger,
        [result, create_constant(embed("<eos>"))],
        embed(" "),
    )
    return output_node, pos_encoding, embedding


def create_network() -> Unembedding:
    output_node, pos_encoding, embedding = create_network_parts()
    return create_unembedding(output_node, embedding)
