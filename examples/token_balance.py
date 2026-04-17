"""Token balance checker — equal counts of 'a' and 'b'.

Parses a variable-length sequence of 'a' and 'b' tokens terminated by
'\\n', and outputs 'Y' if the counts are equal, 'N' otherwise.

This is an intermediate example that validates the parallel prefix sum
strategy (Hillis-Steele with position-gated OOB handling) before the
full balanced-parentheses checker builds on top of it.

    Input:  <bos> a b a b \n
    Output: Y

    Input:  <bos> a a b \n
    Output: N
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import add_scaled_nodes, compare, negate
from torchwright.ops.inout_nodes import (
    create_literal_value,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import bool_all_true, equals_vector
from torchwright.ops.map_select import select
from torchwright.ops.prefix_ops import prefix_sum
from torchwright.ops.sequence_ops import output_sequence

D_MODEL = 256


def create_network_parts(
    n_stages: int = 5,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the token-balance checker graph.

    Args:
        n_stages: Prefix-sum doubling stages (handles up to 2**n_stages positions).
    """
    vocab = list("abYN ") + ["\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()
    embed = embedding.get_embedding

    # --- Token classification ---
    is_a = equals_vector(embedding, embed("a"))
    is_b = equals_vector(embedding, embed("b"))
    # +1 for 'a', -1 for 'b', 0 for anything else
    contribution = add_scaled_nodes(0.5, is_a, -0.5, is_b)

    # --- Running count via parallel prefix sum ---
    depth = prefix_sum(pos_encoding, contribution, n_stages)

    # --- Check depth == 0 at every position ---
    depth_ge_zero = compare(depth, -0.5)
    depth_le_zero = compare(negate(depth), -0.5)
    depth_is_zero = bool_all_true([depth_ge_zero, depth_le_zero])

    # --- Output at trigger ---
    is_trigger = equals_vector(embedding, embed("\n"))
    result = select(
        depth_is_zero,
        create_literal_value(embed("Y")),
        create_literal_value(embed("N")),
    )
    output_node = output_sequence(
        pos_encoding,
        is_trigger,
        [result, create_literal_value(embed("<eos>"))],
        embed(" "),
    )
    return output_node, pos_encoding, embedding


def create_network() -> Unembedding:
    output_node, pos_encoding, embedding = create_network_parts()
    return create_unembedding(output_node, embedding)
