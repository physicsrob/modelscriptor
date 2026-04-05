"""Caesar cipher with runtime shift parameter.

Parses "<shift> <letters>\\n" where <shift> is a single digit (0-9) and
<letters> is a sequence of lowercase letters (fixed length). Outputs each
letter shifted forward by the given amount modulo 26.

    Input:  <bos> 3 h e l l o \\n
    Output: k h o o r

The shift is a runtime parameter — the same compiled transformer handles
any shift 0-9. This demonstrates per-token transformation with cross-position
data propagation (the shift digit must reach every letter position).

Uses a single 260-entry lookup table with 16D concatenated keys
(letter_embedding ⊕ shift_embedding) rather than 10 separate tables
with switch dispatch.
"""

from typing import Tuple

import torch

from torchwright.graph import Node, Embedding, Concatenate
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import concat
from torchwright.ops.inout_nodes import (
    create_constant,
    create_embedding,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import equals_vector
from torchwright.ops.map_select import map_to_table
from torchwright.ops.sequence_ops import output_sequence

# Letters in the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"

D_MODEL = 512
MAX_LETTERS = 5


def create_network_parts(
    max_letters: int = MAX_LETTERS,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the Caesar cipher computation graph.

    Args:
        max_letters: Fixed number of letter positions in the input.
            Shorter inputs should be space-padded on the right.
    """
    vocab = list(LETTERS) + list("0123456789") + [" ", "\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()
    embed = embedding.get_embedding

    is_trigger = equals_vector(embedding, embed("\n"))

    # --- Detect and propagate the shift digit ---
    # The shift digit is at position 1 (after <bos>). Detect it with a
    # digit lookup, then latch it to all subsequent positions.
    is_digit = map_to_table(
        inp=embedding,
        key_to_value={embed(str(i)): torch.tensor([1.0]) for i in range(10)},
        default=torch.tensor([-1.0]),
    )
    latched_shift = pos_encoding.get_prev_value(embedding, is_digit)

    # --- Per-position shifted letter via combined lookup ---
    # Concatenate current letter embedding (8D) + latched shift embedding (8D)
    # into a 16D key, then look up the shifted letter in a single 260-entry table.
    combined = concat([embedding, latched_shift])

    shift_table = {}
    for s in range(10):
        for i, letter in enumerate(LETTERS):
            shifted = LETTERS[(i + s) % 26]
            key = torch.cat([embed(letter), embed(str(s))])
            shift_table[key] = embed(shifted)

    shifted_letter = map_to_table(combined, shift_table, default=embed(" "))

    # --- Collect shifted letters at trigger ---
    # At the \n position, read each letter position via backward attention,
    # then latch the values for autoregressive output.
    # Input format: <bos> <digit> <L1> <L2> ... <Ln> \n
    # Letter positions are at offsets -(max_letters) through -1 from \n.
    output_letters = []
    for i in range(max_letters):
        offset = -(max_letters - i)  # -5, -4, -3, -2, -1 for max_letters=5
        raw = pos_encoding.attend_to_offset(shifted_letter, delta_pos=offset)
        latched = pos_encoding.get_prev_value(raw, is_trigger)
        output_letters.append(latched)

    output_letters.append(create_constant(embed("<eos>")))

    output_node = output_sequence(
        pos_encoding,
        is_trigger,
        output_letters,
        embed(" "),
    )
    return output_node, pos_encoding, embedding


def create_network(max_letters: int = MAX_LETTERS) -> Unembedding:
    """Create a Caesar cipher network."""
    output_node, pos_encoding, embedding = create_network_parts(max_letters)
    return create_unembedding(output_node, embedding)
