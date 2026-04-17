"""Attention-based digit sort (V2): pure MLP selection, emit via scalar_to_embedding.

Input: a string of single-digit integers terminated by ``"\\n"``.
Output: the digits sorted ascending. Handles duplicates stably.

    Input:  <bos> 1 1 2 1 \\n
    Output: 1 1 1 2

**What V2 is meant to show.** V2 exists as a *foil* for V1 and V4.
Where V1 and V4 use content-based attention to discover the next item,
V2 does the selection entirely in MLP ops and only uses attention for
counting (the input-phase ``prefix_sum``). The output phase emits
digits via ``scalar_to_embedding`` — a pure MLP conversion from a
scalar 0..9 to the digit's embedding, with no attention at all.

This is the bottom of the "how much does attention do?" spectrum:
attention counts, MLP decides, MLP emits. Comparing V2 to V4 makes
explicit which parts of the sort each variant assigns to which
substrate.

Algorithm.

1. **Input phase.** For each digit ``d ∈ {0..9}``, count the total
   number of input-digit positions with that value using one parallel
   ``prefix_sum`` each. Latch the totals at the trigger via
   ``get_prev_value``.
2. **Unrolled emission loop.** Maintain an in-Python dict
   ``count_so_far[d]`` tracking how many of each digit have been
   "emitted" across prior unroll iterations (one Node per d, all
   literal-zero at iteration 0). At each iteration ``k``:
   - **Find d_next** = smallest ``d`` with ``total_count[d] >
     count_so_far[d]``: a compile-time reversed scan of
     ``select(has_remaining, literal(d), d_next_so_far)``.
   - **Emit** ``seq[k] = scalar_to_embedding(d_next, embedding)`` —
     the MLP that reconstructs the digit embedding from a scalar.
   - **Advance** ``count_so_far[d_next] += 1`` via a per-d
     ``select``/``bool_all_true`` check.
3. **Slot gating** via ``_emit_by_slot_index`` (same helper as V4,
   avoiding the ``attend_to_offset`` aliasing bug in
   ``output_sequence``).

Everything at pre-trigger positions is junk because the latched total
counts are undefined there, but ``_emit_by_slot_index`` gates the
output on ``has_triggered`` so only post-trigger values are consumed.
"""

from typing import Dict, List, Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_scaled_nodes,
    compare,
    negate,
    sum_nodes,
)
from torchwright.ops.inout_nodes import (
    create_embedding,
    create_literal_value,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import (
    bool_all_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.map_select import select
from torchwright.ops.prefix_ops import prefix_sum
from torchwright.ops.scalar_encoding import digit_to_scaled_scalar, scalar_to_embedding
from torchwright.ops.sequence_ops import check_is_digit

D_MODEL = 1024
MAX_OUT = 5  # Kept small: each unrolled iteration is ~20 MLP layers deep.
N_STAGES = 5


def _emit_by_slot_index(
    pos_encoding: PosEncoding,
    is_trigger: Node,
    seq: list,
    default_output: torch.Tensor,
) -> Node:
    """See ``sort_digits_v4.py`` for the full rationale."""
    max_out = len(seq)
    has_triggered = pos_encoding.get_prev_value(is_trigger, is_trigger)
    pos_scalar = pos_encoding.get_position_scalar()
    trigger_pos_scalar = pos_encoding.get_prev_value(pos_scalar, is_trigger)
    steps_since = add_scaled_nodes(1.0, pos_scalar, -1.0, trigger_pos_scalar)

    gated = []
    for k in range(max_out):
        cond_k = bool_all_true(
            [
                compare(steps_since, k - 0.5),
                compare(negate(steps_since), -(k + 0.5)),
            ]
        )
        gated.append(cond_gate(cond_k, seq[k]))

    return select(
        cond=has_triggered,
        true_node=sum_nodes(gated),
        false_node=create_literal_value(default_output),
    )


def create_network_parts(
    max_out: int = MAX_OUT,
    n_stages: int = N_STAGES,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the V2 MLP-brain sort graph."""
    vocab = list("0123456789") + [" ", "\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()
    embed = embedding.get_embedding

    is_trigger = equals_vector(embedding, embed("\n"))

    # --- Per-position input features ---
    digit_scalar = digit_to_scaled_scalar(embedding, embedding, place_value=1.0)
    is_digit_pos = check_is_digit(embedding)
    has_triggered = pos_encoding.get_prev_value(is_trigger, is_trigger)
    is_pre_trigger = bool_not(has_triggered)
    is_input_digit = bool_all_true([is_digit_pos, is_pre_trigger])

    # --- Per-digit total counts (latched at trigger) ---
    latched_counts: Dict[int, Node] = {}
    for d in range(10):
        is_this_digit = bool_all_true(
            [
                is_input_digit,
                compare(digit_scalar, d - 0.5),
                compare(negate(digit_scalar), -(d + 0.5)),
            ]
        )
        is_this_digit_01 = select(
            is_this_digit,
            create_literal_value(torch.tensor([1.0])),
            create_literal_value(torch.tensor([0.0])),
        )
        running = prefix_sum(pos_encoding, is_this_digit_01, n_stages)
        latched_counts[d] = pos_encoding.get_prev_value(running, is_trigger)

    total_count = sum_nodes(list(latched_counts.values()))

    # --- Unrolled emission loop ---
    count_so_far: Dict[int, Node] = {
        d: create_literal_value(torch.tensor([0.0])) for d in range(10)
    }

    seq: List[Node] = []
    for _ in range(max_out):
        # Find d_next = smallest d with remaining > 0, scanning in
        # reverse so the first (smallest d) wins.
        d_next: Node = create_literal_value(torch.tensor([10.0]))  # sentinel
        for d in reversed(range(10)):
            remaining_d = add_scaled_nodes(
                1.0,
                latched_counts[d],
                -1.0,
                count_so_far[d],
            )
            has_remaining = compare(remaining_d, 0.5)
            d_next = select(
                has_remaining,
                create_literal_value(torch.tensor([float(d)])),
                d_next,
            )

        # Emit the embedding for d_next. For slots beyond the input
        # count this gives a garbage embedding (d_next = 10, sentinel);
        # the select below patches those to " ".
        digit_embed = scalar_to_embedding(d_next, embedding)
        in_range = compare(total_count, float(len(seq)) + 0.5)
        space_literal = create_literal_value(embed(" "))
        seq.append(select(in_range, digit_embed, space_literal))

        # Advance count_so_far: add 1 to the emitted digit's counter.
        for d in range(10):
            is_d = bool_all_true(
                [
                    compare(d_next, d - 0.5),
                    compare(negate(d_next), -(d + 0.5)),
                ]
            )
            count_so_far[d] = select(
                is_d,
                sum_nodes([count_so_far[d], create_literal_value(torch.tensor([1.0]))]),
                count_so_far[d],
            )

    output_node = _emit_by_slot_index(
        pos_encoding,
        is_trigger,
        seq,
        embed(" "),
    )
    return output_node, pos_encoding, embedding


def create_network() -> Unembedding:
    output_node, pos_encoding, embedding = create_network_parts()
    return create_unembedding(output_node, embedding)
