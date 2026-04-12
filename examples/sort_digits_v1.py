"""Attention-based digit sort (V1): indicator-basis next-above-threshold.

Input: string of single-digit integers terminated by ``"\\n"``.
Output: the digits sorted ascending, one per autoregressive step,
starting at the ``"\\n"`` trigger.

    Input:  <bos> 9 5 8 3 \\n
    Output: 3 5 8 9

**Does not handle duplicates.** This variant uses the digit value alone
as its ordering key. With ``attend_argmin_above_integer`` the softmax
weighted-averages equal-score positions, collapsing duplicates into a
single emission — so ``"1121"`` decodes to ``"12"``, not ``"1112"``. For
full-generality sorts with stable duplicate ordering, see V2, V3, V4.

**Why ship V1.** Of the four variants it is the most literal expression
of "at each step, one attention head directly discovers the next item
in the sort order." The search the attention performs is unambiguous —
"give me the smallest digit strictly greater than the last emitted
digit" — and it happens inside one vanilla attention head per output
slot, driven by an **indicator basis** on the key side. No mask vector
is carried forward across output positions; the only state is a single
latched ``prev_digit`` scalar.

How the "next above threshold" attention works in one vanilla head.
The key obstacle: ``score_i > prev_digit_j`` is a step function mixing
per-key ``score_i`` and per-query ``prev_digit_j``, which a plain
bilinear ``Q·K^T`` cannot compute. We solve this with an **indicator
basis** — at each key position, precompute the width-10 vector
``indicators_above[c] = I(digit_i > (c - 1)) AND is_input_digit_i`` for
``c ∈ {0, 1, …, 9}`` (one slot per possible threshold
``d ∈ {-1, 0, …, 8}``). The query side then provides a width-10
one-hot ``threshold_onehot`` selecting which threshold applies at this
query position. Inside the attention, the bilinear product over those
10 dimensions exactly computes ``I(score_i > prev_digit_j)``, while
column 0 carries ``-score + tiebreak·pos`` — and the primitive
``attend_argmin_above_integer`` bakes all of this into one head.

Per-slot unrolling. At iteration ``k`` the ``prev_digit`` is the digit
value emitted at iteration ``k - 1``, latched at the trigger via
``get_prev_value``. Iteration 0 uses ``prev_digit = -1`` so every input
digit is strictly greater than it (unconstrained argmin). We chain
these through ``MAX_OUT`` unrolled iterations; the output sequence is
gated by the same scalar-arithmetic slot gating used by V4 (see
``_emit_by_slot_index`` there for rationale).

Residual footprint. 10 key-side indicator columns + 1 score column +
1 position scalar + value passthrough. No per-input mask. V1 is the
narrowest of the four variants (D_MODEL = 384).
"""

from typing import List, Tuple

import torch

from torchwright.graph import Concatenate, Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import (
    add_scaled_nodes,
    compare,
    negate,
    sum_nodes,
)
from torchwright.ops.attention_ops import attend_argmin_above_integer
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
from torchwright.ops.map_select import in_range, select
from torchwright.ops.scalar_encoding import digit_to_scaled_scalar
from torchwright.ops.sequence_ops import check_is_digit


D_MODEL = 384
MAX_OUT = 10
# The 10 thresholds our indicator basis covers: d ∈ {-1, 0, 1, …, 8}.
# Slot c in the indicator / one-hot corresponds to threshold d = c - 1.
_N_SLOTS = 10


def _emit_by_slot_index(
    pos_encoding: PosEncoding,
    is_trigger: Node,
    seq: list,
    default_output: torch.Tensor,
) -> Node:
    """Same helper as in ``sort_digits_v4.py``; see the docstring there
    for the full rationale. Short version: ``output_sequence``'s slot
    gating uses ``attend_to_offset(is_trigger, delta_pos=-k)`` which
    aliases when ``k`` approaches the period of the fastest sine
    component (≈ 6). We replace it with an exact integer comparison on
    ``pos_scalar - trigger_pos_scalar``.
    """
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


def _build_indicators_above(digit_scalar: Node, is_input_digit: Node) -> Node:
    """Build the width-10 indicator basis for an input-digit key side.

    Slot ``c`` is ``1.0`` iff the current position is an input-digit
    position whose digit scalar is strictly greater than ``c - 1``.
    Non-input-digit positions get all-zero indicators (they will
    therefore never win the attention's argmin-above search).
    """
    # is_input_digit is {+1, -1}; we need {0, 1} to multiply through
    # the per-threshold step.
    is_input_digit_01 = select(
        is_input_digit,
        create_literal_value(torch.tensor([1.0])),
        create_literal_value(torch.tensor([0.0])),
    )
    cols = []
    for c in range(_N_SLOTS):
        d = c - 1  # threshold value
        # ``compare(digit, d + 0.5, true_level=1.0, false_level=0.0)``
        # gives 1 iff digit > d for integer digits. Multiply (via
        # ``select``, since the graph has no elementwise mul) by the
        # input-digit mask to zero out non-input positions.
        step = compare(digit_scalar, d + 0.5, true_level=1.0, false_level=0.0)
        col = select(
            is_input_digit,
            step,
            create_literal_value(torch.tensor([0.0])),
        )
        cols.append(col)
    return Concatenate(cols)


def _threshold_onehot(prev_digit: Node) -> Node:
    """Convert a scalar ``prev_digit ∈ {-1, 0, …, 8}`` to a width-10
    ``{0, 1}`` one-hot selecting the matching threshold slot.

    Slot ``c = prev_digit + 1``: slot 0 means "prev=-1" (initial),
    slot 1 means "prev=0", etc.
    """
    shifted = add_scaled_nodes(
        1.0, prev_digit, 1.0, create_literal_value(torch.tensor([1.0]))
    )
    shifted_plus_one = add_scaled_nodes(
        1.0, shifted, 1.0, create_literal_value(torch.tensor([1.0]))
    )
    onehot_bool = in_range(shifted, shifted_plus_one, _N_SLOTS)  # ±1
    return add_scaled_nodes(
        0.5,
        onehot_bool,
        0.5,
        create_literal_value(torch.ones(_N_SLOTS)),
    )


def create_network_parts(
    max_out: int = MAX_OUT,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the V1 distinct-digit selection-sort graph."""
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

    # --- Shared key-side indicator basis (computed once per position) ---
    indicators_above = _build_indicators_above(digit_scalar, is_input_digit)

    # --- Unrolled selection sort ---
    # The iteration-k threshold is the digit emitted at iteration k-1,
    # latched at the trigger. Iteration 0 starts with prev = -1.
    #
    # Each iteration runs *two* attention heads sharing the same Q/K
    # machinery but with different ``value`` projections:
    #   - ``selected_embed`` (value=embedding) is the digit embedding
    #     that gets placed in ``seq[k]`` for autoregressive emission.
    #   - ``selected_digit_scalar`` (value=digit_scalar) is the digit
    #     value as a raw scalar, threaded through ``get_prev_value``
    #     into the next iteration's threshold_onehot.
    #
    # Why two heads instead of extracting the scalar from the embedding
    # with ``digit_to_scaled_scalar``: the softmax-averaged embedding at
    # the winner position is close to the winner's digit embedding but
    # not *exactly* equal (small leakage from other softmax terms).
    # ``digit_to_scaled_scalar`` routes through ``map_to_table``, whose
    # ``embedding_step_sharpness`` threshold is tight — a tiny dot
    # product loss of ~0.5 translates to a scalar output of ~0.5 instead
    # of the intended integer. That error compounds through the chain of
    # latched prev_digits and the sort drifts. Reading the scalar
    # directly as the attention ``value`` gives a softmax-weighted sum
    # of per-position ``digit_scalar`` values, which stays within 1e-3
    # of the integer answer even under softmax leakage.
    prev_digit: Node = create_literal_value(torch.tensor([-1.0]))

    seq: List[Node] = []
    for _ in range(max_out):
        threshold_onehot = _threshold_onehot(prev_digit)
        selected_embed = attend_argmin_above_integer(
            pos_encoding=pos_encoding,
            score=digit_scalar,
            indicators_above=indicators_above,
            threshold_onehot=threshold_onehot,
            value=embedding,
        )
        seq.append(selected_embed)

        # Same argmin-above attention but with value=digit_scalar so
        # the scalar threads cleanly through the chain.
        selected_digit_scalar = attend_argmin_above_integer(
            pos_encoding=pos_encoding,
            score=digit_scalar,
            indicators_above=indicators_above,
            threshold_onehot=threshold_onehot,
            value=digit_scalar,
        )
        prev_digit = pos_encoding.get_prev_value(selected_digit_scalar, is_trigger)

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
