"""Attention-based digit sort (V4): masked argmin via unrolled state.

Inputs: a string of single-digit integers terminated by ``"\\n"``.
Output: the digits sorted in ascending order, one per autoregressive
step, starting at the newline trigger.

    Input:  <bos> 9 5 8 3 \\n
    Output: 3 5 8 9

This is the **primary** sort variant in the attention-sort study. Every
output step is one genuine masked-argmin attention: at step k, the
``attend_argmin_unmasked`` primitive runs over all input positions,
picks the one with the smallest ``(digit, position)`` lex score whose
slot isn't set in a running mask, and copies its digit embedding into
the output. After each step the mask is OR'd with the one-hot of the
position just selected, so subsequent steps skip it.

How the recurrence is expressed (without delay-1 self-references).
torchwright's compiler does **not** support node self-references
(``mask_node = f(..., attend_to_offset(mask_node, -1), ...)``), so we
unroll the recurrence into ``MAX_OUT`` distinct Nodes, matching the
pattern ``torchwright.ops.prefix_ops.prefix_sum`` already uses.
``mask_0`` is a literal zero vector; every subsequent ``mask_{k+1}`` is
defined as ``elementwise_max(mask_k, selection_onehot_k)``, so the DAG
only ever references *prior* iterations, never itself. Each iteration
adds one attention head for ``selection_onehot_k`` and another for
``digit_embed_k``, plus the elementwise max.

Why duplicates work. The attention's ordering key is
``score_i = 10 * digit_i + digit_index_i``, where ``digit_index_i`` is
the input position's index among the digit positions only (0-indexed,
computed via ``prefix_sum`` on an ``is_digit`` indicator). This gives
every input digit a unique score, with duplicates broken by input
order — a stable sort.

Score envelope. Valid digit scores land in ``[0, 99]`` (digit ``0..9``
times 10, plus digit_index ``0..9``). Non-digit positions are pushed to
the sentinel value ``_NON_DIGIT_SCORE`` via ``select``, which is above
every valid score so the attention always skips them. The sentinel is
kept under ``_MAX_SCORE_UNMASKED_ABS`` from
``torchwright.ops.attention_ops`` so the attention logit stays above
the causal-mask floor.

Mask width. The mask and the position one-hot are both width
``MAX_INPUT`` — not the total sequence length. Using the digit-index as
the slot key keeps the mask narrow (10 slots) regardless of how much
``<bos>`` / ``\\n`` / output padding the sequence carries.
"""

from typing import List, Tuple

import torch

from torchwright.graph import Node, Embedding
from torchwright.graph.embedding import Unembedding
from torchwright.graph.pos_encoding import PosEncoding
from torchwright.ops.arithmetic_ops import add_scaled_nodes, compare, negate, sum_nodes
from torchwright.ops.arithmetic_ops import max as elementwise_max
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import (
    create_embedding,
    create_literal_value,
    create_pos_encoding,
    create_unembedding,
)
from torchwright.ops.logic_ops import bool_all_true, bool_not, cond_gate, equals_vector
from torchwright.ops.map_select import in_range, select
from torchwright.ops.prefix_ops import prefix_sum
from torchwright.ops.scalar_encoding import digit_to_scaled_scalar
from torchwright.ops.sequence_ops import check_is_digit


D_MODEL = 512
MAX_INPUT = 10
MAX_OUT = 10
N_STAGES = 5  # 2**5 = 32 ≥ 1 (bos) + MAX_INPUT + 1 (\n) + MAX_OUT
_NON_DIGIT_SCORE = 100.0  # Just above the max valid digit score (99).


def _emit_by_slot_index(
    pos_encoding: PosEncoding,
    is_trigger: Node,
    seq: list,
    default_output: torch.Tensor,
) -> Node:
    """Autoregressive emission gated by a scalar step counter.

    This is a drop-in replacement for ``output_sequence`` that avoids
    ``attend_to_offset`` for the slot-index gating. ``output_sequence``
    uses ``attend_to_offset(is_trigger, delta_pos=-k)`` to test whether
    we are exactly ``k`` steps past the trigger. That call is vulnerable
    to sinusoidal aliasing in the positional encoding: for any ``k``
    close to the period of the fastest sine component (≈ 6.28), the
    attention picks the trigger position itself as the closest match
    and incorrectly returns ``+1``, firing the wrong slot. This matters
    for V4 because ``len(seq) = MAX_OUT`` can exceed that period.

    Instead we compute the step index arithmetically:
    ``steps_since = pos_scalar - trigger_pos_scalar`` where
    ``trigger_pos_scalar`` is latched at the trigger via
    ``get_prev_value``. ``compare`` then tests ``steps_since == k`` for
    each compile-time k, and the gated values are summed and selected
    against ``default_output`` using ``has_triggered``.

    The compile cost is higher than ``output_sequence`` (one ``compare``
    pair per slot instead of one ``attend_to_offset``) but the slot
    gating is exact and immune to aliasing.
    """
    max_out = len(seq)
    has_triggered = pos_encoding.get_prev_value(is_trigger, is_trigger)
    pos_scalar = pos_encoding.get_position_scalar()
    # Latch the trigger's position scalar at the trigger itself, held
    # forward for all subsequent positions via ``get_prev_value``.
    trigger_pos_scalar = pos_encoding.get_prev_value(pos_scalar, is_trigger)
    # Integer number of positions since the trigger fired (0 at the
    # trigger itself, 1 at P+1, etc.). At positions before the trigger
    # this is garbage, but the ``has_triggered`` select below discards
    # the entire sum in that case.
    steps_since = add_scaled_nodes(
        1.0, pos_scalar, -1.0, trigger_pos_scalar
    )

    gated = []
    for k in range(max_out):
        # cond_k: true iff steps_since == k, tested with a ±0.5-wide
        # band around k so it fires on integer matches.
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
    max_input: int = MAX_INPUT,
    max_out: int = MAX_OUT,
    n_stages: int = N_STAGES,
) -> Tuple[Node, PosEncoding, Embedding]:
    """Build the V4 masked-argmin sort graph.

    Args:
        max_input: Maximum number of input digits (also the mask width).
        max_out: Maximum number of output slots to unroll.
        n_stages: ``prefix_sum`` stages (must cover ``n_pos ≤ 2**n_stages``).
    """
    vocab = list("0123456789") + [" ", "\n", "<bos>", "<eos>"]
    embedding = create_embedding(vocab=vocab)
    pos_encoding = create_pos_encoding()
    embed = embedding.get_embedding

    is_trigger = equals_vector(embedding, embed("\n"))

    # --- Per-position input features ---
    # digit_scalar: 0..9 at digit positions, 0 elsewhere (map_to_table fallback).
    digit_scalar = digit_to_scaled_scalar(embedding, embedding, place_value=1.0)
    # is_digit_position: {+1, -1} boolean. Fires for both input digit
    # positions AND emitted output digits, so on its own it can't be
    # used to count "how many *input* digits have we seen" — the output
    # phase re-emits digit tokens which would otherwise pollute the
    # running count.
    is_digit_pos = check_is_digit(embedding)

    # has_triggered: {+1, -1} boolean, true from the trigger onwards.
    # ``get_prev_value(is_trigger, is_trigger)`` is the standard
    # torchwright idiom for "has the trigger fired at or before me".
    has_triggered = pos_encoding.get_prev_value(is_trigger, is_trigger)
    is_pre_trigger = bool_not(has_triggered)
    # is_input_digit: only fires at digit positions strictly before the
    # trigger. Emitted output digits return False here and therefore
    # don't advance the count or get treated as input.
    is_input_digit = bool_all_true([is_digit_pos, is_pre_trigger])

    # digit_index = running count of *input* digits so far, minus 1
    # (0-indexed at digit positions). Needs a {0, 1} indicator for the
    # prefix sum.
    is_input_digit_01 = select(
        is_input_digit,
        create_literal_value(torch.tensor([1.0])),
        create_literal_value(torch.tensor([0.0])),
    )
    running_input_digit_count = prefix_sum(
        pos_encoding, is_input_digit_01, n_stages
    )
    digit_index = add_scaled_nodes(
        1.0,
        running_input_digit_count,
        -1.0,
        create_literal_value(torch.tensor([1.0])),
    )
    # At non-input-digit positions the digit_index carries the most
    # recent input digit's index (or -1 before the first digit). We
    # don't care about those positions' one-hots or scores — they are
    # excluded by ``_NON_DIGIT_SCORE`` before ever being considered by
    # the attention.

    # position_onehot: width MAX_INPUT, 1 exactly at slot == digit_index.
    # ``in_range`` returns ±1 per slot; convert to {0, 1}.
    digit_index_plus_one = add_scaled_nodes(
        1.0,
        digit_index,
        1.0,
        create_literal_value(torch.tensor([1.0])),
    )
    position_onehot_bool = in_range(digit_index, digit_index_plus_one, max_input)
    position_onehot = add_scaled_nodes(
        0.5,
        position_onehot_bool,
        0.5,
        create_literal_value(torch.ones(max_input)),
    )

    # score: 10 * digit + digit_index at *input* digit positions,
    # sentinel elsewhere. Valid range [0, 99]; sentinel 100 guarantees
    # every other position (including emitted output digits after the
    # trigger) is beaten by any real input-digit position in the argmin.
    score_if_digit = add_scaled_nodes(10.0, digit_scalar, 1.0, digit_index)
    score = select(
        is_input_digit,
        score_if_digit,
        create_literal_value(torch.tensor([_NON_DIGIT_SCORE])),
    )

    # --- Unrolled selection sort ---
    # mask_k is a width-MAX_INPUT {0, 1} vector whose value at every
    # position is the set of digit-index slots already emitted before
    # step k. mask_0 is the zero vector; each subsequent mask_k folds in
    # the one-hot returned by the previous iteration's attention. No DAG
    # cycles: each mask_k only references mask_{k-1} and
    # selection_onehot_{k-1}, both defined before it.
    mask_k: Node = create_literal_value(torch.zeros(max_input))
    seq: List[Node] = []
    for _ in range(max_out):
        selection_onehot = attend_argmin_unmasked(
            pos_encoding=pos_encoding,
            score=score,
            mask_vector=mask_k,
            position_onehot=position_onehot,
            value=position_onehot,
        )
        selected_digit_embed = attend_argmin_unmasked(
            pos_encoding=pos_encoding,
            score=score,
            mask_vector=mask_k,
            position_onehot=position_onehot,
            value=embedding,
        )
        seq.append(selected_digit_embed)
        # Elementwise max of two {0, 1} vectors is the bitwise OR.
        mask_k = elementwise_max(mask_k, selection_onehot)

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
