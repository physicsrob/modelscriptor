"""Content-based attention primitives built on top of ``graph.Attn``.

Each primitive compiles to a single vanilla attention head. The trick is all
in the Q/K/V weight matrices: by carefully choosing what the attention logit
computes, one head can search for "the position with the smallest score",
"the position with the smallest score among valid positions", or "the
position with the smallest score whose input-index isn't already in a
running mask".

All primitives here follow this template:

- ``query_in`` is the positional encoding (and, for the ``_above_integer``
  / ``_unmasked`` / ``_dot`` variants, also the per-query signal that
  rendezvous with a key-side vector). ``query_matrix`` populates row
  ``d_pos - 1`` — the slowest *cosine* component ``pos_enc[j, d_pos-1]``,
  which is ≈ 1 for any realistic ``j`` — with ``_QUERY_GAIN`` so that
  column 0 of the logit gets a stable positive per-query gain.
- ``key_in`` contains **only** what ``key_matrix`` reads from: the
  positional encoding (for the tiebreak — see below), plus the content
  nodes that drive selection (score, validity, indicators, onehot, etc.).
  Never concat a node into ``key_in`` without wiring up at least one
  non-zero ``key_matrix`` row for it; ``Attn.__init__`` enforces this.
- The **tiebreak** lives on row ``d_pos - 2`` of ``key_matrix`` — the
  slowest *sine* component ``sin(j · freq)`` where
  ``freq = pos_encoding.slow_sin_freq()``. For sort lengths well inside
  the first quarter-period, this is monotonically increasing in ``j``
  and approximately equal to ``j · freq``, giving us a linear-in-position
  tiebreak without materialising a separate ``position_scalar`` subtree.
  Coefficients are scaled by ``1 / freq`` so the effective per-position
  strength matches ``_TIEBREAK_COEFF``.
- ``value_in`` is whatever node we want to read at the selected key
  position; ``value_matrix`` and ``output_matrix`` are identity projections
  that copy it through unchanged.

Query gain.  With the causal-mask sentinel at ``-1e6`` in
``Attn.compute``, ``_QUERY_GAIN = 80`` keeps the worst valid logit at
``80 × 120 = 9600``, far above ``-1e6``.  A unit score delta produces a
logit delta of 80 → ``exp(80) ≈ 5.5e34`` softmax weight ratio —
effectively hard selection for any non-degenerate gap.  Exact ties are
still weighted-averaged, matching ``get_prev_value``'s behaviour.

Step-function logits (e.g. strict ``>`` comparisons against a runtime
threshold) are not expressible in bilinear Q·K. The ``_where`` and
``_unmasked`` variants take a pre-computed validity / mask signal as
input rather than synthesising the step function inside the attention op.
"""

import torch

from torchwright.graph import Node, Concatenate, Attn
from torchwright.graph.pos_encoding import PosEncoding


# Coefficient applied to the slowest-cosine component of the positional
# encoding inside the query projection. Chosen so that
# Coefficient applied to the slowest-cosine component of the positional
# encoding inside the query projection. Chosen so that
# ``Q[j, 0] ≈ _QUERY_GAIN`` for every position ``j`` in a realistic
# sequence length (the slowest cosine is ``cos(j · d[-1]) ≈ 1`` for
# ``j`` up to a few thousand). With the causal-mask sentinel at -1e6
# in ``Attn.compute``, gains up to ~8000 are safe for |score| ≤ 120.
# We use 80 for a comfortable 10× margin: a unit score delta produces
# a logit delta of 80, i.e. ``exp(80) ≈ 5.5e34`` softmax weight
# ratio — effectively hard selection for any non-degenerate score gap.
_QUERY_GAIN = 80.0

# Coefficient on the position-scalar tiebreak in key-space. Must satisfy
# ``_TIEBREAK_COEFF * n_pos < 1`` so that a unit score difference
# always dominates the tiebreak. For ``n_pos <= 100`` this gives headroom
# of one order of magnitude; bump it down if you sort longer sequences
# with unit-integer scores.
_TIEBREAK_COEFF = 0.001

# "Infinity-substitute" magnitude for the validity penalty in the
# ``_where`` variants. Must exceed the maximum possible ``|score|`` so
# validity always dominates score. With the causal-mask sentinel at
# -1e6, the old ceiling of 125 no longer applies; 1000 gives a
# comfortable margin over any realistic score.
_VALIDITY_LARGE = 1000.0

# Maximum ``|score|`` supported by these primitives. With sentinel at
# -1e6 and ``_QUERY_GAIN = 80``, the worst logit at |score| = 120 is
# ``80 × (-120) = -9600``, far above -1e6. The old constraint
# (``gain × score < 1000``) is gone; this ceiling is now just a
# documentation hint for callers about the tested range.
_MAX_SCORE_ABS = 120.0

# Penalty (in *logit* space, not key space) applied by
# ``attend_argmin_unmasked`` to masked positions. Must exceed
# ``_QUERY_GAIN * _MAX_SCORE_UNMASKED_ABS`` so a masked position with
# the best score still loses to an unmasked position with the worst
# score. With gain=80 and max_score=100: 80×100 = 8000, so 10000
# gives a comfortable margin.
_UNMASKED_PENALTY = 10000.0

# Maximum ``|score|`` supported by ``attend_argmin_unmasked``.
_MAX_SCORE_UNMASKED_ABS = 100.0

# Bonus applied to "above threshold" positions in
# ``attend_argmin_above_integer``. Added directly to the logit (not
# scaled by _QUERY_GAIN, because the bonus goes straight into the
# query_matrix entries rather than through the slow-cosine multiplier).
# Must exceed ``_QUERY_GAIN · (max_score - min_score)`` so a valid
# position with the worst score still beats any invalid position with
# the best score. For ``score ∈ [0, 9]`` that gives ``_QUERY_GAIN · 9
# = 720``; ``_ABOVE_BONUS = 1000`` buys a comfortable margin.
_ABOVE_BONUS = 1000.0


def _assert_value_fits(pos_encoding: PosEncoding, value: Node) -> int:
    """Shared precondition: value must fit inside the attention head width.

    Returns the chosen ``d_head`` (= ``pos_encoding.d_pos``).
    """
    d_head = pos_encoding.d_pos
    assert len(value) <= d_head, (
        f"attend_*: value width ({len(value)}) must be <= d_pos ({d_head}). "
        "If you need a wider value, project it down first or split it "
        "across multiple attention primitives."
    )
    return d_head


def _build_selection_attn(
    pos_encoding: PosEncoding,
    key_in: Node,
    key_matrix: torch.Tensor,
    value: Node,
) -> Attn:
    """Wire up a selection-style attention head.

    Callers supply ``key_in`` (a Concatenate of pos_encoding and content
    nodes) and a populated ``key_matrix``; this helper fills in the
    query / value / output matrices shared across all primitives below.
    """
    d_head = _assert_value_fits(pos_encoding, value)

    # query_in = pos_encoding. We need ``Q[j, 0]`` to be a stable positive
    # constant (independent of query position ``j``) so the softmax
    # decisiveness doesn't vary with where we are in the sequence. The
    # slowest cosine component ``pos_enc[j, d_pos - 1]`` is
    # ``cos(j · d[-1])`` which equals ``~1`` for ``j`` up to a few
    # thousand — nearly constant over any realistic sort length. Scaling
    # it by ``_QUERY_GAIN`` gives a per-position query coefficient close
    # to 8, which is both large enough that a unit score delta is
    # decisive (``exp(8) ≈ 3000``) and small enough that our
    # ``|score| ≤ _MAX_SCORE_ABS`` envelope keeps every valid logit above
    # the ``CAUSAL_MASK_SENTINEL`` in ``Attn.compute``. Other
    # columns of Q don't matter because ``K`` has only column 0
    # populated; we zero them out for clarity.
    query_matrix = torch.zeros((len(pos_encoding), d_head))
    query_matrix[-1, 0] = _QUERY_GAIN

    # Identity pass-through for value. value_matrix embeds value into the
    # first len(value) columns of d_head; output_matrix reads them back.
    value_matrix = torch.eye(len(value), d_head)
    output_matrix = torch.eye(d_head, len(value))

    return Attn(
        query_in=pos_encoding,
        key_in=key_in,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )


def attend_argmin(pos_encoding: PosEncoding, score: Node, value: Node) -> Node:
    """Attend to the position with the *minimum* score.

    For each query position, this returns ``value`` at the position within
    the causal window (positions ``<= current``) whose ``score`` is
    smallest. Ties break toward the most recent (latest) position — the
    same convention ``get_prev_value`` uses.

    To mask positions you want the attention to ignore, pass a score that
    is very large at those positions (a few hundred is enough; see
    ``_VALIDITY_LARGE`` for the scale). For a cleaner valid/invalid API,
    use :func:`attend_argmin_where` instead.

    Compile cost: exactly one vanilla attention head.

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node (``len(score) == 1``).
        value: Node whose value to read at the winning position.
            Must satisfy ``len(value) <= pos_encoding.d_pos``.

    Returns:
        A new ``Attn`` node of width ``len(value)`` equal to ``value`` at
        the argmin-of-``score`` key position within the causal window.

    See also:
        :func:`attend_argmax`, :func:`attend_argmin_where`.
    """
    assert len(score) == 1, "attend_argmin expects a 1D scalar score node"
    d_head = _assert_value_fits(pos_encoding, value)

    # key_in = [pos_encoding (d_pos), score (1)]
    # Slow-sin row (d_pos - 2) carries the linear-in-position tiebreak.
    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    key_in = Concatenate([pos_encoding, score])
    key_matrix = torch.zeros((len(key_in), d_head))
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq  # latest-position tiebreak
    key_matrix[d_pos, 0] = -1.0                         # smaller score → larger logit

    return _build_selection_attn(pos_encoding, key_in, key_matrix, value)


def attend_argmax(pos_encoding: PosEncoding, score: Node, value: Node) -> Node:
    """Attend to the position with the *maximum* score.

    Sign-flipped twin of :func:`attend_argmin`. Ties break toward the
    latest position.

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node.
        value: Node whose value to read. ``len(value) <= pos_encoding.d_pos``.

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        argmax-of-``score`` key position within the causal window.
    """
    assert len(score) == 1, "attend_argmax expects a 1D scalar score node"
    d_head = _assert_value_fits(pos_encoding, value)

    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    key_in = Concatenate([pos_encoding, score])
    key_matrix = torch.zeros((len(key_in), d_head))
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq  # latest-position tiebreak
    key_matrix[d_pos, 0] = 1.0                          # larger score → larger logit

    return _build_selection_attn(pos_encoding, key_in, key_matrix, value)


def attend_argmin_where(
    pos_encoding: PosEncoding,
    score: Node,
    validity: Node,
    value: Node,
) -> Node:
    """Argmin of ``score`` restricted to positions where ``validity`` is true.

    The workhorse primitive for selection-sort variants. At each query
    position, the attention returns ``value`` at the causal-window
    position where ``validity`` is true **and** ``score`` is smallest.

    ``validity`` follows the usual torchwright boolean convention: +1.0
    means "valid", −1.0 means "invalid". The logit at key position ``i``
    is

        −score[i]  +  _VALIDITY_LARGE · validity[i]  +  _TIEBREAK_COEFF · pos[i]

    Because ``_VALIDITY_LARGE`` is much larger than any reasonable score
    delta, the softmax always prefers a valid position over an invalid
    one regardless of their scores; among valid positions, smaller score
    wins; ties break toward the later position.

    **When no position is valid.** The softmax still runs and produces a
    weighted average over all positions — effectively garbage. Callers
    must ensure at least one valid position exists within the causal
    window at every query position whose output is actually consumed, or
    wrap the result in a ``select`` against a sentinel literal.

    Compile cost: exactly one vanilla attention head.

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to read. ``len(value) <= pos_encoding.d_pos``.

    Returns:
        Attn node of width ``len(value)``.

    See also:
        :func:`attend_argmax_where` — maximum-score dual.
    """
    assert len(score) == 1, "attend_argmin_where expects a 1D scalar score"
    assert len(validity) == 1, "attend_argmin_where expects a 1D boolean validity"
    d_head = _assert_value_fits(pos_encoding, value)

    # key_in = [pos_encoding (d_pos), score (1), validity (1)]
    # Slow-sin row of pos_encoding carries the linear-in-position tiebreak.
    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    key_in = Concatenate([pos_encoding, score, validity])
    key_matrix = torch.zeros((len(key_in), d_head))
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq   # latest-valid-position tiebreak
    key_matrix[d_pos, 0] = -1.0                          # smaller score → larger logit
    key_matrix[d_pos + 1, 0] = _VALIDITY_LARGE           # validity dominates score

    return _build_selection_attn(pos_encoding, key_in, key_matrix, value)


def attend_argmax_where(
    pos_encoding: PosEncoding,
    score: Node,
    validity: Node,
    value: Node,
) -> Node:
    """Argmax of ``score`` restricted to positions where ``validity`` is true.

    Sign-flipped twin of :func:`attend_argmin_where`. Same caveats about
    all-invalid windows.

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to read.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmax_where expects a 1D scalar score"
    assert len(validity) == 1, "attend_argmax_where expects a 1D boolean validity"
    d_head = _assert_value_fits(pos_encoding, value)

    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    key_in = Concatenate([pos_encoding, score, validity])
    key_matrix = torch.zeros((len(key_in), d_head))
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq   # latest-valid-position tiebreak
    key_matrix[d_pos, 0] = 1.0                           # larger score → larger logit
    key_matrix[d_pos + 1, 0] = _VALIDITY_LARGE           # validity dominates score

    return _build_selection_attn(pos_encoding, key_in, key_matrix, value)


def attend_argmin_above_integer(
    pos_encoding: PosEncoding,
    score: Node,
    indicators_above: Node,
    threshold_onehot: Node,
    value: Node,
) -> Node:
    """Argmin of ``score`` among positions strictly above a runtime threshold.

    This is the "next-above-threshold" selection primitive used by the
    V1 sort variant. It solves the pattern "give me the smallest score
    strictly greater than a threshold that's known per-query-position"
    under the vanilla-attention constraint.

    The key trick is an **indicator basis** on the key side. Because
    the comparison ``score_i > threshold_j`` is a step function that
    mixes per-key and per-query info, bilinear ``Q·K^T`` cannot compute
    it directly. Instead, for a fixed integer-valued score with a known
    set of possible thresholds ``{t_0, …, t_{N-1}}``, we precompute at
    each key position a width-``N`` indicator vector
    ``indicators_above[c] = I(score_i > t_c)``. The query side provides
    a one-hot ``threshold_onehot`` selecting which threshold applies at
    this query position. The bilinear sum then evaluates to

        Σ_c threshold_onehot_j[c] · indicators_above_i[c]
            = I(score_i > threshold_j)

    exactly, entirely inside the attention head.

    Combining this with a ``-score_i + tiebreak`` term in column 0 and
    a large above-bonus in columns ``1..N`` gives an attention logit of

        _QUERY_GAIN · (−score_i + tiebreak·pos_scalar_i)
            + _ABOVE_BONUS_LOGIT · I(score_i > threshold_j)

    whose argmax (argmin of score) is the smallest-score position
    strictly above the threshold.

    Caller responsibilities.

    - ``indicators_above`` must be built once at each key position.
      The caller decides the set of possible thresholds. A typical
      pattern for sorting digits 0..9: precompute
      ``[I(digit > -1), I(digit > 0), I(digit > 1), …, I(digit > 8)]``
      (width 10), one slot per possible ``prev_digit`` value in
      ``{-1, 0, 1, …, 8}``.
    - ``threshold_onehot`` must be a ``{0, 1}`` one-hot of the same
      width, with exactly one entry set to 1 at each query position
      indicating which threshold that position uses.
    - If the query's threshold is such that no valid position exists
      (e.g. threshold 9 with max digit 9), the output is garbage — wrap
      the result in a ``select`` against a sentinel at the call site.

    Compile cost: one vanilla attention head. The width of the
    indicator basis determines ``d_head`` (roughly
    ``1 + N + len(value)``).

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node (score at each key position).
        indicators_above: Width-``N`` node where slot ``c`` is the
            precomputed indicator ``I(score_i > threshold_c)``.
        threshold_onehot: Width-``N`` node whose value at each query
            position is a one-hot selecting the active threshold.
        value: Node to read at the selected key position.

    Returns:
        Attn node of width ``len(value)``.
    """
    assert len(score) == 1, "attend_argmin_above_integer expects a 1D scalar score"
    assert len(indicators_above) == len(threshold_onehot), (
        "indicators_above and threshold_onehot must have the same width "
        f"(got {len(indicators_above)} and {len(threshold_onehot)})"
    )
    n_thresholds = len(indicators_above)
    d_value = len(value)
    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    # d_head layout:
    #   col 0:                             score + tiebreak logit
    #   cols 1..n_thresholds:              threshold_onehot · indicators_above terms
    #   cols n_thresholds+1 .. +d_value:   value pass-through
    d_head = 1 + n_thresholds + d_value

    query_in = Concatenate([pos_encoding, threshold_onehot])
    key_in = Concatenate([pos_encoding, score, indicators_above])

    # --- Query matrix, shape (d_pos + n_thresholds, d_head) ---
    query_matrix = torch.zeros((len(query_in), d_head))
    # Col 0: stable positive gain for the scoring logit.
    query_matrix[d_pos - 1, 0] = _QUERY_GAIN
    # Cols 1..n_thresholds: _ABOVE_BONUS · threshold_onehot[c] routed
    # to the matching column for the bilinear rendezvous with
    # indicators_above on the key side.
    for c in range(n_thresholds):
        query_matrix[d_pos + c, 1 + c] = _ABOVE_BONUS

    # --- Key matrix, shape (d_pos + 1 + n_thresholds, d_head) ---
    key_matrix = torch.zeros((len(key_in), d_head))
    score_row = d_pos
    indicators_start_row = d_pos + 1
    # Col 0: -score + tiebreak · slow_sin(pos).
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq
    key_matrix[score_row, 0] = -1.0
    # Cols 1..n_thresholds: each indicator_above column.
    for c in range(n_thresholds):
        key_matrix[indicators_start_row + c, 1 + c] = 1.0

    # --- Value pass-through into the tail cols of d_head. ---
    value_matrix = torch.zeros((d_value, d_head))
    output_matrix = torch.zeros((d_head, d_value))
    for v in range(d_value):
        value_matrix[v, 1 + n_thresholds + v] = 1.0
        output_matrix[1 + n_thresholds + v, v] = 1.0

    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )


def attend_argmin_unmasked(
    pos_encoding: PosEncoding,
    score: Node,
    mask_vector: Node,
    position_onehot: Node,
    value: Node,
) -> Node:
    """Argmin of ``score`` over positions whose index isn't set in the mask.

    This is the primitive that the V4 selection-sort variant exists to
    motivate. It solves the pattern "I have a per-query-position mask
    vector that says which *input-position indices* have already been
    emitted, and I want to find the smallest-score input position whose
    index isn't in the mask yet."

    The mask / position-onehot rendezvous. At each *key* position ``i``
    we precompute a one-hot ``position_onehot_i`` of width ``N``, where
    ``N`` is the number of distinguishable input slots. At each *query*
    position ``j`` we carry a width-``N`` mask vector ``mask_vector_j``
    whose bit ``c`` is 1 iff input slot ``c`` has already been selected
    at or before ``j``. The attention logit at ``(j, i)`` then has the
    shape

        _QUERY_GAIN · (−score_i + _TIEBREAK_COEFF · pos_scalar_i)
        −_UNMASKED_PENALTY · mask_vector_j[position_i]

    The first term lives in ``d_head`` column 0 as usual. The second term
    is built by putting ``−_UNMASKED_PENALTY · mask_vector_j[c]`` in
    ``Q[j, c+1]`` and ``position_onehot_i[c]`` in ``K[i, c+1]`` for each
    ``c ∈ {0, …, N-1}``. The bilinear sum then equals
    ``−_UNMASKED_PENALTY · mask_vector_j[position_i]`` — a very negative
    penalty exactly when the query's mask has a bit set at the key's
    position index. An additional slab of ``d_head`` columns at the end
    carries ``value`` through unchanged.

    **Score constraint.** Scores must satisfy
    ``|score| <= _MAX_SCORE_UNMASKED_ABS`` (= 100). This is tighter than
    the ``_MAX_SCORE_ABS`` constraint on the other primitives because the
    penalty machinery needs headroom to let an unmasked worst-score
    position still beat a masked best-score position. If you need to sort
    larger-magnitude scores, normalise them first.

    **Score uniqueness and stable sort.** The caller controls stability.
    For a selection-sort over possibly-duplicate input items, use a
    lexicographic score like ``digit_i · N + pos_scalar_i`` so the
    scores are distinct — otherwise the softmax will weighted-average
    duplicate-score unmasked positions.

    **When every unmasked position is exhausted.** If the mask covers
    every causally-visible position, the best remaining logit is
    ``−_UNMASKED_PENALTY`` which equals ``-10000`` — still above the
    ``CAUSAL_MASK_SENTINEL``, so the attention will return the
    weighted-average of the *least-bad* masked positions rather than
    wandering into "future" positions. Callers that care about this edge
    case should wrap the result in a ``select`` against a sentinel.

    Compile cost: exactly one vanilla attention head. The cost is
    primarily in d_head, which grows with the mask width
    (``d_head = 1 + N + len(value)``).

    Args:
        pos_encoding: The graph's positional encoding node.
        score: 1D scalar node.
        mask_vector: Width-``N`` node whose value at query position ``j``
            is a ``{0, 1}`` mask — bit ``c`` set means "input slot ``c``
            has been emitted already, skip it".
        position_onehot: Width-``N`` node whose value at key position
            ``i`` is the one-hot of that position's *input slot index*.
            Must have the same width as ``mask_vector``.
        value: Node to read. ``len(value)`` can be larger than
            ``pos_encoding.d_pos`` here — ``d_head`` scales with value
            width rather than being capped at ``d_pos``.

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        unmasked argmin-of-``score`` position within the causal window.
    """
    assert len(score) == 1, "attend_argmin_unmasked expects a 1D scalar score"
    assert len(mask_vector) == len(position_onehot), (
        "mask_vector and position_onehot must have the same width "
        f"(got {len(mask_vector)} and {len(position_onehot)})"
    )
    n_slots = len(mask_vector)
    d_value = len(value)
    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()
    # Layout of d_head:
    #   col 0:                         score + tiebreak logit
    #   cols 1 .. n_slots:             mask · position_onehot dot-product terms
    #   cols n_slots+1 .. n_slots+d_value:  value pass-through
    d_head = 1 + n_slots + d_value

    query_in = Concatenate([pos_encoding, mask_vector])
    key_in = Concatenate([pos_encoding, score, position_onehot])

    # --- Query matrix, shape (d_pos + n_slots, d_head) ---
    query_matrix = torch.zeros((len(query_in), d_head))
    # Col 0: stable positive gain from the slowest-cos component of pos_enc.
    query_matrix[d_pos - 1, 0] = _QUERY_GAIN
    # Cols 1 .. n_slots: -_UNMASKED_PENALTY · mask_vector[c].
    for c in range(n_slots):
        query_matrix[d_pos + c, 1 + c] = -_UNMASKED_PENALTY

    # --- Key matrix, shape (d_pos + 1 + n_slots, d_head) ---
    key_matrix = torch.zeros((len(key_in), d_head))
    # Row order in key_in: [pos_enc (d_pos), score (1), onehot (n_slots)]
    score_row = d_pos
    onehot_start_row = d_pos + 1
    # Col 0: -score + tiebreak · slow_sin(pos).
    key_matrix[d_pos - 2, 0] = _TIEBREAK_COEFF / freq
    key_matrix[score_row, 0] = -1.0
    # Cols 1 .. n_slots: position_onehot[c]  (identity into d_head cols 1..n_slots).
    for c in range(n_slots):
        key_matrix[onehot_start_row + c, 1 + c] = 1.0

    # --- Value pass-through into the tail cols of d_head. ---
    value_matrix = torch.zeros((d_value, d_head))
    output_matrix = torch.zeros((d_head, d_value))
    for v in range(d_value):
        value_matrix[v, 1 + n_slots + v] = 1.0
        output_matrix[1 + n_slots + v, v] = 1.0

    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )


def attend_mean_where(
    pos_encoding: PosEncoding,
    validity: Node,
    value: Node,
) -> Node:
    """Uniform mean of ``value`` across positions where ``validity`` is true.

    At each query position, the attention returns the uniform average of
    ``value`` over all causally-visible positions where ``validity`` is
    +1.  Invalid positions (``validity`` = −1) receive a large negative
    logit penalty and contribute negligibly to the output.

    All valid positions share the same logit (no score term, no position
    tiebreak), so softmax assigns them equal weight — producing an exact
    mean rather than a weighted combination.

    A typical use is reduce-any: map boolean flags to 0/1 with
    :func:`~torchwright.ops.arithmetic_ops.bool_to_01`, average them
    with this primitive, and threshold the result.

    **When no position is valid.** The softmax still runs and produces a
    weighted average over all positions — effectively garbage.  Callers
    must ensure at least one valid position exists within the causal
    window at every query position whose output is consumed.

    Compile cost: one attention head (auto-split across multiple
    physical heads by the compiler when ``d_v > d_head``).
    ``d_qk = 1``, ``d_v = len(value)``.

    Args:
        pos_encoding: The graph's positional encoding node.
        validity: 1D boolean node (+1 valid, −1 invalid).
        value: Node to average.  No width constraint — the compiler
            splits wide V/O across multiple physical heads.

    Returns:
        Attn node of width ``len(value)`` equal to the uniform mean of
        ``value`` across valid key positions in the causal window.

    See also:
        :func:`attend_argmin_where` — selects one position (min score)
        instead of averaging.
    """
    assert len(validity) == 1, "attend_mean_where expects a 1D boolean validity"

    # d_qk = 1: the only scoring column carries the validity bonus.
    # Q reads from the slowest cosine of pos_encoding (stable ≈ 1).
    # K reads only validity.  No tiebreak → all valid positions get
    # the same logit → uniform softmax weights → exact mean.
    d_qk = 1
    d_v = len(value)

    query_matrix = torch.zeros((len(pos_encoding), d_qk))
    query_matrix[-1, 0] = _QUERY_GAIN

    # No tiebreak needed, so pos_encoding doesn't appear in key_in.
    key_matrix = torch.zeros((len(validity), d_qk))
    key_matrix[0, 0] = _VALIDITY_LARGE

    value_matrix = torch.eye(d_v)
    output_matrix = torch.eye(d_v)

    return Attn(
        query_in=pos_encoding,
        key_in=validity,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )


def attend_argmax_dot(
    pos_encoding: PosEncoding,
    query_vector: Node,
    key_vector: Node,
    value: Node,
    match_gain: float = 200.0,
) -> Node:
    """Argmax of a vector dot-product score.

    At each query position, the attention returns ``value`` at the
    causal-window position whose ``key_vector`` has the highest dot
    product with ``query_vector``.  Ties break toward the earliest
    (first) position — "first match wins."

    The logit at key position ``i`` seen from query position ``j`` is

        match_gain · (query_vector_j · key_vector_i)
        − (match_gain / 100) · pos_scalar_i

    The tiebreak coefficient is derived from ``match_gain`` so the
    caller controls both match strength and tiebreak hardness through
    one parameter.  Logit per position = ``match_gain / 100``.
    Match dominates tiebreak for sequences up to 200 positions.
    Increase ``match_gain`` for harder selection over longer sequences.

    **Type isolation.**  This primitive does not include a validity
    parameter.  Callers should use
    :func:`~torchwright.ops.logic_ops.cond_gate` to zero out
    ``key_vector`` and ``value`` at non-participating positions.  A
    zero ``key_vector`` produces a dot product of 0, well below
    ``match_gain`` for any matching position — providing effective type
    isolation without a separate validity signal.

    Compile cost: one attention head (auto-split across multiple
    physical heads by the compiler when ``d_v > d_head``).
    ``d_qk = len(query_vector) + 1``, ``d_v = len(value)``.

    Args:
        pos_encoding: The graph's positional encoding node.
        query_vector: Width-``W`` node at each query position (e.g. a
            column one-hot mapped to 0/1 via ``bool_to_01``).
        key_vector: Width-``W`` node at each key position (e.g. a
            visibility mask in ±1).  Must have the same width as
            ``query_vector``.
        value: Node to read at the winning position.  No width
            constraint — the compiler splits wide V/O across
            multiple physical heads.
        match_gain: Coefficient applied to the dot-product term.

    Returns:
        Attn node of width ``len(value)`` equal to ``value`` at the
        best-matching key position within the causal window.

    See also:
        :func:`attend_argmax_where` — scalar-score variant with
        explicit validity.
    """
    assert len(query_vector) == len(key_vector), (
        "query_vector and key_vector must have the same width "
        f"(got {len(query_vector)} and {len(key_vector)})"
    )
    W = len(query_vector)
    d_v = len(value)

    # d_qk layout:
    #   cols 0..W-1:  match dimensions (query_vector · key_vector)
    #   col  W:       position tiebreak (earliest wins)
    d_qk = W + 1

    d_pos = pos_encoding.d_pos
    freq = pos_encoding.slow_sin_freq()

    # --- Query ---
    # Columns 0..W-1: match_gain * query_vector[c]
    # Column W: _QUERY_GAIN from the slowest cosine of pos_encoding
    query_in = Concatenate([pos_encoding, query_vector])
    query_matrix = torch.zeros((len(query_in), d_qk))
    for c in range(W):
        query_matrix[d_pos + c, c] = match_gain
    query_matrix[d_pos - 1, W] = _QUERY_GAIN

    # --- Key ---
    # Columns 0..W-1: key_vector[c] (identity)
    # Column W: -tiebreak * slow_sin(pos) (earliest wins).
    # The tiebreak is derived from match_gain so the caller controls
    # both match and tiebreak strength through one parameter.
    # Logit per position = match_gain / _DOT_TB_DIVISOR.
    # Match dominates tiebreak when span < 2 * _DOT_TB_DIVISOR (= 200).
    # Increase match_gain for harder selection over longer sequences.
    _DOT_TB_DIVISOR = 100.0
    dot_tiebreak = match_gain / (_QUERY_GAIN * _DOT_TB_DIVISOR)
    key_in = Concatenate([pos_encoding, key_vector])
    key_matrix = torch.zeros((len(key_in), d_qk))
    for c in range(W):
        key_matrix[d_pos + c, c] = 1.0
    key_matrix[d_pos - 2, W] = -dot_tiebreak / freq

    # --- Value / Output: identity pass-through ---
    value_matrix = torch.eye(d_v)
    output_matrix = torch.eye(d_v)

    return Attn(
        query_in=query_in,
        key_in=key_in,
        value_in=value,
        query_matrix=query_matrix,
        key_matrix=key_matrix,
        value_matrix=value_matrix,
        output_matrix=output_matrix,
    )
