"""Past — handle to all positions' state, accessed via attention-like primitives.

The framework constructs a `Past` and passes it to your `forward()`. You
never construct one yourself. You query it through the methods below;
you never hold raw values from past positions.

## Publish, then attend

To make a Vec visible to `past.*` calls — at this position and at every
later position — call `past.publish(name, vec)` during `forward()`. After
publish, the Vec is one of the candidates that `past.*` searches.

Publishing is the analog of writing a column to the residual stream: a
later attention layer (or a later position's attention) can read it. The
order of publishes within a single `forward()` matters: you can only
attend to a name *after* you've published it (or after some earlier
position published it). This mirrors the transformer rule that an
attention layer reads from positions' residual streams as of the most
recent prior write.

`past.*` searches all positions ≤ the current one, so the current
position can attend to itself for any name it has already published —
the analog of self-attention. There is no separate "include_self" knob.

## Auto-published input slots

At the start of every position the framework auto-publishes:

- `input.type`: a one-hot Vec of width N_TYPES with the bit set for
  the input token's type.
- `input.<slot>`: a 1-shape Vec carrying the input token's value for
  each declared slot, e.g. `input.col`, `input.x`.

These are queryable like anything else. Publishing under a name that
starts with `input.` is forbidden — those names are reserved.

## Depth

Depth follows the producer-must-be-ready-before-consumer rule and
varies by primitive:

- `pick_argmax` / `pick_argmin` / `lookup` / `pick_most_recent`:
  `max(query.depth, deepest key, deepest value) + 1`.
- `pick_above_argmin`: same as above plus `threshold.depth`.
- `pick_argmax_by` / `pick_argmin_by`: `max(deepest score, deepest
  value) + 1` (no query).
- `pick_above_argmin_by`: same plus `threshold.depth`.
- `mean`: `max(deepest contributor) + 1`.

Every position that contributes (under either name) raises the
result's depth, not just the one that wins the pick.

## Margin and blending

`pick_*` and `lookup` share a uniqueness threshold of `1.0`. When the
top score exceeds the second-best by 1.0 or more, `pick_*` returns a
clean pick and `lookup` succeeds. When the gap is smaller than 1.0,
`pick_*` blends linearly between the top two candidates and `lookup`
raises with a diagnostic message naming the two close scores. The
threshold is fixed (not configurable) because the project's designed
keys — E8 codes, quadratic-equality, one-hot — all produce score gaps
well above 1.0 by design; a near-tie is evidence of a key-design bug,
not something to tune around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..runtime.noise import add_noise
from . import _runtime
from .vec import Vec, _make_vec

if TYPE_CHECKING:
    from ..runtime.embedding import Layout
    from .tokens import Token


MARGIN: float = 1.0


# Stable per-op IDs for noise seeding (separate ranges from PWL IDs to
# guarantee they never collide with each other across runs).
_ID_PICK_ARGMAX = 9_000_001
_ID_PICK_ARGMIN = 9_000_002
_ID_PICK_ABOVE_ARGMIN = 9_000_003
_ID_LOOKUP = 9_000_004
_ID_PICK_MOST_RECENT = 9_000_005
_ID_PICK_ARGMAX_BY = 9_000_006
_ID_PICK_ARGMIN_BY = 9_000_007
_ID_PICK_ABOVE_ARGMIN_BY = 9_000_008
_ID_MEAN = 9_000_009


@dataclass
class _Record:
    """Per-position record stored in Past: input token + published Vecs."""

    input_token: "Token"
    exports: dict[str, Vec]


class Past:
    """Handle to all positions. Constructed by the framework; never by user code."""

    def __init__(self, layout: "Layout"):
        self._layout = layout
        self._records: list[_Record] = []
        # In-flight position state: set by _begin_position before the
        # framework calls forward(), finalized into a record by
        # _end_position after forward() returns. None outside the
        # forward dispatch.
        self._current_input_token: "Token | None" = None
        self._pending: dict[str, Vec] = {}

    # --- Public API: publish ---

    def publish(self, name: str, vec: Vec) -> None:
        """Make `vec` queryable as `name` at this and every later position.

        Subsequent `past.*` calls within the current forward (and at all
        later positions) will see `name` as one of the candidates. The
        analog of writing a column to the residual stream.

        Re-publishing under the same name overwrites — analogous to a
        residual-stream column being rewritten by a later layer.

        Names starting with `input.` are reserved (auto-published by
        the framework) and cannot be republished.

        Must be called inside `forward()`.
        """
        if not _runtime._FORWARD_RUNNING:
            raise RuntimeError(
                "publish() can only be called inside forward()"
            )
        if self._current_input_token is None:
            raise RuntimeError(
                "publish() requires an active position; called outside "
                "the framework's forward dispatch"
            )
        if name.startswith("input."):
            raise ValueError(
                f"Cannot publish under reserved name {name!r}; "
                f"`input.*` names are auto-published by the framework"
            )
        self._pending[name] = vec

    # --- Framework lifecycle hooks (not part of the agent-facing API) ---

    def _begin_position(self, input_token: "Token") -> None:
        """Framework-internal: open a new position. Auto-publishes input.*."""
        self._current_input_token = input_token
        self._pending = self._auto_inputs(input_token)

    def _end_position(self) -> None:
        """Framework-internal: finalize the in-flight position into a record."""
        assert self._current_input_token is not None
        self._records.append(
            _Record(
                input_token=self._current_input_token,
                exports=dict(self._pending),
            )
        )
        self._current_input_token = None
        self._pending = {}

    def _abort_position(self) -> None:
        """Framework-internal: discard the in-flight position without
        finalizing. Used when forward() raises so partial publishes
        don't contaminate future state."""
        self._current_input_token = None
        self._pending = {}

    def _add(self, input_token: "Token", exports: dict[str, Vec]) -> None:
        """Framework-internal: append a finalized record (used by tests).

        Auto-published `input.*` entries are merged in alongside the
        supplied exports. Agent code never calls this — it uses
        `publish` during forward.
        """
        full_exports = self._auto_inputs(input_token)
        full_exports.update(exports)
        self._records.append(
            _Record(input_token=input_token, exports=full_exports)
        )

    def _auto_inputs(self, input_token: "Token") -> dict[str, Vec]:
        """Build the auto-published input.* exports for a token.

        Iterates over every slot declared on the canonical token type
        (not just the slots the caller passed in `Token.values`) so an
        omitted slot is still queryable as `input.<slot>` with value 0
        — consistent with `extract_*_slot`'s "missing slot reads as 0"
        behavior.
        """
        out: dict[str, Vec] = {}
        n_types = len(self._layout.types)
        type_data = np.zeros(n_types, dtype=np.float64)
        type_data[self._layout.type_columns[input_token.type.name]] = 1.0
        out["input.type"] = _make_vec(type_data, depth=0)
        canonical = self._layout.types_by_name[input_token.type.name]
        for slot_name in canonical.slots:
            value = input_token.values.get(slot_name, 0)
            out[f"input.{slot_name}"] = _make_vec(
                np.array([float(value)], dtype=np.float64), depth=0
            )
        return out

    # --- Internal lookup helpers ---

    def _all_position_indices(self) -> range:
        """All indices to scan: finalized records plus the in-flight position
        (if currently inside a forward call)."""
        n = len(self._records)
        if self._current_input_token is not None:
            return range(n + 1)
        return range(n)

    def _resolve(self, position_idx: int, name: str) -> Vec | None:
        """Look up the Vec under `name` at position `position_idx`.

        Returns `None` if the position doesn't have `name` (the position
        skips that lookup). The in-flight position uses `_pending`;
        finalized ones use their `_Record.exports`.
        """
        n = len(self._records)
        if position_idx < n:
            exports = self._records[position_idx].exports
        elif position_idx == n and self._current_input_token is not None:
            exports = self._pending
        else:
            return None
        return exports.get(name)

    def _collect_qk(
        self,
        query: Vec,
        key_name: str,
        value_name: str,
    ) -> list[tuple[int, Vec, Vec]]:
        """Return [(position_idx, key_vec, value_vec)] for positions that
        have both names. Validates each key matches `query.shape`."""
        out: list[tuple[int, Vec, Vec]] = []
        for i in self._all_position_indices():
            k = self._resolve(i, key_name)
            v = self._resolve(i, value_name)
            if k is None or v is None:
                continue
            if k.shape != query.shape:
                raise ValueError(
                    f"key {key_name!r} at position {i} has shape {k.shape}, "
                    f"does not match query shape {query.shape}"
                )
            out.append((i, k, v))
        return out

    def _collect_score(
        self,
        score_name: str,
        value_name: str,
    ) -> list[tuple[int, Vec, Vec]]:
        """Return [(position_idx, score_vec, value_vec)] where score is a
        1-shape Vec (the precomputed scalar)."""
        out: list[tuple[int, Vec, Vec]] = []
        for i in self._all_position_indices():
            s = self._resolve(i, score_name)
            v = self._resolve(i, value_name)
            if s is None or v is None:
                continue
            if s.shape != 1:
                raise ValueError(
                    f"score_name {score_name!r} at position {i} has shape "
                    f"{s.shape}, must be a 1-shape Vec for *_by primitives"
                )
            out.append((i, s, v))
        return out

    # --- Query-keyed (score = query · key at each position) ---

    def pick_argmax(
        self, query: Vec, key_name: str, value_name: str
    ) -> Vec:
        """Pick the position with the highest `query·key` score; return its value."""
        kv = self._collect_qk(query, key_name, value_name)
        if not kv:
            raise RuntimeError(
                f"pick_argmax: no position has both "
                f"{key_name!r} and {value_name!r}"
            )
        scores = np.array([float(np.dot(query._data, k._data)) for _, k, _ in kv])
        result = _blend_pick(scores, [v for _, _, v in kv], mode="max")
        depth = _depth_for(query, kv) + 1
        return _make_vec(add_noise(result, _ID_PICK_ARGMAX), depth=depth)

    def pick_argmin(
        self, query: Vec, key_name: str, value_name: str
    ) -> Vec:
        """Pick the position with the lowest `query·key` score; return its value."""
        kv = self._collect_qk(query, key_name, value_name)
        if not kv:
            raise RuntimeError(
                f"pick_argmin: no position has both "
                f"{key_name!r} and {value_name!r}"
            )
        scores = np.array([float(np.dot(query._data, k._data)) for _, k, _ in kv])
        result = _blend_pick(scores, [v for _, _, v in kv], mode="min")
        depth = _depth_for(query, kv) + 1
        return _make_vec(add_noise(result, _ID_PICK_ARGMIN), depth=depth)

    def pick_above_argmin(
        self, query: Vec, key_name: str, value_name: str, threshold: Vec
    ) -> Vec:
        """Among positions with `query·key > threshold`, pick the lowest-scoring; return its value."""
        if threshold.shape != 1:
            raise ValueError(
                f"threshold must be a 1-shape Vec, got shape {threshold.shape}"
            )
        kv = self._collect_qk(query, key_name, value_name)
        if not kv:
            raise RuntimeError(
                f"pick_above_argmin: no position has both "
                f"{key_name!r} and {value_name!r}"
            )
        scores = np.array([float(np.dot(query._data, k._data)) for _, k, _ in kv])
        cutoff = float(threshold._data[0])
        mask = scores > cutoff
        filtered_kv = [item for item, m in zip(kv, mask) if m]
        if not filtered_kv:
            raise RuntimeError(
                f"pick_above_argmin: no position has score > {cutoff} "
                f"on key {key_name!r}"
            )
        filtered_scores = scores[mask]
        result = _blend_pick(
            filtered_scores, [v for _, _, v in filtered_kv], mode="min"
        )
        # Depth: every candidate (filtered or not) contributes — the
        # transformer attention has to wait on all keys/values to decide.
        depth = max(_depth_for(query, kv), threshold.depth) + 1
        return _make_vec(add_noise(result, _ID_PICK_ABOVE_ARGMIN), depth=depth)

    def lookup(
        self, query: Vec, key_name: str, value_name: str
    ) -> Vec:
        """Equality lookup — same `query·key` scoring as `pick_*`, but
        raises if the top score doesn't exceed the second-best by at
        least 1.0, or if the search set is empty. Use when you're
        certain exactly one position should match.

        See the module docstring for the shared margin/blending
        threshold rationale.
        """
        kv = self._collect_qk(query, key_name, value_name)
        if not kv:
            raise RuntimeError(
                f"lookup: no position has both "
                f"{key_name!r} and {value_name!r}"
            )
        scores = np.array([float(np.dot(query._data, k._data)) for _, k, _ in kv])
        order = np.argsort(-scores)
        top = order[0]
        if len(scores) >= 2:
            runner = order[1]
            gap = scores[top] - scores[runner]
            if gap < MARGIN:
                raise RuntimeError(
                    f"lookup: ambiguous match (gap {gap:.4f} < {MARGIN}). "
                    f"Top: position {kv[top][0]} score {scores[top]:.4f}; "
                    f"runner-up: position {kv[runner][0]} score {scores[runner]:.4f}"
                )
        result = kv[top][2]._data.copy()
        depth = _depth_for(query, kv) + 1
        return _make_vec(add_noise(result, _ID_LOOKUP), depth=depth)

    def pick_most_recent(
        self, query: Vec, key_name: str, value_name: str
    ) -> Vec:
        """Among positions matching the query (within `MARGIN` of the
        top `query·key` score across all candidates), return the value
        at the most recent. Raises only if no position has both names.

        "Matching" is defined relative to the candidate set's top score
        — if no position truly matches the query the function still
        returns a value (degraded to recency among similarly-low
        scores), mirroring the real graph's
        `attend_most_recent_matching` behavior. Wrap with a
        select-against-sentinel pattern at the call site if the
        caller can't guarantee at least one real match exists.
        """
        kv = self._collect_qk(query, key_name, value_name)
        if not kv:
            raise RuntimeError(
                f"pick_most_recent: no position has both "
                f"{key_name!r} and {value_name!r}"
            )
        scores = np.array([float(np.dot(query._data, k._data)) for _, k, _ in kv])
        best = float(scores.max())
        matching = [i for i, s in enumerate(scores) if best - s < MARGIN]
        # Among matching kv entries, pick the one with the largest position index.
        chosen_kv_idx = max(matching, key=lambda i: kv[i][0])
        result = kv[chosen_kv_idx][2]._data.copy()
        depth = _depth_for(query, kv) + 1
        return _make_vec(add_noise(result, _ID_PICK_MOST_RECENT), depth=depth)

    # --- Aggregation across positions ---

    def mean(self, value_name: str) -> Vec:
        """Elementwise mean of `value_name` across all positions that
        published it.

        Mirrors the transformer's natural attention primitive (softmax
        over a uniform key set is a uniform mean). Single-contributor
        broadcast (e.g. PLAYER_X published once): the mean of one value
        is that value. Multi-contributor aggregation (e.g. M `BSP_NODE`
        positions each publishing an M-wide slot-vector): the mean is
        `sum / M`; if you need the sum you multiply by M yourself, or
        have producers scale their contributions by M up front.

        Recovering a sum requires `M` to be a Python int known at
        module load — PWL weights, `linear` matrices, and any
        scaling factors are frozen before `forward()` runs. That is
        why phases that use `mean × count` commit to fixed-dimension
        scenes and pad smaller fixtures with neutral values (see the
        per-phase PHASE.md's "Fixed dimensions and padding" section).

        All contributing values must have the same shape. Adds depth
        +1 over the deepest contributor."""
        contributors: list[Vec] = []
        for i in self._all_position_indices():
            v = self._resolve(i, value_name)
            if v is not None:
                contributors.append(v)
        if not contributors:
            raise RuntimeError(
                f"mean: no position has {value_name!r}"
            )
        shape = contributors[0].shape
        for v in contributors[1:]:
            if v.shape != shape:
                raise ValueError(
                    f"mean: contributors of {value_name!r} have inconsistent "
                    f"shapes ({shape} vs {v.shape})"
                )
        stacked = np.stack([v._data for v in contributors], axis=0)
        result = stacked.mean(axis=0)
        depth = max(v.depth for v in contributors) + 1
        return _make_vec(add_noise(result, _ID_MEAN), depth=depth)

    # --- Score-keyed (score is a precomputed scalar exported by the producer) ---

    def pick_argmax_by(self, score_name: str, value_name: str) -> Vec:
        """Pick the position with the highest precomputed `score_name`; return its value."""
        sv = self._collect_score(score_name, value_name)
        if not sv:
            raise RuntimeError(
                f"pick_argmax_by: no position has both "
                f"{score_name!r} and {value_name!r}"
            )
        scores = np.array([float(s._data[0]) for _, s, _ in sv])
        result = _blend_pick(scores, [v for _, _, v in sv], mode="max")
        depth = _depth_for(None, sv) + 1
        return _make_vec(add_noise(result, _ID_PICK_ARGMAX_BY), depth=depth)

    def pick_argmin_by(self, score_name: str, value_name: str) -> Vec:
        """Pick the position with the lowest precomputed `score_name`; return its value."""
        sv = self._collect_score(score_name, value_name)
        if not sv:
            raise RuntimeError(
                f"pick_argmin_by: no position has both "
                f"{score_name!r} and {value_name!r}"
            )
        scores = np.array([float(s._data[0]) for _, s, _ in sv])
        result = _blend_pick(scores, [v for _, _, v in sv], mode="min")
        depth = _depth_for(None, sv) + 1
        return _make_vec(add_noise(result, _ID_PICK_ARGMIN_BY), depth=depth)

    def pick_above_argmin_by(
        self, score_name: str, value_name: str, threshold: Vec
    ) -> Vec:
        """Among positions with `score_name > threshold`, pick the lowest-scoring; return its value."""
        if threshold.shape != 1:
            raise ValueError(
                f"threshold must be a 1-shape Vec, got shape {threshold.shape}"
            )
        sv = self._collect_score(score_name, value_name)
        if not sv:
            raise RuntimeError(
                f"pick_above_argmin_by: no position has both "
                f"{score_name!r} and {value_name!r}"
            )
        scores = np.array([float(s._data[0]) for _, s, _ in sv])
        cutoff = float(threshold._data[0])
        mask = scores > cutoff
        filtered_sv = [item for item, m in zip(sv, mask) if m]
        if not filtered_sv:
            raise RuntimeError(
                f"pick_above_argmin_by: no position has "
                f"{score_name!r} > {cutoff}"
            )
        filtered_scores = scores[mask]
        result = _blend_pick(
            filtered_scores, [v for _, _, v in filtered_sv], mode="min"
        )
        depth = max(_depth_for(None, sv), threshold.depth) + 1
        return _make_vec(add_noise(result, _ID_PICK_ABOVE_ARGMIN_BY), depth=depth)


def _blend_pick(
    scores: np.ndarray,
    values: list[Vec],
    *,
    mode: str,
) -> np.ndarray:
    """Apply the 1.0-margin pick/blend rule to a set of (score, value) pairs.

    `mode` is "max" (highest score wins) or "min" (lowest wins).
    Returns the chosen / blended raw data array (no noise applied).
    """
    if mode == "max":
        order = np.argsort(-scores)
    elif mode == "min":
        order = np.argsort(scores)
    else:  # pragma: no cover
        raise ValueError(f"_blend_pick mode must be 'max' or 'min', got {mode!r}")
    top = order[0]
    if len(scores) == 1:
        return values[top]._data.copy()
    runner = order[1]
    if mode == "max":
        gap = float(scores[top] - scores[runner])
    else:
        gap = float(scores[runner] - scores[top])
    if gap >= MARGIN:
        return values[top]._data.copy()
    if values[top].shape != values[runner].shape:
        raise ValueError(
            f"blend-zone candidates have inconsistent value shapes "
            f"({values[top].shape} vs {values[runner].shape})"
        )
    top_w = 0.5 + gap / 2.0
    runner_w = 1.0 - top_w
    return top_w * values[top]._data + runner_w * values[runner]._data


def _depth_for(
    query: Vec | None,
    kv: list[tuple[int, Vec, Vec]],
) -> int:
    """Worst-case depth of inputs to a past op (before the +1 for the op itself)."""
    deepest_k = max((k.depth for _, k, _ in kv), default=0)
    deepest_v = max((v.depth for _, _, v in kv), default=0)
    if query is None:
        return max(deepest_k, deepest_v)
    return max(query.depth, deepest_k, deepest_v)
