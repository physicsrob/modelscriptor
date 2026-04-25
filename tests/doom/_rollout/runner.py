"""Driver that runs a scenario through ``step_frame`` and exposes the
captured token stream as a structured :class:`Rollout` view.

Built on top of the existing ``step_frame`` machinery (no new step
loop) — ``trace.token_id_log`` already holds every autoregressive
token position, including thinking, SORTED, and RENDER.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from torchwright.doom.compile import step_frame
from torchwright.doom.embedding import (
    IDENTIFIER_NAMES,
    N_VALUES,
    VALUE_RANGE_BY_NAME,
    vocab_id,
)
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import MapSubset
from torchwright.doom.thinking_readback import INT_IDENTIFIER_NAMES
from torchwright.doom.trace import FrameTrace
from torchwright.ops.quantization import DEFAULT_N_LEVELS
from torchwright.reference_renderer.types import RenderConfig


_STEPS_PER_WALL = 35  # 1 marker + 17 identifier + 17 value
_RESOLVED_X_OFFSET = 1  # within the post-walls block
_RESOLVED_Y_OFFSET = 3
_RESOLVED_ANGLE_OFFSET = 5
_SORTED_HANDOFF_OFFSET = 6


def _slot_for(identifier_name: str) -> int:
    """0-based index of the identifier in the per-wall sequence."""
    return IDENTIFIER_NAMES.index(identifier_name)


@dataclass
class Rollout:
    """Structured view over one scenario's autoregressive token stream."""

    token_id_log: List[int]
    max_walls: int

    # ------------------------------------------------------------
    # Token decoding
    # ------------------------------------------------------------

    def _dequant(self, token_id: int, name: str) -> float:
        """Decode a VALUE token to its float value.

        SORT_RESULT (and other INT identifiers) emit ``VALUE_<int>``
        directly — the token ID *is* the integer.  Continuous identifiers
        emit ``VALUE_q`` where ``q`` is the quantized scaled value, so
        the float is ``lo + q*(hi-lo)/65535``.
        """
        if name in INT_IDENTIFIER_NAMES:
            return float(int(token_id))
        lo, hi = VALUE_RANGE_BY_NAME[name]
        q = max(0, min(N_VALUES - 1, int(token_id)))
        return lo + q * (hi - lo) / float(DEFAULT_N_LEVELS - 1)

    # ------------------------------------------------------------
    # Per-wall thinking values
    # ------------------------------------------------------------

    def per_wall_value(self, wall_i: int, identifier_name: str) -> float:
        """Decoded VALUE for an identifier at a specific wall.

        Per-wall sequence: [marker, id_0, val_0, id_1, val_1, ...].
        VALUE for slot ``s`` lives at offset ``2 + 2*s`` from the
        marker, which itself sits at ``wall_i * _STEPS_PER_WALL``.
        """
        slot = _slot_for(identifier_name)
        pos = wall_i * _STEPS_PER_WALL + 2 + 2 * slot
        return self._dequant(self.token_id_log[pos], identifier_name)

    def per_wall_bool(self, wall_i: int, identifier_name: str) -> bool:
        """For VALUEs that encode 0/1 booleans, recover the bit via
        a half-range threshold (the wire VALUE_k for boolean encodings
        sits near 0 or 65535)."""
        slot = _slot_for(identifier_name)
        pos = wall_i * _STEPS_PER_WALL + 2 + 2 * slot
        return int(self.token_id_log[pos]) > 32767

    def per_wall_token(self, wall_i: int, identifier_name: str) -> int:
        """Raw token ID at the per-wall VALUE position."""
        slot = _slot_for(identifier_name)
        pos = wall_i * _STEPS_PER_WALL + 2 + 2 * slot
        return int(self.token_id_log[pos])

    def marker_token(self, wall_i: int) -> int:
        return int(self.token_id_log[wall_i * _STEPS_PER_WALL])

    def identifier_token(self, wall_i: int, identifier_name: str) -> int:
        slot = _slot_for(identifier_name)
        pos = wall_i * _STEPS_PER_WALL + 1 + 2 * slot
        return int(self.token_id_log[pos])

    # ------------------------------------------------------------
    # Resolved-state block (after the last wall)
    # ------------------------------------------------------------

    @property
    def _resolved_base(self) -> int:
        return self.max_walls * _STEPS_PER_WALL

    def resolved_value(self, identifier_name: str) -> float:
        offsets = {
            "RESOLVED_X": _RESOLVED_X_OFFSET,
            "RESOLVED_Y": _RESOLVED_Y_OFFSET,
            "RESOLVED_ANGLE": _RESOLVED_ANGLE_OFFSET,
        }
        pos = self._resolved_base + offsets[identifier_name]
        return self._dequant(self.token_id_log[pos], identifier_name)

    def resolved_identifier_token(self, identifier_name: str) -> int:
        offsets = {
            "RESOLVED_X": _RESOLVED_X_OFFSET - 1,
            "RESOLVED_Y": _RESOLVED_Y_OFFSET - 1,
            "RESOLVED_ANGLE": _RESOLVED_ANGLE_OFFSET - 1,
        }
        pos = self._resolved_base + offsets[identifier_name]
        return int(self.token_id_log[pos])

    @property
    def sorted_handoff_token(self) -> int:
        pos = self._resolved_base + _SORTED_HANDOFF_OFFSET
        return int(self.token_id_log[pos])

    # ------------------------------------------------------------
    # SORTED + RENDER stream
    # ------------------------------------------------------------

    @property
    def post_thinking_start(self) -> int:
        """First position after the SORTED_WALL hand-off token."""
        return self._resolved_base + _SORTED_HANDOFF_OFFSET + 1

    def sort_value_token_positions(self) -> List[int]:
        """Positions in ``token_id_log`` of every SORT_RESULT VALUE.

        Pattern after the hand-off: SORT_RESULT id → VALUE → RENDER...
        The SORTED_WALL marker is the previous position (the hand-off
        for slot 0; subsequent slots are emitted from RENDER's
        ``advance_wall`` next-token).
        """
        sort_result_id = vocab_id("SORT_RESULT")
        positions: List[int] = []
        i = self.post_thinking_start
        while i < len(self.token_id_log) - 1:
            if self.token_id_log[i] == sort_result_id:
                positions.append(i + 1)
                i += 2
                continue
            i += 1
        return positions

    def sort_result_wall_indices(self) -> List[int]:
        """Decode the per-sort-slot wall index from each SORT_RESULT
        VALUE position."""
        return [
            int(round(self._dequant(self.token_id_log[pos], "SORT_RESULT")))
            for pos in self.sort_value_token_positions()
        ]

    def render_positions_per_sort_slot(self) -> List[List[int]]:
        """For each sort slot, the list of token-log positions that
        emitted a RENDER token.

        Walks the post-thinking stream: a SORT_RESULT VALUE opens a new
        slot; every RENDER token until the next SORTED_WALL belongs to
        it.
        """
        render_id = vocab_id("RENDER")
        sort_result_id = vocab_id("SORT_RESULT")
        sorted_marker_id = vocab_id("SORTED_WALL")
        slots: List[List[int]] = []
        current: Optional[List[int]] = None
        in_value = False
        i = self.post_thinking_start
        while i < len(self.token_id_log):
            tok = self.token_id_log[i]
            if tok == sort_result_id:
                # The next position's VALUE opens this slot.
                in_value = True
                i += 1
                continue
            if in_value:
                current = []
                slots.append(current)
                in_value = False
                i += 1
                continue
            if tok == sorted_marker_id:
                # New slot starts on the next SORT_RESULT id.
                current = None
                i += 1
                continue
            if tok == render_id and current is not None:
                current.append(i)
            i += 1
        return slots

    # ------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------

    def first_position_of(self, token_id: int) -> Optional[int]:
        for i, t in enumerate(self.token_id_log):
            if t == token_id:
                return i
        return None


def run_rollout(
    *,
    module,
    px: float,
    py: float,
    angle: int,
    inputs: PlayerInput,
    subset: MapSubset,
    config: RenderConfig,
    move_speed: float = 0.3,
    turn_speed: int = 4,
) -> Rollout:
    """Run a single ``step_frame`` and wrap its trace as a :class:`Rollout`."""
    state = GameState(
        x=px,
        y=py,
        angle=angle,
        move_speed=move_speed,
        turn_speed=turn_speed,
    )
    trace = FrameTrace()
    step_frame(
        module,
        state,
        inputs,
        subset,
        config,
        textures=subset.textures,
        trace=trace,
    )
    max_walls = int(module.metadata.get("max_walls", 8))
    return Rollout(token_id_log=list(trace.token_id_log), max_walls=max_walls)
