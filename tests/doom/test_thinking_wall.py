"""Phase A Part 2 thinking-token trace + dual-path tests.

Two kinds of assertion:

1. **Dual-path** (``TestThinkingWallDualPath``): compares the per-wall
   HIT_FULL/HIT_X/HIT_Y values the compiled graph emits against a pure-
   Python reference of the same intersect-segment math.  If all three
   hits per wall agree with the reference across the parameterised
   player/input configurations, the hit-math path through the embedding
   carrier is end-to-end correct (current-wall_index attention, prompt-
   geometry attention, hit math, VALUE ID emission).

2. **Full-sequence trace** (``TestThinkingSequenceTrace``): asserts the
   compiled graph emits the exact expected token at every position of
   the Part 2 thinking sequence — markers, identifiers, VALUE_0 stubs,
   HIT_* values, RESOLVED_* chain, and the SORTED_WALL hand-off.  This
   covers the cascade state machine itself (which identifier comes
   next, when RESOLVED fires, when SORTED is handed off) independently
   of hit-math correctness.

The references in both tests are the same intersect-segment math the
WALL stage runs (see ``torchwright.doom.stages.wall._collision_validity``),
reimplemented in pure Python so we can compare without depending on the
WALL stage's output (which is only consumed by EOS — the dual paths are
independent).
"""

import math

import pytest

from torchwright.doom.compile import compile_game, step_frame
from torchwright.doom.embedding import IDENTIFIER_NAMES, value_id, vocab_id
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.doom.map_subset import build_scene_subset
from torchwright.doom.trace import FrameTrace
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment

_TRIG = generate_trig_table()
_MOVE_SPEED = 0.3
_TURN_SPEED = 4

# Thinking phase layout per wall (Part 2, 27 autoregressive steps):
#
#   step+0:  marker          (THINKING_WALL_N)
#   step+1:  BSP_RANK_ID     step+2:  VALUE (stub = VALUE_0)
#   step+3:  IS_RENDERABLE_ID step+4:  VALUE (stub)
#   step+5:  CROSS_A_ID       step+6:  VALUE (stub)
#   step+7:  DOT_A_ID         step+8:  VALUE (stub)
#   step+9:  CROSS_B_ID       step+10: VALUE (stub)
#   step+11: DOT_B_ID         step+12: VALUE (stub)
#   step+13: T_LO_ID          step+14: VALUE (stub)
#   step+15: T_HI_ID          step+16: VALUE (stub)
#   step+17: VIS_LO_ID        step+18: VALUE (stub)
#   step+19: VIS_HI_ID        step+20: VALUE (stub)
#   step+21: HIT_FULL_ID      step+22: VALUE (hit_full ∈ {0, 1})
#   step+23: HIT_X_ID         step+24: VALUE (hit_x ∈ {0, 1})
#   step+25: HIT_Y_ID         step+26: VALUE (hit_y ∈ {0, 1})
#
# After the last wall's HIT_Y value, the RESOLVED chain runs:
#   +0: RESOLVED_X_ID   +1: VALUE (stub)
#   +2: RESOLVED_Y_ID   +3: VALUE (stub)
#   +4: RESOLVED_ANGLE_ID +5: VALUE (stub)
#   +6: SORTED_WALL (hand-off)
_STEPS_PER_WALL = 27
_HF_OFFSET = 22
_HX_OFFSET = 24
_HY_OFFSET = 26

_RESOLVED_BASE_OFFSET = 0  # from the first post-HIT_Y-of-last-wall step
_RESOLVED_X_OFFSET = 0
_RESOLVED_Y_OFFSET = 2
_RESOLVED_ANGLE_OFFSET = 4
_SORTED_WALL_OFFSET = 6


def _box_room_config():
    # Tiny screen for the thinking-token tests — they only consume
    # ``trace.token_id_log`` from the thinking phase, so resolution
    # doesn't affect what's measured.  Combined with
    # ``stop_after_thinking=True`` in the per-frame run, this drops
    # the test runtime by ~10x vs the 64×80 walkthrough config.
    return RenderConfig(
        screen_width=16,
        screen_height=20,
        fov_columns=32,
        trig_table=_TRIG,
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


def _box_room_segments(half=5.0):
    return [
        Segment(
            ax=half, ay=-half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=0
        ),
        Segment(
            ax=-half, ay=-half, bx=-half, by=half, color=(0.8, 0.2, 0.1), texture_id=1
        ),
        Segment(
            ax=-half, ay=half, bx=half, by=half, color=(0.8, 0.2, 0.1), texture_id=2
        ),
        Segment(
            ax=-half, ay=-half, bx=half, by=-half, color=(0.8, 0.2, 0.1), texture_id=3
        ),
    ]


def _player_velocity(angle: int, inputs: PlayerInput) -> tuple[float, float]:
    """Reference implementation of INPUT-stage velocity computation.

    Mirrors what ``stages.input`` ultimately broadcasts.  The compiled
    path goes through trig table lookup + the strafe sign-flip; here we
    compute it directly from the trig table.
    """
    # Apply turning to get the new angle (one frame's worth).
    new_angle = angle
    if inputs.turn_left:
        new_angle -= _TURN_SPEED
    if inputs.turn_right:
        new_angle += _TURN_SPEED
    new_angle = new_angle % 256

    cos_t = _TRIG[new_angle, 0]
    sin_t = _TRIG[new_angle, 1]

    fwd = (1.0 if inputs.forward else 0.0) - (1.0 if inputs.backward else 0.0)
    strafe = (1.0 if inputs.strafe_right else 0.0) - (
        1.0 if inputs.strafe_left else 0.0
    )

    vel_x = _MOVE_SPEED * (fwd * cos_t - strafe * sin_t)
    vel_y = _MOVE_SPEED * (fwd * sin_t + strafe * cos_t)
    return vel_x, vel_y


def _ref_collision_validity(den, num_t, num_u, epsilon=0.05) -> bool:
    """Ray-segment intersection validity check.

    Same predicates as ``stages.wall._collision_validity`` but on
    floats: the ray hits iff the parametric (t, u) both lie in [0, 1]
    with the configured epsilon margin and the determinant is non-zero.
    """
    sign_den = 1.0 if den > 0 else -1.0
    adj_t = num_t * sign_den
    adj_u = num_u * sign_den
    abs_den = abs(den)
    return (
        abs_den > epsilon
        and adj_t > epsilon
        and (abs_den - adj_t) > -epsilon
        and adj_u > -epsilon
        and (abs_den - adj_u) > -epsilon
    )


def _ref_hits(seg: Segment, px: float, py: float, vx: float, vy: float):
    """Per-wall hit_full/hit_x/hit_y for one segment + player ray.

    Returns a 3-tuple of 0/1 ints matching what HIT_FULL_VALUE,
    HIT_X_VALUE, HIT_Y_VALUE thinking tokens emit.
    """
    ex = seg.bx - seg.ax
    ey = seg.by - seg.ay
    dax = seg.ax - px
    day = seg.ay - py

    p_dx_ey = vx * ey
    p_dy_ex = vy * ex
    p_dax_ey = dax * ey
    p_day_ex = day * ex
    p_dax_dy = dax * vy
    p_day_dx = day * vx

    num_t = p_dax_ey - p_day_ex

    den_full = p_dx_ey - p_dy_ex
    num_u_full = p_dax_dy - p_day_dx
    hit_full = _ref_collision_validity(den_full, num_t, num_u_full)

    den_x = p_dx_ey
    num_u_x = -p_day_dx
    hit_x = _ref_collision_validity(den_x, num_t, num_u_x)

    den_y = -p_dy_ex
    num_u_y = p_dax_dy
    hit_y = _ref_collision_validity(den_y, num_t, num_u_y)

    return int(hit_full), int(hit_x), int(hit_y)


class TestThinkingWallDualPath:
    """Walks per-wall HIT_* across a few player+input configurations."""

    @pytest.fixture(scope="class")
    def scene(self):
        config = _box_room_config()
        textures = default_texture_atlas()
        segs = _box_room_segments()
        subset = build_scene_subset(segs, textures)
        return config, textures, subset, segs

    @pytest.fixture(scope="class")
    def module(self, scene):
        config, textures, subset, segs = scene
        return compile_game(
            config,
            textures,
            max_walls=8,
            d=2048,
            d_head=32,
            verbose=False,
        )

    @pytest.fixture(scope="class")
    def run(self, module, scene):
        config, textures, subset, segs = scene
        cache = {}

        def _run(px, py, angle, **input_kw):
            key = (
                px,
                py,
                angle,
                tuple(sorted(input_kw.items())),
            )
            if key not in cache:
                state = GameState(
                    x=px,
                    y=py,
                    angle=angle,
                    move_speed=_MOVE_SPEED,
                    turn_speed=_TURN_SPEED,
                )
                inp = PlayerInput(**input_kw)
                trace = FrameTrace()
                frame, new_state = step_frame(
                    module,
                    state,
                    inp,
                    subset,
                    config,
                    textures=textures,
                    trace=trace,
                    stop_after_thinking=True,
                )
                cache[key] = (frame, new_state, trace, inp)
            return cache[key]

        return _run

    def _check_wall_hits(self, run, segs, px, py, angle, inp_kw):
        """Run a frame and verify per-wall HIT_*/HIT_X/HIT_Y match reference.

        Phase A Part 3: HIT_* booleans are now uniform-quantized into
        the (0, 1) range like every other VALUE — false → VALUE_0,
        true → VALUE_65535.  Convert wire-format ID back to 0/1 by
        comparing against the half-range.
        """
        _, _, trace, inputs = run(px, py, angle, **inp_kw)
        vx, vy = _player_velocity(angle, inputs)

        log = trace.token_id_log
        assert len(log) >= len(segs) * _STEPS_PER_WALL, (
            f"Token ID log too short ({len(log)} < "
            f"{len(segs) * _STEPS_PER_WALL}); thinking phase likely truncated."
        )

        mismatches = []
        for i, seg in enumerate(segs):
            ref_hf, ref_hx, ref_hy = _ref_hits(seg, px, py, vx, vy)
            base = i * _STEPS_PER_WALL
            # Wire-format VALUE for a 0/1 boolean: 0 → VALUE_0,
            # 1 → VALUE_65535.  Recover the bool by thresholding.
            comp_hf = 1 if int(log[base + _HF_OFFSET]) > 32767 else 0
            comp_hx = 1 if int(log[base + _HX_OFFSET]) > 32767 else 0
            comp_hy = 1 if int(log[base + _HY_OFFSET]) > 32767 else 0
            for name, ref, comp in [
                ("HIT_FULL", ref_hf, comp_hf),
                ("HIT_X", ref_hx, comp_hx),
                ("HIT_Y", ref_hy, comp_hy),
            ]:
                if ref != comp:
                    mismatches.append(f"wall {i} {name}: ref={ref}, compiled={comp}")
        assert not mismatches, (
            f"Thinking-token HIT_* disagrees with reference at "
            f"(px={px}, py={py}, angle={angle}, inputs={inp_kw}):\n"
            + "\n".join(mismatches)
        )

    @pytest.mark.parametrize(
        "px,py,angle,inp",
        [
            # Center, no movement: all hits zero (no ray to intersect).
            (0.0, 0.0, 0, {}),
            # Center, moving forward (east at angle 0): ray goes +x, all
            # short. With move_speed 0.3 and walls at ±5, ray length 0.3
            # vs distance ≥ 5 — no hit on any wall.
            (0.0, 0.0, 0, {"forward": True}),
            # Near east wall, moving forward: hit_full + hit_x for wall 0.
            (4.9, 0.0, 0, {"forward": True}),
            # Near south wall (y=-5), moving south (angle 192 ≈ -y):
            # hit_full + hit_y for wall 3 (south wall).
            (0.0, -4.9, 192, {"forward": True}),
            # Near north wall, moving north.
            (0.0, 4.9, 64, {"forward": True}),
            # West-side, moving west.  Player at (-3, 0) so multiple
            # walls remain renderable (avoids degenerate single-wall
            # SORTED garbage that crashes the trace harness in the
            # existing pipeline; that bug is logged separately).
            (-3.0, 0.0, 128, {"forward": True}),
            # Diagonal-ish: near east wall but on the y-axis, angle 32
            # (NE) so the ray hits east wall well below the corner
            # (intersection at u≈0.515).  Picks up east-wall hit_full
            # and hit_x without the wall-corner u=1.0 boundary case
            # which is genuinely ambiguous between exact and piecewise
            # math.
            (4.85, 0.0, 32, {"forward": True}),
            # Strafing into east wall.
            (4.9, 0.0, 64, {"strafe_right": True}),
        ],
    )
    def test_per_wall_hits_match_reference(self, run, scene, px, py, angle, inp):
        _, _, _, segs = scene
        self._check_wall_hits(run, segs, px, py, angle, inp)

    def test_full_sequence_trace(self, run, scene):
        """Walk the full Part-3 thinking sequence, asserting the exact
        token ID at every autoregressive position.

        Positions covered:
          * 8 marker positions (one per wall; host injects wall 0).
          * 8 × 13 = 104 identifier positions.
          * 8 × 10 = 80 per-wall VALUE positions for the 10 base /
            derived value types (BSP_RANK, IS_RENDERABLE, CROSS/DOT,
            T_LO/T_HI, VIS_LO/VIS_HI) — checked as ints in
            ``[0, 65535]``; per-value dual-path checks land in
            dedicated tests.
          * 8 × 3 = 24 HIT_* VALUE positions (booleans encoded as
            ``VALUE_0`` for false / ``VALUE_65535`` for true under
            uniform 16-bit quantization).
          * 3 RESOLVED identifier positions.
          * 3 RESOLVED VALUE positions (range-checked only — per-value
            dual-path checks land in the pipeline tests).
          * 1 SORTED_WALL hand-off position.

        The HIT_* values are additionally checked against the pure-
        Python reference for the subset-walls.
        """
        max_walls = 8  # matches the compile_game fixture
        # A scenario with at least one real hit so the HIT_*-value
        # assertions exercise both 0 and 1 emissions.
        px, py, angle, inp_kw = 4.9, 0.0, 0, {"forward": True}

        _, _, trace, inputs = run(px, py, angle, **inp_kw)
        vx, vy = _player_velocity(angle, inputs)
        _, _, _, segs = scene

        log = trace.token_id_log
        expected_len = max_walls * _STEPS_PER_WALL + _SORTED_WALL_OFFSET + 1
        assert len(log) >= expected_len, (
            f"token_id_log length {len(log)} < expected {expected_len} "
            f"(thinking phase may be truncated)"
        )

        # Per-slot expected identifier ID (by IDENTIFIER_NAMES order).
        expected_id_at_slot = [vocab_id(name) for name in IDENTIFIER_NAMES]

        mismatches = []

        def _check(pos, expected, label):
            actual = int(log[pos])
            if actual != expected:
                mismatches.append(
                    f"pos {pos} ({label}): expected {expected}, got {actual}"
                )

        def _check_value_range(pos, label):
            """Assert the ID at ``pos`` is a VALUE token (in [0, 65535])."""
            actual = int(log[pos])
            if not (0 <= actual <= 65535):
                mismatches.append(
                    f"pos {pos} ({label}): expected VALUE in [0, 65535], "
                    f"got {actual}"
                )

        def _check_bool_value(pos, ref_bool, label):
            """Assert the ID is the uniform-quantized boolean encoding.

            Phase A Part 3: booleans are emitted as VALUE_0 (false) /
            VALUE_65535 (true) under uniform 16-bit quantization.  The
            factor cascade has ≤1-LSB drift on GPU FP, so the actual
            argmaxed VALUE can land one integer off (VALUE_65534 still
            reads back as ≈1.0 via dequantize, which is correct).  The
            test thresholds the wire value at the half-range to
            recover the intended boolean.
            """
            actual = int(log[pos])
            recovered_bool = actual > 32767
            if recovered_bool != ref_bool:
                mismatches.append(
                    f"pos {pos} ({label}): expected bool={ref_bool} "
                    f"(wire near VALUE_{65535 if ref_bool else 0}), "
                    f"got {actual}"
                )

        # --- Per-wall assertions (all 8 walls). ---
        for wall_i in range(max_walls):
            base = wall_i * _STEPS_PER_WALL

            # Marker at base+0.
            _check(
                base + 0,
                vocab_id(f"THINKING_WALL_{wall_i}"),
                f"wall{wall_i} marker",
            )

            # 13 identifier positions at base + (1, 3, 5, ..., 25).
            # They follow IDENTIFIER_NAMES order for the first 13 entries
            # (RESOLVED_* are handled after the last wall).
            for slot in range(13):
                pos = base + 1 + 2 * slot
                _check(
                    pos,
                    expected_id_at_slot[slot],
                    f"wall{wall_i} identifier {IDENTIFIER_NAMES[slot]}",
                )

            # 10 base/derived VALUE positions (BSP_RANK..VIS_HI).  Range
            # check only — per-value dual-path tests live separately.
            for slot in range(10):
                pos = base + 2 + 2 * slot
                _check_value_range(
                    pos,
                    f"wall{wall_i} VALUE for {IDENTIFIER_NAMES[slot]}",
                )

            # HIT_* VALUEs: compare against the pure-Python reference for
            # the subset walls; phantom walls just need a valid VALUE ID
            # in {0, 65535} since the hit math ran on degenerate inputs.
            if wall_i < len(segs):
                ref_hf, ref_hx, ref_hy = _ref_hits(segs[wall_i], px, py, vx, vy)
                _check_bool_value(
                    base + _HF_OFFSET, bool(ref_hf), f"wall{wall_i} VALUE HIT_FULL"
                )
                _check_bool_value(
                    base + _HX_OFFSET, bool(ref_hx), f"wall{wall_i} VALUE HIT_X"
                )
                _check_bool_value(
                    base + _HY_OFFSET, bool(ref_hy), f"wall{wall_i} VALUE HIT_Y"
                )
            else:
                for offset, lbl in [
                    (_HF_OFFSET, "HIT_FULL"),
                    (_HX_OFFSET, "HIT_X"),
                    (_HY_OFFSET, "HIT_Y"),
                ]:
                    actual = int(log[base + offset])
                    if actual not in (value_id(0), value_id(65535)):
                        mismatches.append(
                            f"pos {base + offset} (wall{wall_i} phantom "
                            f"VALUE {lbl}): expected VALUE_0 or "
                            f"VALUE_65535, got {actual}"
                        )

        # --- RESOLVED chain after the last wall. ---
        resolved_base = max_walls * _STEPS_PER_WALL
        _check(
            resolved_base + _RESOLVED_X_OFFSET,
            vocab_id("RESOLVED_X"),
            "RESOLVED_X identifier",
        )
        _check_value_range(
            resolved_base + _RESOLVED_X_OFFSET + 1,
            "RESOLVED_X VALUE",
        )
        _check(
            resolved_base + _RESOLVED_Y_OFFSET,
            vocab_id("RESOLVED_Y"),
            "RESOLVED_Y identifier",
        )
        _check_value_range(
            resolved_base + _RESOLVED_Y_OFFSET + 1,
            "RESOLVED_Y VALUE",
        )
        _check(
            resolved_base + _RESOLVED_ANGLE_OFFSET,
            vocab_id("RESOLVED_ANGLE"),
            "RESOLVED_ANGLE identifier",
        )
        _check_value_range(
            resolved_base + _RESOLVED_ANGLE_OFFSET + 1,
            "RESOLVED_ANGLE VALUE",
        )

        # --- SORTED_WALL hand-off. ---
        _check(
            resolved_base + _SORTED_WALL_OFFSET,
            vocab_id("SORTED_WALL"),
            "SORTED_WALL hand-off",
        )

        assert not mismatches, "Trace mismatches:\n" + "\n".join(mismatches)
