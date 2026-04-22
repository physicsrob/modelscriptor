"""Phase A M4 dual-path test: thinking-token HIT_* values match reference.

Compiles the full game graph (with the thinking-wall stage), runs a
single frame, and checks that the per-wall HIT_FULL/HIT_X/HIT_Y values
emitted as thinking tokens match a Python reference computed from the
same scene + player state.

The reference is the same intersect-segment math the WALL stage runs
(see ``torchwright.doom.stages.wall._collision_validity``), reimplemented
in pure Python so we can compare without depending on the WALL stage's
output (which is only consumed by EOS — the dual paths are independent).

If a wall's three thinking-token hits agree with the reference for
every wall in the scene, the M4 thinking-token mechanism is end-to-end
correct: marker → identifier → value sequencing, current-wall_index
attention, prompt-geometry attention, hit math, value selection, and
output thinking_value all line up.
"""

import math

import pytest

from torchwright.doom.compile import compile_game, step_frame
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

# Thinking phase layout per wall: 7 autoregressive steps.
#   step+0: marker (THINKING_WALL_N)
#   step+1: HIT_FULL_ID
#   step+2: VALUE for HIT_FULL  ← thinking_value live here
#   step+3: HIT_X_ID
#   step+4: VALUE for HIT_X     ← live
#   step+5: HIT_Y_ID
#   step+6: VALUE for HIT_Y     ← live
_STEPS_PER_WALL = 7
_HF_OFFSET = 2
_HX_OFFSET = 4
_HY_OFFSET = 6


def _box_room_config():
    return RenderConfig(
        screen_width=64,
        screen_height=80,
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
                    module, state, inp, subset, config, textures=textures, trace=trace
                )
                cache[key] = (frame, new_state, trace, inp)
            return cache[key]

        return _run

    def _check_wall_hits(self, run, segs, px, py, angle, inp_kw):
        """Run a frame and verify per-wall HIT_*/HIT_X/HIT_Y match reference."""
        _, _, trace, inputs = run(px, py, angle, **inp_kw)
        vx, vy = _player_velocity(angle, inputs)

        log = trace.thinking_value_log
        assert len(log) >= len(segs) * _STEPS_PER_WALL, (
            f"Thinking value log too short ({len(log)} < "
            f"{len(segs) * _STEPS_PER_WALL}); thinking phase likely truncated."
        )

        mismatches = []
        for i, seg in enumerate(segs):
            ref_hf, ref_hx, ref_hy = _ref_hits(seg, px, py, vx, vy)
            base = i * _STEPS_PER_WALL
            raw_hf = log[base + _HF_OFFSET]
            raw_hx = log[base + _HX_OFFSET]
            raw_hy = log[base + _HY_OFFSET]
            comp_hf = round(raw_hf)
            comp_hx = round(raw_hx)
            comp_hy = round(raw_hy)
            for name, ref, comp, raw in [
                ("HIT_FULL", ref_hf, comp_hf, raw_hf),
                ("HIT_X", ref_hx, comp_hx, raw_hx),
                ("HIT_Y", ref_hy, comp_hy, raw_hy),
            ]:
                if ref != comp:
                    mismatches.append(
                        f"wall {i} {name}: ref={ref}, compiled={comp} (raw={raw:.3f})"
                    )
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
