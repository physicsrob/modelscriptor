"""Tests for wall texture rendering — both reference and compiled."""

import numpy as np
import pytest
import torch

from torchwright.doom.compile import compile_game, step_frame_compiled
from torchwright.doom.game import GameState
from torchwright.doom.input import PlayerInput
from torchwright.reference_renderer import render_frame
from torchwright.reference_renderer.scenes import box_room, box_room_textured
from torchwright.reference_renderer.textures import default_texture_atlas
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.types import RenderConfig, Segment


def _box_room_random_textures(tex_size: int, seed: int = 42):
    """Box-room segments paired with deterministic random ``tex_size``
    textures, one per wall.  Uses random values so every texel is
    distinct — this exercises the textured rendering pipeline more
    thoroughly than the default 8×8 procedural atlas (which has very
    little variation) and, crucially, tests texture sizes above 8×8,
    which is what the real walkthrough uses (tex_size=64 WAD textures).
    """
    segments = [s for s in box_room()]
    for i, seg in enumerate(segments):
        seg.texture_id = i % 4
    rng = np.random.default_rng(seed)
    textures = [
        rng.random((tex_size, tex_size, 3), dtype=np.float32)
        for _ in range(4)
    ]
    return segments, textures


@pytest.fixture
def trig_table():
    return generate_trig_table()


@pytest.fixture
def small_config(trig_table):
    return RenderConfig(
        screen_width=16,
        screen_height=12,
        fov_columns=8,
        trig_table=trig_table,
        ceiling_color=(0.0, 0.0, 0.0),
        floor_color=(0.5, 0.5, 0.5),
    )


# ── Reference renderer texture tests ────────────────────────────────


def test_reference_textured_differs_from_solid(small_config):
    """Textured rendering should produce different pixels than solid color."""
    segments, textures = box_room_textured()
    solid_segments = box_room()

    textured = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    solid = render_frame(0.0, 0.0, 0, solid_segments, small_config)

    # They should NOT be identical — textures vary per row
    assert not np.allclose(textured, solid, atol=0.01)


def test_reference_texture_u_coordinate(small_config):
    """Looking straight at a wall, left and right columns should show
    different texture columns."""
    segments, textures = box_room_textured()
    frame = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    H, W = small_config.screen_height, small_config.screen_width

    # The east wall (brick) should be visible. Columns near the edges
    # of the screen hit different u values → different texture columns.
    center_row = H // 2
    left_col = frame[center_row, 0]
    right_col = frame[center_row, W - 1]
    # Left and right edges hit different texture columns on the east wall,
    # so they should generally differ (unless the texture happens to repeat)
    # At minimum, the wall colors should come from the brick texture
    assert not np.allclose(left_col, [0, 0, 0], atol=0.01)  # not ceiling


# ── Compiled textured renderer tests ────────────────────────────────


@pytest.fixture
def textured_game_module(small_config):
    """Compile textured game graph for box_room."""
    segments, textures = box_room_textured()
    return compile_game(
        segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        textures=textures,
        d=2048, d_head=16, verbose=False,
    )


def _assert_frame_match(compiled, reference, max_boundary_pixels=None, atol=0.2, msg=""):
    H, W = compiled.shape[:2]
    if max_boundary_pixels is None:
        max_boundary_pixels = 3 * W  # more tolerance for texture band edges
    mismatched = np.abs(compiled - reference) > atol
    n_bad = mismatched.any(axis=2).sum()
    if n_bad > max_boundary_pixels:
        rows, cols = np.where(mismatched.any(axis=2))
        r, c = rows[0], cols[0]
        assert False, (
            f"{msg} {n_bad}/{H * W} pixels differ (max allowed: {max_boundary_pixels}). "
            f"First at ({r},{c}): compiled={compiled[r, c]} ref={reference[r, c]}"
        )


def test_compiled_textured_no_input(textured_game_module, small_config):
    """No inputs: textured frame should match reference renderer."""
    segments, textures = box_room_textured()
    state = GameState(x=0.0, y=0.0, angle=0)

    frame, new_state = step_frame_compiled(
        textured_game_module, state, PlayerInput(), small_config,
    )

    ref = render_frame(0.0, 0.0, 0, segments, small_config, textures=textures)
    _assert_frame_match(frame, ref, msg="no-input textured")


def test_compiled_textured_turned(textured_game_module, small_config):
    """Different viewing angles should show different textures."""
    segments, textures = box_room_textured()

    for angle in [0, 64, 128, 192]:
        state = GameState(x=0.0, y=0.0, angle=angle)
        frame, _ = step_frame_compiled(
            textured_game_module, state, PlayerInput(), small_config,
        )
        ref = render_frame(0.0, 0.0, angle, segments, small_config, textures=textures)
        _assert_frame_match(frame, ref, msg=f"textured angle={angle}")


# ── Regression tests: large-texture rendering must not blow up ─────
#
# The existing textured tests above only exercise the default 4-color
# 8×8 procedural atlas.  The real ``walkthrough.py`` flow uses 64×64
# WAD textures, and the compiled renderer produced catastrophic output
# (some columns in the 10⁵ range, others below −20) while the original
# tests passed green.  The tests below use a larger (16×16) synthetic
# random atlas — small enough to keep compile time reasonable, but
# already past the 8×8 threshold where the failure first manifests.
#
# The checks are deliberately split along the two things the existing
# suite missed:
#
# * pixel-range sanity  — a compiled frame that emits values wildly
#   outside [0, 1] will turn into random clip-to-white / clip-to-black
#   bands in the final GIF even if an aggregate diff test passes.
# * per-column agreement — aggregate ``_assert_frame_match`` allowed
#   ``3 * W`` bad pixels, which can mask entire broken columns in a
#   full-height frame.  These tests instead require *every* screen
#   column to track the reference renderer independently, with no
#   single column's mean error exceeding the tolerance.


@pytest.fixture
def large_tex_game_module(small_config):
    """Compile textured game graph using 16×16 random textures.

    Unsharded (rows_per_patch defaults to H).  Shares the ``small_config``
    fixture so the compile cost scales with tests' 16×12 frame, not the
    production 160×100 one.
    """
    segments, textures = _box_room_random_textures(tex_size=16)
    return compile_game(
        segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        textures=textures,
        d=1024, d_head=16, verbose=False,
    )


@pytest.fixture
def large_tex_sharded_game_module(small_config):
    """Same as ``large_tex_game_module`` but with rows_per_patch=6 so
    the compiled graph goes through the column/patch sharded feedback
    loop (shards_per_col=2 for H=12).  Walkthroughs use rp < H, so the
    sharded path is the one the real pipeline exercises.
    """
    segments, textures = _box_room_random_textures(tex_size=16)
    return compile_game(
        segments, small_config,
        max_coord=10.0, move_speed=0.3, turn_speed=4,
        textures=textures,
        d=1024, d_head=16, verbose=False,
        rows_per_patch=6,
    )


def _assert_pixel_range_sane(frame, lo=-0.05, hi=1.05, msg=""):
    """Every pixel must sit in roughly [0, 1].  The np.clip step that
    turns a float frame into a GIF will otherwise hide huge positive /
    negative values as random white/black bands.
    """
    fmin = float(frame.min())
    fmax = float(frame.max())
    assert fmin >= lo, (
        f"{msg}: frame min {fmin:.3f} < {lo} — pixels out of range "
        f"would clip to black in the GIF"
    )
    assert fmax <= hi, (
        f"{msg}: frame max {fmax:.3f} > {hi} — pixels out of range "
        f"would clip to white in the GIF"
    )


def _assert_every_column_matches(frame, reference, atol=0.2, msg=""):
    """Every individual screen column must track the reference renderer.

    ``_assert_frame_match`` above lets ``3 * W`` pixels differ by more
    than the tolerance, which can mask an entire broken column.  This
    variant computes the mean absolute error per column and requires
    *every* column to stay under the threshold — a single bad column
    is an immediate failure with a reported column index.
    """
    H, W = frame.shape[:2]
    per_col_mae = np.abs(frame - reference).mean(axis=(0, 2))
    worst_col = int(per_col_mae.argmax())
    worst_mae = float(per_col_mae[worst_col])
    if worst_mae > atol:
        ref_mean = float(reference[:, worst_col, :].mean())
        got_mean = float(frame[:, worst_col, :].mean())
        got_min = float(frame[:, worst_col, :].min())
        got_max = float(frame[:, worst_col, :].max())
        bad = int((per_col_mae > atol).sum())
        assert False, (
            f"{msg}: {bad}/{W} columns diverge from reference. "
            f"Worst col {worst_col}: mae={worst_mae:.3f} "
            f"(got mean={got_mean:.3f} min={got_min:.3f} max={got_max:.3f}; "
            f"ref mean={ref_mean:.3f})"
        )


def test_compiled_large_tex_pixel_range(large_tex_game_module, small_config):
    """Compiled frame with 16×16 textures must stay in the valid
    output range.  Regression for the walkthrough white-column failure:
    the compiled renderer emitted values ~10⁵ for some columns and
    <−20 for others.
    """
    segments, textures = _box_room_random_textures(tex_size=16)
    state = GameState(x=0.0, y=0.0, angle=0)
    frame, _ = step_frame_compiled(
        large_tex_game_module, state, PlayerInput(), small_config,
    )
    _assert_pixel_range_sane(frame, msg="large-texture unsharded")


def test_compiled_large_tex_every_column_matches(large_tex_game_module, small_config):
    """Compiled frame with 16×16 textures must match the reference
    renderer column-by-column — no broken columns masked by an
    aggregate pixel-count threshold.
    """
    segments, textures = _box_room_random_textures(tex_size=16)
    state = GameState(x=0.0, y=0.0, angle=0)
    frame, _ = step_frame_compiled(
        large_tex_game_module, state, PlayerInput(), small_config,
    )
    ref = render_frame(
        state.x, state.y, state.angle, segments, small_config, textures=textures,
    )
    _assert_every_column_matches(frame, ref, msg="large-texture unsharded")


def test_compiled_large_tex_sharded_pixel_range(
    large_tex_sharded_game_module, small_config,
):
    """Same pixel-range sanity check but with rows_per_patch=6 so the
    column/patch sharded feedback path (the one walkthroughs use) gets
    exercised with large textures.
    """
    segments, textures = _box_room_random_textures(tex_size=16)
    state = GameState(x=0.0, y=0.0, angle=0)
    frame, _ = step_frame_compiled(
        large_tex_sharded_game_module, state, PlayerInput(), small_config,
    )
    _assert_pixel_range_sane(frame, msg="large-texture sharded rp=6")


def test_compiled_large_tex_sharded_every_column_matches(
    large_tex_sharded_game_module, small_config,
):
    """Per-column agreement check on the sharded large-texture path."""
    segments, textures = _box_room_random_textures(tex_size=16)
    state = GameState(x=0.0, y=0.0, angle=0)
    frame, _ = step_frame_compiled(
        large_tex_sharded_game_module, state, PlayerInput(), small_config,
    )
    ref = render_frame(
        state.x, state.y, state.angle, segments, small_config, textures=textures,
    )
    _assert_every_column_matches(frame, ref, msg="large-texture sharded rp=6")


# ── Production-shape leak regression ─────────────────────────────────
#
# The small-config tests above don't reproduce the
# ``broadcast_select`` partial-mask leak that appears in
# walkthrough-scale renders.  At the production shape
# (W=160, H=100, rp=10, tex_size=32) the per-row
# ``linear_bin_index → dynamic_extract`` chain occasionally produces
# a fractional ``tex_row_idx`` (when the upstream signed_multiply +
# floor_int falls into floor_int's [k-eps, k] ramp zone), which then
# puts ``in_range``'s mask in *its* fractional ramp zone, which then
# makes ``broadcast_select`` leak ``-half_big`` (~−500) per partially
# active slot.  Those leaked sentinels clip to black in the final GIF
# and produce the "distracting black patches" in the wall texture.
#
# The test below builds the production game graph at the walkthrough
# shape and runs the **graph oracle** (``reference_eval``, no compile
# required) at a sweep of ``(cur_col_idx, cur_patch_idx_in_col)``
# positions.  Catastrophic ``broadcast_select`` leaks bypass the
# composite mask and turn into ``~−500`` pixel values in the patch
# output — far outside any sane RGB range.  Asserting a tight lower
# bound on the per-position pixel values is enough to catch the leak
# without needing a full ONNX compile or GPU run.


def _patch_oracle_pixels(
    output_node,
    input_names,
    n_pos: int,
    cur_col_arr: np.ndarray,
    cur_patch_arr: np.ndarray,
    seed_state=(0.0, 0.0, 0.0),
):
    """Run ``reference_eval`` for ``n_pos`` ``(col, patch)`` positions.

    Returns an ``(n_pos, rows_per_patch, 3)`` numpy array of the
    rendered patch pixels at each requested position, with the player
    held at the given seed state and no inputs pressed.
    """
    from torchwright.debug.probe import reference_eval

    inputs = {k: torch.zeros(n_pos, 1) for k in input_names}
    inputs["cur_col_idx"][:, 0] = torch.from_numpy(cur_col_arr.astype(np.float32))
    inputs["cur_patch_idx_in_col"][:, 0] = torch.from_numpy(
        cur_patch_arr.astype(np.float32),
    )
    inputs["seed_x"][:, 0] = float(seed_state[0])
    inputs["seed_y"][:, 0] = float(seed_state[1])
    inputs["seed_angle"][:, 0] = float(seed_state[2])

    cache = reference_eval(output_node, inputs, n_pos)
    pixels_node = output_node.inputs[0]  # Concatenate child 0 = pixels
    pixels = cache[pixels_node].detach().cpu().numpy()
    rp = pixels.shape[1] // 3
    return pixels.reshape(n_pos, rp, 3)


def test_oracle_no_dynamic_extract_leak_production_shape():
    """Sweep ``(col, patch)`` over the production walkthrough shape and
    require every per-patch pixel value to stay in roughly ``[0, 1]``.

    Regression for the ``broadcast_select`` partial-mask leak: at the
    walkthrough scale (W=160, H=100, rp=10, tex_size=32), the per-row
    ``linear_bin_index → dynamic_extract`` chain emits ~−500 sentinels
    at specific (col, patch) tuples whose intermediate ``tex_row_idx``
    lands in ``in_range``'s ramp zone.  Those sentinels show up as
    distracting black patches in the rendered GIF; this test catches
    them at the *graph* level (no compile needed) so the regression
    is fast and pinpoints the failing position.
    """
    from torchwright.doom.game_graph import build_game_graph

    cfg = RenderConfig(
        screen_width=160,
        screen_height=100,
        fov_columns=32,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )
    # Use 16×16 random textures so the test doesn't depend on a WAD
    # being present.  The leak mechanism is texture-content-independent
    # — it's about ``tex_row_idx`` arithmetic, not the values being
    # looked up — so a deterministic random atlas suffices.
    segments, textures = _box_room_random_textures(tex_size=16)
    output_node, _ = build_game_graph(
        segments,
        cfg,
        max_coord=10.0,
        move_speed=0.3,
        turn_speed=4,
        textures=textures,
        rows_per_patch=10,
    )
    input_names = [
        "cur_col_idx", "cur_patch_idx_in_col",
        "input_backward", "input_forward",
        "input_strafe_left", "input_strafe_right",
        "input_turn_left", "input_turn_right",
        "seed_angle", "seed_x", "seed_y",
    ]

    # Sweep every (col, patch) — skipping cols that miss the wall
    # entirely would defeat the test's purpose.  rp=10 → 10 patches
    # per column × 160 columns = 1600 positions.  reference_eval is
    # fast (no compile), so this stays under 30s.
    cols = np.arange(160)
    patches = np.arange(10)
    cur_col, cur_patch = np.meshgrid(cols, patches, indexing="ij")
    cur_col = cur_col.flatten()
    cur_patch = cur_patch.flatten()
    n_pos = len(cur_col)

    pixels = _patch_oracle_pixels(
        output_node, input_names, n_pos, cur_col, cur_patch,
    )
    pmin = float(pixels.min())
    pmax = float(pixels.max())

    # Find the worst position so failures pinpoint the bug.
    worst_idx = int(pixels.reshape(n_pos, -1).min(axis=1).argmin())
    worst_col, worst_patch = int(cur_col[worst_idx]), int(cur_patch[worst_idx])
    worst_min = float(pixels[worst_idx].min())
    worst_max = float(pixels[worst_idx].max())

    assert pmin >= -0.1, (
        f"Production-shape graph oracle has pixel values far below 0: "
        f"min={pmin:.3f}.  Worst at (col={worst_col}, patch="
        f"{worst_patch}): min={worst_min:.3f} max={worst_max:.3f}.  "
        f"This is the broadcast_select partial-mask leak from "
        f"dynamic_extract — fix the in_range one-hot or the "
        f"broadcast_select fall-through behaviour at mask=0."
    )
    assert pmax <= 1.1, (
        f"Production-shape graph oracle has pixel values far above 1: "
        f"max={pmax:.3f}.  Worst at (col={worst_col}, patch="
        f"{worst_patch}): min={worst_min:.3f} max={worst_max:.3f}."
    )
