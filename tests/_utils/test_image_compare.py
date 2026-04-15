"""Unit tests for tests._utils.image_compare.

Verifies the utility:
  - passes identical images
  - tolerates 1-pixel spatial shifts (rasterization jitter)
  - rejects larger shifts
  - rejects wrong-color regions (systematic color drift)
  - couples channels (won't match a pixel channel-by-channel
    across different neighbors)
  - reports useful diagnostic info on failure
"""

import numpy as np
import pytest

from tests._utils.image_compare import compare_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checker(h=8, w=8, a=(0.8, 0.8, 0.8), b=(0.3, 0.3, 0.3)):
    """2x2 checker pattern (H, W, 3)."""
    img = np.empty((h, w, 3), dtype=np.float64)
    a = np.asarray(a)
    b = np.asarray(b)
    for r in range(h):
        for c in range(w):
            img[r, c] = a if ((r // 2) + (c // 2)) % 2 == 0 else b
    return img


def _gradient(h=8, w=8):
    """Simple RGB gradient (H, W, 3)."""
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[..., 0] = np.linspace(0.0, 1.0, w)[None, :]
    img[..., 1] = np.linspace(0.0, 1.0, h)[:, None]
    img[..., 2] = 0.5
    return img


# ---------------------------------------------------------------------------
# Pass cases
# ---------------------------------------------------------------------------


def test_identical_images_match_perfectly():
    img = _checker()
    r = compare_images(img, img)
    assert r.matched_fraction == 1.0
    assert r.max_err == 0.0
    r.assert_matches()


def _shift_edge_pad(img: np.ndarray, *, dr: int = 0, dc: int = 0) -> np.ndarray:
    """Shift ``img`` by (dr, dc) and edge-pad the revealed border.

    np.roll wraps the opposite edge in, which creates artificial
    mismatches at the image boundary.  Edge-padding matches what a
    real rasterizer would do: missing data fills with the nearest
    existing column/row, so interior pixels shift by (dr, dc) and
    boundaries stay self-consistent.
    """
    h, w = img.shape[:2]
    ref_pad = np.pad(img, ((abs(dr), abs(dr)), (abs(dc), abs(dc)), (0, 0)),
                     mode="edge")
    r0 = abs(dr) - dr
    c0 = abs(dc) - dc
    return ref_pad[r0:r0 + h, c0:c0 + w].copy()


def test_one_pixel_horizontal_shift_tolerated():
    ref = _gradient(h=8, w=8)
    out = _shift_edge_pad(ref, dc=1)
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    assert r.matched_fraction == 1.0
    r.assert_matches()


def test_one_pixel_vertical_shift_tolerated():
    ref = _gradient(h=8, w=8)
    out = _shift_edge_pad(ref, dr=1)
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    assert r.matched_fraction == 1.0
    r.assert_matches()


def test_small_color_drift_within_tolerance():
    ref = _checker()
    out = ref + 0.02  # every pixel off by 0.02 < 0.05
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    assert r.matched_fraction == 1.0


# ---------------------------------------------------------------------------
# Fail cases — real bugs
# ---------------------------------------------------------------------------


def test_three_pixel_shift_rejected():
    # Gradient: each column differs by ~0.143 along R, so a 3-column
    # shift puts out[r,c] at 3*0.143 ≈ 0.43 away from any ref neighbor
    # within radius 1 (which can only reach ±0.143).
    ref = _gradient(h=8, w=8)
    out = _shift_edge_pad(ref, dc=3)
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    assert r.matched_fraction < 0.5
    with pytest.raises(AssertionError):
        r.assert_matches()


def test_wrong_color_blob_rejected():
    ref = _checker()
    out = ref.copy()
    # Paint a 3x3 pure-red blob — color not present anywhere in ref.
    out[2:5, 2:5] = (1.0, 0.0, 0.0)
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    assert r.matched_fraction < 1.0
    assert r.max_err >= 0.2
    with pytest.raises(AssertionError) as exc:
        r.assert_matches()
    # Diagnostic should point inside the blob region.
    msg = str(exc.value)
    assert "worst pixel" in msg
    assert "output" in msg and "ref" in msg


def test_systematic_color_drift_rejected():
    ref = _checker()
    out = ref + 0.2  # every pixel off by 0.2 > tolerance
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    # No matching neighbor for *any* pixel.
    assert r.matched_fraction < 0.5
    with pytest.raises(AssertionError):
        r.assert_matches()


def test_channels_are_coupled_not_independent():
    """A per-channel-independent filter would accept this mismatch.
    compare_images must reject it — R and B are "close" to different
    neighbors but no single neighbor matches the whole RGB tuple.
    """
    ref = np.zeros((4, 4, 3), dtype=np.float64)
    # Checkerboard where each cell has a distinct RGB signature.
    ref[0, 0] = (1.0, 0.0, 0.0)
    ref[0, 1] = (0.0, 1.0, 0.0)
    ref[0, 2] = (0.0, 0.0, 1.0)
    ref[0, 3] = (1.0, 1.0, 0.0)
    ref[1:] = ref[0]  # repeat row
    ref[2:] = ref[0]
    ref[3:] = ref[0]

    out = ref.copy()
    # Paint a pixel with a combination that doesn't exist anywhere:
    # R=1 (matches (0,0) or (0,3)), G=0 (matches (0,0) or (0,2)),
    # B=1 (matches (0,2) or (0,3)).  No single neighbor is (1,0,1).
    out[1, 1] = (1.0, 0.0, 1.0)
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    # That pixel must be unmatched.
    assert r.unmatched_mask[1, 1]
    assert r.max_err >= 0.9


# ---------------------------------------------------------------------------
# Edge behavior
# ---------------------------------------------------------------------------


def test_edge_padding_handles_boundary_pixels():
    ref = _gradient(h=4, w=4)
    out = ref.copy()
    # Top-left corner matches itself via any window position (edge pad
    # repeats edge).
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.0)
    assert r.matched_fraction == 1.0
    assert r.max_err == 0.0


def test_diagnostic_message_includes_neighborhoods():
    ref = _checker()
    out = ref.copy()
    out[4, 4] = (0.0, 0.0, 0.0)  # force a pure-black pixel
    r = compare_images(out, ref, spatial_radius=1, pixel_tolerance=0.05)
    with pytest.raises(AssertionError) as exc:
        r.assert_matches()
    msg = str(exc.value)
    assert "neighborhood" in msg
    assert "matched_fraction" in msg
    assert "max_err" in msg


def test_radius_zero_requires_exact_pixel_match():
    ref = _gradient(h=8, w=8)
    out = _shift_edge_pad(ref, dc=1)
    r = compare_images(out, ref, spatial_radius=0, pixel_tolerance=0.05)
    # With no spatial slack, even a 1-pixel shift fails: each pixel
    # differs from its position-aligned ref pixel by one gradient step
    # (~0.143) which exceeds the 0.05 tolerance.
    assert r.matched_fraction < 1.0


def test_shape_mismatch_raises():
    a = np.zeros((4, 4, 3))
    b = np.zeros((4, 5, 3))
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_images(a, b)
