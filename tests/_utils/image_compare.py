"""Image comparison with tolerance for sub-pixel spatial jitter.

Comparing the compiled graph's rasterizer against the reference renderer
with per-pixel ``abs(out - ref).max()`` produces false positives at every
discretization boundary: u=k/tex_w texel seams, wall-edge integer rows,
column seams at oblique angles.  Two *correct* rasterizers can place a
discontinuity on opposite sides of one pixel; the absolute difference is
the full texture contrast (0.5 on a checker pattern) but the image is
visually correct — the right color exists one pixel away.

``compare_images`` tolerates this by letting each output pixel match any
pixel in a small neighborhood of the reference, provided the full RGB
triple matches (channels coupled, not independent).  This correctly:

- **tolerates** 1-pixel texel shifts, wall-edge row jitter, column-seam
  disagreements — the right color exists nearby in the reference.
- **catches** wrong textures (the color doesn't exist anywhere nearby),
  structural shifts beyond the search radius (wall at wrong position),
  and systematic color drift (every pixel unmatched by a small amount).

**Do not widen ``spatial_radius`` or ``pixel_tolerance`` to make failing
tests pass.**  The default radius is 1 and tolerance 0.05; these were
chosen to bracket "unavoidable rasterization jitter" tightly.  A test
failing under this metric is signalling a real disagreement, not noise.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ImageCompareResult:
    """Outcome of comparing two images with spatial tolerance."""

    # (H, W) per-pixel L∞ distance to the best-matching neighbor in ref.
    best_match_dist: np.ndarray
    # (H, W) bool mask: True where best_match_dist > pixel_tolerance.
    unmatched_mask: np.ndarray
    # Fraction of pixels whose best-match distance is within tolerance.
    matched_fraction: float
    # Worst best-match distance across the entire image.
    max_err: float
    # (row, col) of the pixel with the worst best-match distance.
    worst_location: Tuple[int, int]
    # Original images (for diagnostic printing).
    output: np.ndarray
    reference: np.ndarray
    # Parameters used.
    spatial_radius: int
    pixel_tolerance: float

    def assert_matches(
        self,
        *,
        min_matched_fraction: float = 0.99,
        max_err: float = 0.15,
    ) -> None:
        """Raise AssertionError with rich diagnostics on failure."""
        ok_fraction = self.matched_fraction >= min_matched_fraction
        ok_max_err = self.max_err <= max_err
        if ok_fraction and ok_max_err:
            return

        r, c = self.worst_location
        R = self.spatial_radius
        H, W = self.best_match_dist.shape
        r0, r1 = max(0, r - R), min(H, r + R + 1)
        c0, c1 = max(0, c - R), min(W, c + R + 1)

        msg_lines = [
            f"image comparison failed (radius={R}, tolerance={self.pixel_tolerance})",
            f"  matched_fraction = {self.matched_fraction:.4f} "
            f"(threshold {min_matched_fraction:.4f})",
            f"  max_err          = {self.max_err:.4f} " f"(threshold {max_err:.4f})",
            f"  unmatched pixels = {int(self.unmatched_mask.sum())} / "
            f"{self.unmatched_mask.size}",
            f"  worst pixel      = (row={r}, col={c})",
            f"    output  {self._fmt_rgb(self.output[r, c])}",
            f"    ref     {self._fmt_rgb(self.reference[r, c])}",
            f"  output neighborhood [{r0}:{r1}, {c0}:{c1}]:",
            self._fmt_patch(self.output[r0:r1, c0:c1]),
            f"  reference neighborhood [{r0}:{r1}, {c0}:{c1}]:",
            self._fmt_patch(self.reference[r0:r1, c0:c1]),
        ]
        raise AssertionError("\n".join(msg_lines))

    @staticmethod
    def _fmt_rgb(rgb: np.ndarray) -> str:
        return "(" + ", ".join(f"{v:.3f}" for v in rgb) + ")"

    @classmethod
    def _fmt_patch(cls, patch: np.ndarray) -> str:
        # Print mean-across-channels (grayscale) 2-decimal grid — enough
        # to eyeball pattern disagreement without flooding output.
        gray = patch.mean(axis=-1)
        rows = ["    " + " ".join(f"{v:.2f}" for v in row) for row in gray]
        return "\n".join(rows)


def compare_images(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    spatial_radius: int = 1,
    pixel_tolerance: float = 0.05,
) -> ImageCompareResult:
    """Compare two (H, W, C) images tolerating small spatial jitter.

    For each output pixel ``out[r, c]``, search the ``(2R+1)×(2R+1)``
    neighborhood of ``reference`` centered at ``(r, c)`` for a pixel
    whose L∞ per-channel distance to ``out[r, c]`` is minimized.  That
    minimum is the pixel's "best-match distance".  A pixel is matched
    iff its best-match distance is within ``pixel_tolerance``.

    Edge pixels use edge-padded reference values so the search window
    is always well-defined.

    Channels are matched as a tuple: the full RGB triple of the output
    pixel must be close to the full RGB triple of *one* reference
    neighbor.  Per-channel independent matching (which would allow R to
    match one neighbor and B another) is explicitly avoided.

    Args:
        output: Compiled / candidate image, shape (H, W, C).
        reference: Ground-truth image, shape (H, W, C).
        spatial_radius: Radius R of the search window.  Default 1
            (=3×3 neighborhood).  Keep small — larger windows hide
            real misalignment bugs.
        pixel_tolerance: L∞ per-channel distance threshold for calling
            a pixel "matched".  Default 0.05.
    """
    if output.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: output {output.shape} vs reference {reference.shape}"
        )
    if output.ndim != 3:
        raise ValueError(f"expected (H, W, C), got shape {output.shape}")
    if spatial_radius < 0:
        raise ValueError(f"spatial_radius must be >= 0, got {spatial_radius}")

    H, W, _ = output.shape
    R = spatial_radius
    ref_pad = np.pad(reference, ((R, R), (R, R), (0, 0)), mode="edge")

    best = np.full((H, W), np.inf, dtype=np.float64)
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            shifted = ref_pad[R + dr : R + dr + H, R + dc : R + dc + W, :]
            # L∞ across channels: all channels of this neighbor must be
            # within `dist` of the output pixel for them to "match".
            dist = np.abs(output - shifted).max(axis=-1)
            np.minimum(best, dist, out=best)

    unmatched = best > pixel_tolerance
    matched_fraction = float(1.0 - unmatched.mean())
    max_err = float(best.max())
    worst_flat = int(np.argmax(best))
    worst_r, worst_c = int(worst_flat // W), int(worst_flat % W)

    return ImageCompareResult(
        best_match_dist=best,
        unmatched_mask=unmatched,
        matched_fraction=matched_fraction,
        max_err=max_err,
        worst_location=(worst_r, worst_c),
        output=output,
        reference=reference,
        spatial_radius=R,
        pixel_tolerance=pixel_tolerance,
    )
