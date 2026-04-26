"""Shared constants for the DOOM game graph.

Centralizes token type E8 codes, piecewise-linear breakpoint tables, and
miscellaneous magic numbers used across the per-stage files under
``torchwright.doom.stages``.  Everything here is pure data — no graph
construction — so importing from either ``game_graph.py`` or any stage
file is safe from circular-import issues.
"""

from torchwright.graph.spherical_codes import index_to_vector

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

# Texture E8 codes live in a dedicated 8-wide bypass slot (not in the
# embedding).  Texture i is identified by ``index_to_vector(TEX_E8_OFFSET + i)``.
# The offset sits after the 8-wide semantic token-type codes; the
# texture-bypass layout retains this convention even though token
# identity now lives in the W_EMBED vocab lookup rather than a separate
# overlaid carrier.
TEX_E8_OFFSET = 8


# ---------------------------------------------------------------------------
# Breakpoints for piecewise_linear / piecewise_linear_2d products
# ---------------------------------------------------------------------------

DIFF_BP = [
    -40,
    -30,
    -20,
    -15,
    -10,
    -7,
    -5,
    -3,
    -2,
    -1,
    -0.5,
    0,
    0.5,
    1,
    2,
    3,
    5,
    7,
    10,
    15,
    20,
    30,
    40,
]

# Breakpoints for normalized coord differences in the texture-column
# precomputes.  After ``normalize_coord`` each raw coord lands in
# [-1, +1]; the differences ``norm_w_ex = norm_sel_bx - norm_sel_ax``
# (and friends) span [-2, +2], with most realistic samples concentrated
# near 0.  Padding the grid out to ±2.5 absorbs float32 round-off at the
# extremes without inflating cell area near 0; the asymmetric sample
# density (denser near 0, sparser near ±2) keeps mid-cell error small
# where the texture-column path is most sensitive.
NORM_DIFF_BP = [
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.7,
    -0.5,
    -0.3,
    -0.2,
    -0.1,
    -0.05,
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
    1.5,
    2.0,
    2.5,
]
TRIG_BP = [-1, -0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 0.9, 1]
SQRT_BP = [0, 0.25, 1, 2, 4, 9, 16, 25, 36, 49, 64, 100, 225, 400, 900, 1600, 3200]
VEL_BP = [-0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 0.7]
ATAN_BP = [
    -20,
    -10,
    -5,
    -3,
    -2,
    -1.5,
    -1,
    -0.75,
    -0.5,
    -0.25,
    0,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    5,
    10,
    20,
]


# ---------------------------------------------------------------------------
# Misc magic numbers
# ---------------------------------------------------------------------------

BIG_DISTANCE = 1000.0
