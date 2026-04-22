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

TOKEN_INPUT = 0
TOKEN_WALL = 1
TOKEN_EOS = 2
TOKEN_SORTED_WALL = 3
TOKEN_RENDER = 4
TOKEN_TEX_COL = 5
TOKEN_BSP_NODE = 7
TOKEN_PLAYER_X = 240
TOKEN_PLAYER_Y = 241
TOKEN_PLAYER_ANGLE = 242

# Phase A M4: thinking-token vocabulary entries.
#
# Markers (one per wall, 8 total): TOKEN_THINKING_WALL_BASE..+7.
# Identifier tokens name the value the next step should compute.
# A single shared TOKEN_THINKING_VALUE token type tags all value steps;
# the value-step graph reads "most recent identifier" from the KV cache
# to decide what to compute and what identifier comes next.
TOKEN_THINKING_WALL_BASE = 250  # 250..257 for wall 0..7
TOKEN_HIT_FULL_ID = 258
TOKEN_HIT_X_ID = 259
TOKEN_HIT_Y_ID = 260
TOKEN_THINKING_VALUE = 261

E8_INPUT = index_to_vector(TOKEN_INPUT)
E8_WALL = index_to_vector(TOKEN_WALL)
E8_EOS = index_to_vector(TOKEN_EOS)
E8_SORTED_WALL = index_to_vector(TOKEN_SORTED_WALL)
E8_RENDER = index_to_vector(TOKEN_RENDER)
E8_TEX_COL = index_to_vector(TOKEN_TEX_COL)
E8_BSP_NODE = index_to_vector(TOKEN_BSP_NODE)
E8_PLAYER_X = index_to_vector(TOKEN_PLAYER_X)
E8_PLAYER_Y = index_to_vector(TOKEN_PLAYER_Y)
E8_PLAYER_ANGLE = index_to_vector(TOKEN_PLAYER_ANGLE)

E8_THINKING_WALL = [index_to_vector(TOKEN_THINKING_WALL_BASE + i) for i in range(8)]
E8_HIT_FULL_ID = index_to_vector(TOKEN_HIT_FULL_ID)
E8_HIT_X_ID = index_to_vector(TOKEN_HIT_X_ID)
E8_HIT_Y_ID = index_to_vector(TOKEN_HIT_Y_ID)
E8_THINKING_VALUE = index_to_vector(TOKEN_THINKING_VALUE)

# Texture E8 codes start at index 8 (after the 8 semantic token types),
# so texture i maps to ``index_to_vector(8 + i)``.
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
