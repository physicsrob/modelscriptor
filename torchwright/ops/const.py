"""Constants for the ops graph-building layer.

Boolean convention: throughout ops, boolean-valued nodes use
1.0 for true and -1.0 for false, with 0.0 as the decision threshold.
Functions like ``compare``, ``equals_vector``, ``bool_not``, etc.
all follow this convention.
"""

step_sharpness = 10.0
embedding_step_sharpness = 1.0  # For embedding-space ops (map_to_table, equals_vector).
# Lower than step_sharpness because the margin (1/speed) must
# absorb dot-product errors from approximate embeddings.
# Embedding norms are ~40 (self-dot ~1600), so even tiny
# Euclidean errors become large dot-product errors.
big_offset = 1000.0  # Needs to be larger than any element we route in a comparison
