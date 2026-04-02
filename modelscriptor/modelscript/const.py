turn_on_speed = 10.0
embedding_turn_on_speed = (
    1.0  # For embedding-space ops (map_to_table, compare_to_vector).
)
# Lower than turn_on_speed because the margin (1/speed) must
# absorb dot-product errors from approximate embeddings.
# Embedding norms are ~40 (self-dot ~1600), so even tiny
# Euclidean errors become large dot-product errors.
big_offset = 1000.0  # Needs to be larger than any element we route in a comparison
