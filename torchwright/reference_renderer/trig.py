import numpy as np


def generate_trig_table() -> np.ndarray:
    """Generate a 256-entry trig lookup table.

    Returns:
        np.ndarray of shape (256, 2) where:
            table[i, 0] = cos(2*pi*i/256)
            table[i, 1] = sin(2*pi*i/256)

    This table is the single source of truth for trig values.
    Both the reference renderer and the transformer graph must use
    identical values from this table.
    """
    angles = np.arange(256) * (2.0 * np.pi / 256.0)
    table = np.empty((256, 2), dtype=np.float64)
    table[:, 0] = np.cos(angles)
    table[:, 1] = np.sin(angles)
    return table
