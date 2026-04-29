from pathlib import Path

import torch


def load_spherical_codes(filename):
    with open(filename, "r") as f:
        # Read lines from the file
        lines = f.readlines()

        # Split each line into numbers and convert them to floats
        data = [[float(num) for num in line.split()] for line in lines]

        # Convert the data to a PyTorch tensor
        tensor = torch.tensor(data, dtype=torch.float32)

    return tensor


# Locate E8.8.1024.txt regardless of cwd. The repo-root path (parent.parent.parent
# from this file) covers workspace and standalone clones; the bare filename
# fallback covers Modal, where the file is staged at cwd via add_local_file.
_E8_FILENAME = "E8.8.1024.txt"
_E8_AT_SOURCE = Path(__file__).resolve().parent.parent.parent / _E8_FILENAME
spherical_codes = 10.0 * load_spherical_codes(
    str(_E8_AT_SOURCE) if _E8_AT_SOURCE.exists() else _E8_FILENAME
)


def get_spherical_codes(d: int, max_index: int = 1024):
    assert d == 8
    assert max_index == 1024, "Only support max_index = 1024"
    return spherical_codes


def index_to_vector(index: int, max_index: int = 1024):
    assert max_index == 1024, "Only support 1024 currently"
    return spherical_codes[index]
