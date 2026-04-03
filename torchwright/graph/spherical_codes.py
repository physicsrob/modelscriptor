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


spherical_codes = 10.0 * load_spherical_codes("E8.8.1024.txt")


def get_spherical_codes(d: int, max_index: int = 1024):
    assert d == 8
    assert max_index == 1024, "Only support max_index = 1024"
    return spherical_codes


def index_to_vector(index: int, max_index: int = 1024):
    assert max_index == 1024, "Only support 1024 currently"
    return spherical_codes[index]
