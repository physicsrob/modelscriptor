import pytest
import torch

from modelscriptor.compiler.device import set_device


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="auto",
        help="Torch device for tests: 'cpu', 'cuda', or 'auto' (default: auto-detect)",
    )


@pytest.fixture(scope="session", autouse=True)
def device(request):
    """Auto-detect the best available torch device, or use --device override."""
    choice = request.config.getoption("--device")
    if choice == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(choice)

    set_device(dev)
    return dev
