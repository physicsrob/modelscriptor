import pytest
import torch

from torchwright.compiler.device import set_device


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


def pytest_sessionstart(session):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n[gpu] {props.name}, {props.total_memory / 2**30:.1f} GiB VRAM")


def pytest_sessionfinish(session, exitstatus):
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"\n[gpu] peak allocated: {peak:.2f} GiB, peak reserved: {reserved:.2f} GiB")
