import pytest
from typing import Any


def pytest_configure():
    """
    Initial configuration hook.
    Ensures torch_opencl is imported so PyTorch registration happens early.
    """
    try:
        # We use a dummy assignment to satisfy "not accessed" linters
        import torch

        _ = torch
    except ImportError:
        pass


@pytest.fixture(scope="session")
def opencl_backend() -> Any:
    """
    Provides the opencl module.
    Skips tests if the package isn't built/installed.
    """
    try:
        import torch.opencl as opencl  # pyright: ignore

        return opencl
    except ImportError:
        pytest.fail("torch_opencl is not installed. Please run 'pip install -e .'")


@pytest.fixture(scope="session")
def has_device(opencl_backend: Any) -> bool:
    """
    Checks for hardware availability.
    The parameter 'opencl_backend' is used to satisfy the 'not accessed' check.
    """
    if opencl_backend.device_count() <= 0:
        pytest.skip("No OpenCL compatible device found on this system.")
    return True


@pytest.fixture(autouse=True)
def reset_device_state(has_device: bool):
    """
    Resets device state.
    Using 'has_device' as a parameter ensures the gatekeeper is active.
    """
    # Silencing the 'unused' warning for Pyright explicitly
    _ = has_device

    yield

    try:
        import torch

        torch.opencl.set_device(0)  # pyright: ignore
    except Exception:
        pass


@pytest.fixture
def device_count(opencl_backend: Any) -> int:
    """Returns the number of available OpenCL devices."""
    return opencl_backend.device_count()
