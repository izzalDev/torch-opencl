import torch
import pytest


@pytest.mark.usefixtures("has_device", "opencl_backend")
def test_registration():
    # test_torch_opencl_attribute_exists
    # Verify that importing torch_opencl patches the torch module.
    assert hasattr(torch, "opencl")

    # test_backend_methods_are_callable
    # Verify that the primary backend methods are exposed correctly.
    assert callable(torch.opencl.is_available)  # type: ignore
    assert callable(torch.opencl.device_count)  # type: ignore

    # test_torch_device_integration
    # Verify that PyTorch's native device class accepts 'opencl'.
    from torch import device

    d = device("opencl")
    assert d.type == "opencl"

    d_indexed = device("opencl", 0)
    assert d_indexed.index == 0

    # test_is_available_logic
    # Check that is_available consistency holds true.
    assert torch.opencl.is_available() == (torch.opencl.device_count() > 0)  # type: ignore
