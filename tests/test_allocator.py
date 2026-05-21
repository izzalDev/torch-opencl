import pytest
import torch


@pytest.mark.usefixtures("has_device")
def test_opencl_empty():
    x = torch.empty(4, device="opencl")
    assert x.device.type == "opencl"
    assert x.numel() == 4


@pytest.mark.usefixtures("has_device")
def test_zero_size_allocation():
    # Allocating an empty tensor with shape containing zero
    x = torch.empty(0, device="opencl")
    assert x.device.type == "opencl"
    assert x.numel() == 0
    assert x.shape == torch.Size([0])

    y = torch.empty((2, 0, 3), device="opencl")
    assert y.device.type == "opencl"
    assert y.numel() == 0
    assert y.shape == torch.Size([2, 0, 3])


@pytest.mark.usefixtures("has_device")
def test_allocator_dtypes():
    dtypes = [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ]
    for dtype in dtypes:
        x = torch.empty((2, 5), dtype=dtype, device="opencl")
        assert x.device.type == "opencl"
        assert x.dtype == dtype
        assert x.shape == torch.Size([2, 5])


@pytest.mark.usefixtures("has_device")
def test_pin_memory_raises():
    with pytest.raises(RuntimeError, match="Pin memory can only be on CPU"):
        torch.empty(5, pin_memory=True, device="opencl")


@pytest.mark.usefixtures("has_device")
def test_unsupported_layout_raises():
    with pytest.raises(Exception):
        torch.empty(5, layout=torch.sparse_coo, device="opencl")
