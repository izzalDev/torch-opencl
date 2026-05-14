import torch


def test_opencl_empty():
    x = torch.empty(4, device="opencl")
    assert x.device.type == "opencl"
    assert x.numel() == 4
