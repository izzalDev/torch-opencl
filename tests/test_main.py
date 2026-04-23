import torch_opencl
import torch

def test_hello_from_bin():
    print(torch.__version__)
    assert torch_opencl.add(1, 1) == 2
