"""torch_opencl — OpenCL backend for PyTorch."""

import torch

import torch_opencl._C  # type: ignore[misc]
import torch_opencl.opencl

torch.utils.rename_privateuse1_backend("opencl")
torch._register_device_module("opencl", torch_opencl.opencl)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


def _autoload():
    # Entry point placeholder for auto-loading the backend.
    pass
