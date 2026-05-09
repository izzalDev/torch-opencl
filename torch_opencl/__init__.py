import torch as _torch

from . import opencl
from .opencl import (
    current_device,
    device,
    device_count,
    init,
    is_available,
    is_initialized,
    set_device,
)


def _setup_backend():
    from torch.utils.backend_registration import (
        generate_methods_for_privateuse1_backend as _generate,
        rename_privateuse1_backend as _rename,
    )

    _rename("opencl")
    _torch._register_device_module("opencl", opencl)
    _generate(for_storage=True)


_setup_backend()

__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "init",
    "is_initialized",
]
