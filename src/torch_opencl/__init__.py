# torch_opencl/__init__.py
import sys
import types
import torch
from torch_opencl._core import *  # ← re-export ke torch_opencl namespace
import torch_opencl._core as _core

# buat torch.opencl sebagai module buatan
_opencl_module = types.ModuleType("torch.opencl")
_opencl_module.is_available          = _core.is_available
_opencl_module.device_count          = _core.device_count
_opencl_module.set_device            = _core.set_device
_opencl_module.current_device        = _core.current_device
_opencl_module.synchronize           = _core.synchronize
_opencl_module.get_device_properties = _core.get_device_properties

# inject ke torch dan sys.modules
torch.opencl = _opencl_module
sys.modules["torch.opencl"] = _opencl_module
