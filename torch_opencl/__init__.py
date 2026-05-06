"""torch_opencl — OpenCL backend for PyTorch."""

import torch  # noqa: F401  # preload libtorch shared libraries before _C

from torch_opencl._C import *  # type: ignore[import]
from torch_opencl._C import __doc__  # re-export module docstring
