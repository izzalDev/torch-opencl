"""Public API for torch_opencl — mirrors torch.cuda interface."""

import torch

import torch_opencl._C  # type: ignore[misc]

_initialized = False
_is_in_bad_fork = getattr(torch_opencl._C, "_is_in_bad_fork", lambda: False)


class Device:
    r"""Context-manager that changes the selected OpenCL device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_opencl._C._exchange_device(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = torch_opencl._C._set_device(self.prev_idx)
        return False


def is_available() -> bool:
    """Return True if OpenCL devices are available."""
    return device_count() > 0


def device_count() -> int:
    """Return the number of available OpenCL GPU devices."""
    return torch_opencl._C._get_device_count()


def current_device() -> int:
    """Return the index of the current OpenCL device."""
    _lazy_init()
    return torch_opencl._C._get_device()


def set_device(device) -> None:
    """Set the current OpenCL device.

    Args:
        device: index of the device to set as current.
    """
    if device >= 0:
        torch_opencl._C._set_device(device)


def exchange_device(device) -> int:
    """Set the current device and return the previous one."""
    _lazy_init()
    return torch_opencl._C._exchange_device(device)


def maybe_exchange_device(device) -> int:
    """Set the current device only if different, return previous."""
    _lazy_init()
    return torch_opencl._C._maybe_exchange_device(device)


def init() -> None:
    """Explicitly initialize OpenCL. Usually unnecessary."""
    _lazy_init()


def is_initialized() -> bool:
    """Return True if OpenCL has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_init() -> None:
    global _initialized
    if is_initialized():
        return
    if _is_in_bad_fork():
        raise RuntimeError(
            "Cannot re-initialize OpenCL in forked subprocess. To use OpenCL with "
            "multiprocessing, you must use the 'spawn' start method."
        )
    torch_opencl._C._init()
    _initialized = True


__all__ = [
    "Device",
    "device_count",
    "current_device",
    "set_device",
    "exchange_device",
    "maybe_exchange_device",
    "is_available",
    "init",
    "is_initialized",
]
