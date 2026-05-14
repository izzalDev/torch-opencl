import torch
import torch_opencl._C

_initialized = False
_is_in_bad_fork = getattr(torch_opencl._C, "is_in_bad_fork", lambda: False)


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        idx = torch.accelerator._utils._get_device_index(device, optional=True)
        self.idx: int = idx if idx is not None else -1
        self.prev_idx = -1

    def __enter__(self):
        if self.idx >= 0:
            self.prev_idx = torch_opencl._C.exchange_device(self.idx)

    def __exit__(self, *_):
        if self.idx >= 0:
            self.idx = torch_opencl._C.exchange_device(self.prev_idx)
        return False


def is_available():
    return device_count() > 0


def device_count() -> int:
    return torch_opencl._C.get_device_count()


def current_device():
    return torch_opencl._C.get_device()


# LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
def set_device(device) -> None:
    if device >= 0:
        torch_opencl._C.set_device(device)


# LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION


def init():
    _lazy_init()


def is_initialized():
    return _initialized and not _is_in_bad_fork()


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    if _is_in_bad_fork():
        raise RuntimeError(
            "Cannot re-initialize OpenReg in forked subprocess. To use OpenReg with "
            "multiprocessing, you must use the 'spawn' start method"
        )
    torch_opencl._C.init()
    _initialized = True


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "init",
    "is_initialized",
]
