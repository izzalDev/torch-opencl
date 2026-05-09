"""torch_opencl C extension module for PyTorch OpenCL backend."""

def is_in_bad_fork() -> bool:
    """
    Checks if the process has been forked in a way that is incompatible with the OpenCL runtime.
    """

def init() -> None:
    """Initializes the OpenCL runtime and discovers available devices."""

def get_device_count() -> int:
    """Returns the number of OpenCL-capable devices available on the system."""

def get_device() -> int:
    """Returns the index of the currently active OpenCL device."""

def set_device(device: int) -> None:
    """
    Sets the current OpenCL device.

    Args:
        device (int): The index of the device to make current.
    """

def exchange_device(device: int) -> int:
    """
    Exchanges the current OpenCL device with the specified one and returns the previous device index.
    """

def maybe_exchange_device(device: int) -> int:
    """
    Sets the device if it's different from the current one. Returns an optional containing the previous device index if changed.
    """
