import pytest


def test_get_device_count(opencl_backend):
    """Verify device count is non-negative and consistent with C++."""
    from torch_opencl import _C

    count = opencl_backend.device_count()
    assert count >= 0
    assert count == _C.get_device_count()


@pytest.mark.usefixtures("has_device")
def test_current_device_management(opencl_backend):
    """Test setting and getting the current active device."""
    # has_device ensures we have at least 1 device at index 0
    opencl_backend.set_device(0)
    assert opencl_backend.current_device() == 0


@pytest.mark.usefixtures("has_device")
def test_set_device_out_of_range(opencl_backend):
    """Verify that setting an invalid device index raises an exception."""
    with pytest.raises(Exception):
        opencl_backend.set_device(9999)


@pytest.mark.usefixtures("has_device")
def test_device_context_manager(opencl_backend):
    """Test the 'with opencl.device(i):' context manager."""
    initial = opencl_backend.current_device()

    # Using device 0 as a guaranteed target
    with opencl_backend.device(0):
        assert opencl_backend.current_device() == 0

    # Ensure it restores the previous device after exiting block
    assert opencl_backend.current_device() == initial


def test_device_context_manager_nesting(device_count, opencl_backend):
    """Test that nested device contexts restore state in the correct order."""
    if device_count < 2:
        pytest.skip("Test requires at least 2 OpenCL devices")

    with opencl_backend.device(1):
        assert opencl_backend.current_device() == 1
        with opencl_backend.device(0):
            assert opencl_backend.current_device() == 0
        assert opencl_backend.current_device() == 1


@pytest.mark.usefixtures("has_device")
def test_lifecycle_apis(opencl_backend):
    """Test high-level initialization and status APIs."""
    # Since we've already run tests, opencl should be initialized
    assert opencl_backend.is_initialized() is True

    # Calling init again should be safe and idempotent
    opencl_backend.init()
    assert opencl_backend.is_initialized() is True


def test_is_in_bad_fork():
    """Verify is_in_bad_fork binding runs and returns a boolean."""
    from torch_opencl import _C

    res = _C.is_in_bad_fork()
    assert isinstance(res, bool)


@pytest.mark.usefixtures("has_device")
def test_exchange_device_binding(opencl_backend):
    """Test the low-level _C.exchange_device binding directly."""
    from torch_opencl import _C

    initial = _C.get_device()

    # Switch to 0 (always valid if has_device)
    prev = _C.exchange_device(0)
    assert prev == initial
    assert _C.get_device() == 0

    # Restore the initial device
    _C.exchange_device(initial)
    assert _C.get_device() == initial
