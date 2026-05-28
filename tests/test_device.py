import pytest


@pytest.mark.usefixtures("has_device")
def test_device_operations(opencl_backend, device_count):
    from torch_opencl import _C

    # test_get_device_count
    # Verify device count is non-negative and consistent with C++.
    count = opencl_backend.device_count()
    assert count >= 0
    assert count == _C.get_device_count()

    # test_current_device_management
    # Test setting and getting the current active device.
    opencl_backend.set_device(0)
    assert opencl_backend.current_device() == 0

    # test_set_device_out_of_range
    # Verify that setting an invalid device index raises an exception.
    with pytest.raises(Exception):
        opencl_backend.set_device(9999)

    # test_device_context_manager
    # Test the 'with opencl.device(i):' context manager.
    initial = opencl_backend.current_device()

    # Using device 0 as a guaranteed target
    with opencl_backend.device(0):
        assert opencl_backend.current_device() == 0

    # Ensure it restores the previous device after exiting block
    assert opencl_backend.current_device() == initial

    # test_device_context_manager_nesting
    # Test that nested device contexts restore state in the correct order.
    if device_count >= 2:
        with opencl_backend.device(1):
            assert opencl_backend.current_device() == 1
            with opencl_backend.device(0):
                assert opencl_backend.current_device() == 0
            assert opencl_backend.current_device() == 1

    # test_lifecycle_apis
    # Test high-level initialization and status APIs.
    assert opencl_backend.is_initialized() is True

    # Calling init again should be safe and idempotent
    opencl_backend.init()
    assert opencl_backend.is_initialized() is True

    # test_is_in_bad_fork
    # Verify is_in_bad_fork binding runs and returns a boolean.
    res = _C.is_in_bad_fork()
    assert isinstance(res, bool)

    # test_exchange_device_binding
    # Test the low-level _C.exchange_device binding directly.
    initial_dev = _C.get_device()

    # Switch to 0 (always valid if has_device)
    prev = _C.exchange_device(0)
    assert prev == initial_dev
    assert _C.get_device() == 0

    # Restore the initial device
    _C.exchange_device(initial_dev)
    assert _C.get_device() == initial_dev
