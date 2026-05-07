# tests/test_torch_opencl.py
"""Tests for torch_opencl package."""

import pyopencl as cl
import pytest

import torch_opencl


def get_pyopencl_gpu_count() -> int:
    """Return total GPU device count from all OpenCL platforms."""
    count = 0

    try:
        platforms = cl.get_platforms()
    except cl.LogicError:
        return 0

    for platform in platforms:
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            count += len(devices)
        except cl.LogicError:
            continue

    return count


def test_device_count():
    """torch_opencl device count should match PyOpenCL GPU count."""
    count = torch_opencl._C._get_device_count()

    assert isinstance(count, int)
    assert count >= 0

    pyopencl_count = get_pyopencl_gpu_count()

    assert count == pyopencl_count, (
        f"torch_opencl reports {count} devices, but PyOpenCL found {pyopencl_count}"
    )


def test_has_device_count():
    """torch_opencl should expose _get_device_count."""
    assert hasattr(torch_opencl, "_C")
    assert hasattr(torch_opencl._C, "_get_device_count")
    assert callable(torch_opencl._C._get_device_count)


@pytest.mark.parametrize("call_count", range(3))
def test_device_count_is_stable(call_count):
    """Repeated calls should return the same value."""
    first = torch_opencl._C._get_device_count()
    second = torch_opencl._C._get_device_count()

    assert first == second
