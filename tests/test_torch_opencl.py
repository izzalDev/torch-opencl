# tests/test_torch_opencl.py
"""Tests for torch_opencl package."""

import pyopencl as cl

import torch_opencl


def test_device_count():
    """device_count() should match PyOpenCL's GPU count across all platforms."""
    count = torch_opencl.device_count()
    assert isinstance(count, int)
    assert count >= 0

    pyopencl_count = 0
    for platform in cl.get_platforms():
        try:
            pyopencl_count += len(platform.get_devices(device_type=cl.device_type.GPU))
        except cl.LogicError:
            continue

    assert count == pyopencl_count, (
        f"torch_opencl reports {count} devices, but PyOpenCL found {pyopencl_count}"
    )


def test_has_device_count():
    """torch_opencl must expose device_count at the top-level namespace."""
    assert hasattr(torch_opencl, "device_count")
    assert callable(torch_opencl.device_count)
