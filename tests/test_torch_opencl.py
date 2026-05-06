import torch
import torch_opencl
import pyopencl as cl
import pytest

def test_device_count():
    # Get device count from torch-opencl
    count = torch_opencl.device_count()
    assert isinstance(count, int)
    assert count >= 0
    
    # Get device count from pyopencl for comparison
    platforms = cl.get_platforms()
    pyopencl_count = 0
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        pyopencl_count += len(devices)
    
    # Both should match as they both look for GPUs
    assert count == pyopencl_count
    print(f"Detected {count} OpenCL GPU device(s)")

def test_import():
    assert hasattr(torch_opencl, "device_count")
