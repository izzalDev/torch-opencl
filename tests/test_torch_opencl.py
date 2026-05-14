# tests/test_is_available.py
import pyopencl as cl
import torch_opencl


def _has_gpu() -> bool:
    """Cek ketersediaan GPU via pyopencl sebagai ground truth."""
    try:
        for platform in cl.get_platforms():
            devices = platform.get_devices(cl.device_type.GPU)
            if devices:
                return True
    except cl.Error:
        pass
    return False


def test_return_type():
    """is_available() harus mengembalikan bool."""
    result = torch_opencl.is_available()
    assert isinstance(result, bool)


def test_match_pyopencl():
    """Hasil is_available() harus konsisten dengan pyopencl."""
    expected = _has_gpu()
    result = torch_opencl.is_available()
    assert result == expected


def test_consistent():
    """is_available() harus memberikan hasil yang sama jika dipanggil berulang."""
    first = torch_opencl.is_available()
    second = torch_opencl.is_available()
    assert first == second


def test_device_count_type():
    assert isinstance(torch_opencl.device_count(), int)


def test_device_count_non_negative():
    assert torch_opencl.device_count() >= 0


def test_device_count_matches_availability():
    """Jika is_available() True, device_count() harus >= 1, dan sebaliknya."""
    assert torch_opencl.is_available() == (torch_opencl.device_count() > 0)
