# tests/test_torch_opencl.py
"""Tests for torch_opencl package."""

import pyopencl as cl
import pytest
import torch

import torch_opencl.opencl as opencl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pyopencl_gpu_count() -> int:
    """Return GPU device count via PyOpenCL, 0 if no runtime available."""
    try:
        count = 0
        for platform in cl.get_platforms():
            try:
                count += len(platform.get_devices(device_type=cl.device_type.GPU))
            except cl.LogicError:
                continue
        return count
    except cl.LogicError:
        return 0


HAS_DEVICE = opencl.is_available()
skip_no_device = pytest.mark.skipif(not HAS_DEVICE, reason="No OpenCL device available")


# ---------------------------------------------------------------------------
# device_count
# ---------------------------------------------------------------------------


def test_device_count_type():
    """device_count() must return an int."""
    assert isinstance(opencl.device_count(), int)


def test_device_count_non_negative():
    """device_count() must be >= 0."""
    assert opencl.device_count() >= 0


def test_device_count_matches_pyopencl():
    """device_count() must match PyOpenCL GPU count."""
    count = opencl.device_count()
    pyopencl_count = _pyopencl_gpu_count()
    assert count == pyopencl_count, (
        f"torch_opencl reports {count} devices, PyOpenCL found {pyopencl_count}"
    )


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


def test_is_available_type():
    """is_available() must return a bool."""
    assert isinstance(opencl.is_available(), bool)


def test_is_available_consistent_with_device_count():
    """is_available() must be True iff device_count() > 0."""
    assert opencl.is_available() == (opencl.device_count() > 0)


# ---------------------------------------------------------------------------
# init / is_initialized
# ---------------------------------------------------------------------------


def test_is_initialized_type():
    """is_initialized() must return a bool."""
    assert isinstance(opencl.is_initialized(), bool)


@skip_no_device
def test_init_sets_initialized():
    """init() must set is_initialized() to True."""
    opencl.init()
    assert opencl.is_initialized()


# ---------------------------------------------------------------------------
# current_device / set_device
# ---------------------------------------------------------------------------


@skip_no_device
def test_current_device_type():
    """current_device() must return an int."""
    assert isinstance(opencl.current_device(), int)


@skip_no_device
def test_current_device_in_range():
    """current_device() must be within [0, device_count())."""
    idx = opencl.current_device()
    assert 0 <= idx < opencl.device_count()


@skip_no_device
def test_set_device_changes_current():
    """set_device(i) must change current_device() to i."""
    original = opencl.current_device()
    try:
        for i in range(opencl.device_count()):
            opencl.set_device(i)
            assert opencl.current_device() == i
    finally:
        opencl.set_device(original)


@skip_no_device
def test_set_device_negative_is_noop():
    """set_device() with negative index must be a no-op."""
    original = opencl.current_device()
    opencl.set_device(-1)
    assert opencl.current_device() == original


@skip_no_device
def test_set_device_out_of_range():
    """set_device() with out-of-range index must raise."""
    with pytest.raises(RuntimeError):
        opencl.set_device(opencl.device_count())


# ---------------------------------------------------------------------------
# exchange_device
# ---------------------------------------------------------------------------


@skip_no_device
def test_exchange_device_returns_previous():
    """exchange_device(i) must return the previous device index."""
    original = opencl.current_device()
    target = (original + 1) % opencl.device_count()
    try:
        prev = opencl.exchange_device(target)
        assert prev == original
        assert opencl.current_device() == target
    finally:
        opencl.set_device(original)


@skip_no_device
def test_exchange_device_negative_returns_minus_one():
    """exchange_device(-1) must return -1 and not change current device."""
    original = opencl.current_device()
    result = opencl.exchange_device(-1)
    assert result == -1
    assert opencl.current_device() == original


# ---------------------------------------------------------------------------
# maybe_exchange_device
# ---------------------------------------------------------------------------


@skip_no_device
def test_maybe_exchange_device_same_device():
    """maybe_exchange_device(current) must not change device."""
    current = opencl.current_device()
    prev = opencl.maybe_exchange_device(current)
    assert prev == current
    assert opencl.current_device() == current


@skip_no_device
def test_maybe_exchange_device_different_device():
    """maybe_exchange_device(other) must switch to other and return previous."""
    original = opencl.current_device()
    target = (original + 1) % opencl.device_count()
    try:
        prev = opencl.maybe_exchange_device(target)
        assert prev == original
        assert opencl.current_device() == target
    finally:
        opencl.set_device(original)


@skip_no_device
def test_maybe_exchange_device_negative_returns_minus_one():
    """maybe_exchange_device(-1) must return -1 and not change current device."""
    original = opencl.current_device()
    result = opencl.maybe_exchange_device(-1)
    assert result == -1
    assert opencl.current_device() == original


# ---------------------------------------------------------------------------
# device context manager
# ---------------------------------------------------------------------------


@skip_no_device
def test_device_context_manager():
    """opencl.device() context manager must restore previous device on exit."""
    original = opencl.current_device()
    target = (original + 1) % opencl.device_count()
    with opencl.Device(target):
        assert opencl.current_device() == target
    assert opencl.current_device() == original


@skip_no_device
def test_device_context_manager_restores_on_exception():
    """opencl.device() must restore device even if an exception is raised."""
    original = opencl.current_device()
    target = (original + 1) % opencl.device_count()
    try:
        with opencl.Device(target):
            assert opencl.current_device() == target
            raise RuntimeError("test exception")
    except RuntimeError:
        pass
    assert opencl.current_device() == original


# ---------------------------------------------------------------------------
# torch integration
# ---------------------------------------------------------------------------


def test_backend_registered():
    """opencl backend must be registered in torch."""
    assert hasattr(torch, "opencl")


def test_torch_opencl_is_available():
    """torch.opencl.is_available() must match torch_opencl.opencl.is_available()."""
    assert torch.opencl.is_available() == opencl.is_available()


def test_torch_opencl_device_count():
    """torch.opencl.device_count() must match torch_opencl.opencl.device_count()."""
    assert torch.opencl.device_count() == opencl.device_count()
