import torch
import pytest


@pytest.mark.usefixtures("has_device")
def test_empty_strided():
    # 1-D tensor with stride [1].
    t = torch.empty_strided((4,), (1,), device="opencl")
    assert t.device.type == "opencl"
    assert t.shape == torch.Size([4])
    assert t.stride() == (1,)

    # Row-major 2-D: strides match contiguous layout.
    t = torch.empty_strided((3, 4), (4, 1), device="opencl")
    assert t.shape == torch.Size([3, 4])
    assert t.stride() == (4, 1)
    assert t.is_contiguous()

    # Column-major (Fortran order) strides.
    t = torch.empty_strided((3, 4), (1, 3), device="opencl")
    assert t.stride() == (1, 3)
    assert not t.is_contiguous()

    # 0-D tensor (scalar): empty size and stride.
    t = torch.empty_strided((), (), device="opencl")
    assert t.shape == torch.Size([])
    assert t.ndim == 0

    # Tensor with a zero-sized dimension allocates no memory.
    t = torch.empty_strided((0, 4), (4, 1), device="opencl")
    assert t.numel() == 0

    # dtype float64
    t = torch.empty_strided((2, 2), (2, 1), dtype=torch.float64, device="opencl")
    assert t.dtype == torch.float64

    # dtype int32
    t = torch.empty_strided((5,), (1,), dtype=torch.int32, device="opencl")
    assert t.dtype == torch.int32

    # mismatched size stride raises
    with pytest.raises(RuntimeError, match="must match dimensionality"):
        torch.empty_strided((3, 4), (1,), device="opencl")

    # non-strided layout raises
    with pytest.raises(Exception):
        torch.empty_strided((4,), (1,), layout=torch.sparse_coo, device="opencl")


@pytest.mark.usefixtures("has_device")
def test_copy_from():
    # (A) CPU → OpenCL: data must be preserved.
    cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cl = cpu.to("opencl")
    result = cl.cpu()
    assert cl.device.type == "opencl"
    assert torch.allclose(cpu, result)

    # (B) OpenCL → CPU: data must survive the round-trip.
    original = torch.arange(8, dtype=torch.float32)
    back = original.to("opencl").cpu()
    assert back.device.type == "cpu"
    assert torch.allclose(original, back)

    # (C) OpenCL → OpenCL same device: values must match.
    src = torch.ones(4, dtype=torch.float32).to("opencl")
    dst = src.clone()
    assert torch.allclose(src.cpu(), dst.cpu())

    # 2-D tensor round-trip preserves shape and values.
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    back = cpu.to("opencl").cpu()
    assert back.shape == cpu.shape
    assert torch.allclose(cpu, back)

    # roundtrip dtype int32
    cpu = torch.tensor([10, 20, 30], dtype=torch.int32)
    back = cpu.to("opencl").cpu()
    assert (cpu == back).all()

    # peer to peer
    if torch.opencl.device_count() >= 2:  # type: ignore[attr-defined]
        src = torch.ones(4).to("opencl:0")
        with pytest.raises(
            RuntimeError,
            match="_copy_from: copy between OpenCL devices is not supported yet ",
        ):
            src.to("opencl:1")

    # copy mismatched numel raises
    src = torch.ones(4).to("opencl")
    dst = torch.zeros(5).to("opencl")
    with pytest.raises(RuntimeError, match="numel mismatch"):
        torch.ops.aten._copy_from(src, dst)

    # copy non-contiguous raises
    src = torch.empty_strided((2, 2), (1, 2), device="opencl")
    dst = torch.empty((2, 2), device="opencl")
    assert not src.is_contiguous()
    with pytest.raises(RuntimeError, match="src must be contiguous"):
        torch.ops.aten._copy_from(src, dst)


@pytest.mark.usefixtures("has_device")
def test_as_strided():
    # Verify basic as_strided usage and storage sharing.
    base = torch.arange(8, dtype=torch.float32).to("opencl")
    view = torch.as_strided(base, (3,), (2,), 1)
    assert view.device.type == "opencl"
    assert view.shape == torch.Size([3])
    assert view.stride() == (2,)
    assert view.storage_offset() == 1
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

    # Verify that negative strides are rejected.
    base = torch.empty(4, device="opencl")
    with pytest.raises(RuntimeError, match="negative stride not supported"):
        torch.as_strided(base, (2,), (-1,), 0)

    # Verify size and stride dimensionality mismatch raises.
    base = torch.empty(4, device="opencl")
    with pytest.raises(RuntimeError, match="must have same length"):
        torch.as_strided(base, (2, 2), (1,), 0)

    # Verify negative storage offset raises.
    base = torch.empty(4, device="opencl")
    with pytest.raises(RuntimeError, match="invalid storage_offset"):
        torch.as_strided(base, (2,), (1,), -1)


@pytest.mark.usefixtures("has_device")
def test_resize():
    # Verify resizing a tensor smaller updates metadata and retains storage.
    t = torch.empty((4, 4), device="opencl")
    original_ptr = t.untyped_storage().data_ptr()
    t.resize_((2, 2))
    assert t.shape == torch.Size([2, 2])
    assert t.untyped_storage().data_ptr() == original_ptr

    # Verify resizing a scalar tensor.
    t = torch.empty((), device="opencl")
    assert t.ndim == 0
    t.resize_((1,))
    assert t.shape == torch.Size([1])


@pytest.mark.usefixtures("has_device")
def test_reshape_alias():
    # Verify _reshape_alias behaves as an alias and shares storage.
    base = torch.ones((2, 3)).to("opencl")
    reshaped = torch.ops.aten._reshape_alias(base, [6], [1])
    assert reshaped.shape == torch.Size([6])
    assert reshaped.device.type == "opencl"
    assert reshaped.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
