import torch
import pytest


@pytest.mark.usefixtures("has_device")
class TestEmptyStrided:
    def test_basic_1d(self):
        """1-D tensor with stride [1]."""
        t = torch.empty_strided((4,), (1,), device="opencl")
        assert t.device.type == "opencl"
        assert t.shape == torch.Size([4])
        assert t.stride() == (1,)

    def test_contiguous_2d(self):
        """Row-major 2-D: strides match contiguous layout."""
        t = torch.empty_strided((3, 4), (4, 1), device="opencl")
        assert t.shape == torch.Size([3, 4])
        assert t.stride() == (4, 1)
        assert t.is_contiguous()

    def test_column_major_2d(self):
        """Column-major (Fortran order) strides."""
        t = torch.empty_strided((3, 4), (1, 3), device="opencl")
        assert t.stride() == (1, 3)
        assert not t.is_contiguous()

    def test_scalar_tensor(self):
        """0-D tensor (scalar): empty size and stride."""
        t = torch.empty_strided((), (), device="opencl")
        assert t.shape == torch.Size([])
        assert t.ndim == 0

    def test_empty_dim(self):
        """Tensor with a zero-sized dimension allocates no memory."""
        t = torch.empty_strided((0, 4), (4, 1), device="opencl")
        assert t.numel() == 0

    def test_dtype_float64(self):
        t = torch.empty_strided((2, 2), (2, 1), dtype=torch.float64, device="opencl")
        assert t.dtype == torch.float64

    def test_dtype_int32(self):
        t = torch.empty_strided((5,), (1,), dtype=torch.int32, device="opencl")
        assert t.dtype == torch.int32

    def test_mismatched_size_stride_raises(self):
        with pytest.raises(Exception, match="same length"):
            torch.empty_strided((3, 4), (1,), device="opencl")

    def test_non_strided_layout_raises(self):
        with pytest.raises(Exception):
            torch.empty_strided((4,), (1,), layout=torch.sparse_coo, device="opencl")


@pytest.mark.usefixtures("has_device")
class TestCopyFrom:
    def test_cpu_to_opencl(self):
        """(A) CPU → OpenCL: data must be preserved."""
        cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        cl = cpu.to("opencl")
        result = cl.cpu()
        assert cl.device.type == "opencl"
        assert torch.allclose(cpu, result)

    def test_opencl_to_cpu(self):
        """(B) OpenCL → CPU: data must survive the round-trip."""
        original = torch.arange(8, dtype=torch.float32)
        back = original.to("opencl").cpu()
        assert back.device.type == "cpu"
        assert torch.allclose(original, back)

    def test_opencl_to_opencl_same_device(self):
        """(C) OpenCL → OpenCL same device: values must match."""
        src = torch.ones(4, dtype=torch.float32).to("opencl")
        dst = src.clone()
        assert torch.allclose(src.cpu(), dst.cpu())

    def test_roundtrip_multidim(self):
        """2-D tensor round-trip preserves shape and values."""
        cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        back = cpu.to("opencl").cpu()
        assert back.shape == cpu.shape
        assert torch.allclose(cpu, back)

    def test_roundtrip_dtype_int32(self):
        cpu = torch.tensor([10, 20, 30], dtype=torch.int32)
        back = cpu.to("opencl").cpu()
        assert (cpu == back).all()

    @pytest.mark.skipif(
        torch.opencl.device_count() < 2,  # type: ignore[attr-defined]
        reason="peer-to-peer test requires at least 2 OpenCL devices",
    )
    def test_peer_to_peer_raises(self):
        """Cross-device copy must raise until P2P is implemented."""
        src = torch.ones(4).to("opencl:0")
        with pytest.raises(RuntimeError, match="peer-to-peer"):
            src.to("opencl:1")
