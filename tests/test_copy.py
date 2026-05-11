"""
Tests untuk aten::copy_ — transfer data CPU ↔ OpenCL.
"""

import pytest
import torch


@pytest.fixture
def cpu_tensor():
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


@pytest.mark.usefixtures("has_device")
class TestH2D:
    """Host → Device"""

    def test_basic_h2d(self, cpu_tensor):
        ocl = cpu_tensor.to("opencl")
        assert ocl.device.type == "opencl"

    def test_values_survive_h2d_d2h(self, cpu_tensor):
        ocl = cpu_tensor.to("opencl")
        back = ocl.cpu()
        assert torch.allclose(cpu_tensor, back)

    def test_zeros(self):
        t = torch.zeros(128)
        ocl = t.to("opencl")
        back = ocl.cpu()
        assert torch.allclose(t, back)

    def test_ones(self):
        t = torch.ones(64)
        ocl = t.to("opencl")
        back = ocl.cpu()
        assert torch.allclose(t, back)

    def test_float32(self):
        t = torch.randn(256, dtype=torch.float32)
        assert torch.allclose(t, t.to("opencl").cpu())

    def test_2d_tensor(self):
        t = torch.arange(12.0).reshape(3, 4)
        assert torch.allclose(t, t.to("opencl").cpu())

    def test_empty_tensor(self):
        t = torch.empty(0)
        ocl = t.to("opencl")
        back = ocl.cpu()
        assert back.numel() == 0

    def test_large_tensor(self):
        t = torch.randn(1024 * 1024)
        assert torch.allclose(t, t.to("opencl").cpu(), atol=1e-5)


@pytest.mark.usefixtures("has_device")
class TestD2H:
    """Device → Host"""

    def test_basic_d2h(self, cpu_tensor):
        back = cpu_tensor.to("opencl").cpu()
        assert back.device.type == "cpu"
        assert torch.allclose(cpu_tensor, back)

    def test_dtype_preserved(self):
        for dtype in [torch.float32, torch.int32, torch.int64]:
            t = torch.zeros(8, dtype=dtype)
            back = t.to("opencl").cpu()
            assert back.dtype == dtype

    def test_shape_preserved(self):
        for shape in [(4,), (2, 4), (2, 3, 4)]:
            t = torch.randn(*shape)
            back = t.to("opencl").cpu()
            assert back.shape == t.shape


@pytest.mark.usefixtures("has_device")
class TestD2D:
    """Device → Device (same device)"""

    def test_clone_on_device(self, cpu_tensor):
        ocl = cpu_tensor.to("opencl")
        cloned = ocl.clone()
        assert cloned.device.type == "opencl"
        assert torch.allclose(cpu_tensor, cloned.cpu())

    def test_clone_is_independent(self):
        """Mutasi pada clone tidak mempengaruhi sumber."""
        t = torch.tensor([1.0, 2.0, 3.0])
        ocl = t.to("opencl")
        cloned = ocl.clone()

        # Modifikasi cloned di CPU, pastikan original tidak berubah
        cloned_cpu = cloned.cpu()
        cloned_cpu[0] = 999.0

        original_back = ocl.cpu()
        assert original_back[0].item() == pytest.approx(1.0)


@pytest.mark.usefixtures("has_device")
class TestMultiDevice:
    """Cross-device copy (butuh ≥ 2 OpenCL device)."""

    def test_cross_device_copy(self, device_count):
        if device_count < 2:
            pytest.skip("Butuh minimal 2 OpenCL device")

        t = torch.tensor([1.0, 2.0, 3.0])
        ocl0 = t.to(torch.device("opencl", 0))
        ocl1 = ocl0.to(torch.device("opencl", 1))
        back = ocl1.cpu()
        assert torch.allclose(t, back)
