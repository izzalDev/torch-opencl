import torch
import pytest


@pytest.mark.usefixtures("has_device")
class TestAddInPlace:
    def test_add_contiguous_1d(self):
        """1-D tensor in-place addition on OpenCL."""
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="opencl")
        b = torch.tensor([0.5, 1.5, 2.5, 3.5], device="opencl")
        a.add_(b)

        expected = torch.tensor([1.5, 3.5, 5.5, 7.5])
        assert torch.allclose(a.cpu(), expected)

    def test_add_contiguous_2d(self):
        """2-D tensor in-place addition on OpenCL."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="opencl")
        b = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="opencl")
        a.add_(b)

        expected = torch.tensor([[1.1, 2.2], [3.3, 4.4]])
        assert torch.allclose(a.cpu(), expected)

    def test_add_alpha(self):
        """In-place addition with a non-default alpha value."""
        a = torch.tensor([1.0, 2.0], device="opencl")
        b = torch.tensor([10.0, 20.0], device="opencl")
        a.add_(b, alpha=0.5)

        expected = torch.tensor([6.0, 12.0])
        assert torch.allclose(a.cpu(), expected)

    def test_add_non_contiguous_fallback(self):
        """Verify fallback works correctly for non-contiguous tensors."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="opencl")
        b = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device="opencl")

        a_view = a.t()  # Transposed tensor is non-contiguous
        b_view = b.t()
        assert not a_view.is_contiguous()

        a_view.add_(b_view, alpha=2.0)

        expected = torch.tensor([[2.0, 4.0], [3.0, 5.0]])
        assert torch.allclose(a_view.cpu(), expected)

    def test_add_storage_offset(self):
        """Verify addition with sliced tensors (which have non-zero storage offsets)."""
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="opencl")
        b = torch.tensor([10.0, 20.0, 30.0, 40.0], device="opencl")

        # Slice from index 1 to 3
        a_slice = a[1:3]
        b_slice = b[1:3]

        assert a_slice.storage_offset() == 1
        assert b_slice.storage_offset() == 1

        a_slice.add_(b_slice, alpha=1.0)

        expected_slice = torch.tensor([22.0, 33.0])
        assert torch.allclose(a_slice.cpu(), expected_slice)

        expected_full = torch.tensor([1.0, 22.0, 33.0, 4.0])
        assert torch.allclose(a.cpu(), expected_full)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
    def test_add_dtypes(self, dtype):
        """Verify addition with different data types."""
        a = torch.tensor([1, 2, 3], dtype=dtype, device="opencl")
        b = torch.tensor([4, 5, 6], dtype=dtype, device="opencl")
        a.add_(b)

        expected = torch.tensor([5, 7, 9], dtype=dtype)
        assert (a.cpu() == expected).all()
