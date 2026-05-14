#include "aten/native/Minimal.h"

#include <torch/library.h>

#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"

namespace at::native::opencl {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Allocate OpenCL storage and build a tensor with the given size and stride.
 * Both empty_memory_format and empty_strided funnel through here.
 *
 * Storage size covers the highest byte address reachable via the strides so
 * that non-contiguous layouts (column-major, transposed, etc.) are safe.
 */
static at::Tensor make_opencl_tensor(
    c10::IntArrayRef size, c10::IntArrayRef stride, c10::ScalarType dtype, c10::Device device
)
{
    TORCH_INTERNAL_ASSERT(device.is_privateuseone());

    const c10::DeviceGuard guard(device);

    size_t nbytes = 0;
    if (c10::multiply_integers(size) > 0) {
        // last_offset (in elements) = 1 + sum_i( (size[i]-1) * |stride[i]| )
        int64_t last_offset = 1;
        for (size_t i = 0; i < size.size(); ++i)
            last_offset += (size[i] - 1) * std::abs(stride[i]);

        nbytes = static_cast<size_t>(last_offset) * c10::elementSize(dtype);
    }

    auto *alloc = at::GetAllocator(at::kPrivateUse1);
    TORCH_CHECK(alloc, "OpenCL allocator not registered");

    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        nbytes > 0 ? alloc->allocate(nbytes) : at::DataPtr{},
        alloc,
        /*resizable=*/true
    );

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype)
    );

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return tensor;
}

/**
 * Return the CLAllocation handle stored inside a tensor's DataPtr.
 *
 * CLDeviceAllocator::allocate() passes the CLAllocation* as both the
 * ctx and the data pointer of the DataPtr, so .get() gives us the handle
 * directly — not a pointer into the buffer's byte contents.
 */
static const c10::opencl::CLAllocation *get_cl_allocation(const at::Tensor &t)
{
    return static_cast<const c10::opencl::CLAllocation *>(t.storage().data_ptr().get());
}

// ---------------------------------------------------------------------------
// empty.memory_format
// Strides are computed row-major (contiguous) from size.
// ---------------------------------------------------------------------------

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt
)
{
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "empty_memory_format: non-strided layout not supported"
    );
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "empty_memory_format: pin_memory not supported on OpenCL"
    );

    const auto fmt = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
    TORCH_CHECK(
        fmt == c10::MemoryFormat::Contiguous || fmt == c10::MemoryFormat::Preserve,
        "empty_memory_format: only Contiguous memory format is supported, got ",
        fmt
    );

    const auto dtype = c10::dtype_or_default(dtype_opt);
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(device.is_privateuseone(), "expected opencl device, got ", device);

    const auto ndim = size.size();
    std::vector<int64_t> strides(ndim, 1);
    for (int i = (int)ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * size[i + 1];

    return make_opencl_tensor(size, strides, dtype, device);
}

// ---------------------------------------------------------------------------
// empty_strided
// Caller supplies explicit strides — no layout assumption.
// ---------------------------------------------------------------------------

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt
)
{
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "empty_strided: non-strided layout not supported"
    );
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "empty_strided: pin_memory not supported on OpenCL"
    );
    TORCH_CHECK(
        size.size() == stride.size(),
        "empty_strided: size and stride must have the same length, got ",
        size.size(),
        " vs ",
        stride.size()
    );

    const auto dtype = c10::dtype_or_default(dtype_opt);
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(device.is_privateuseone(), "expected opencl device, got ", device);

    return make_opencl_tensor(size, stride, dtype, device);
}

// ---------------------------------------------------------------------------
// _copy_from
//
// PyTorch calls this to move data between devices.
// Contract: self = source, dst = destination. Returns dst.
//
//   (A) CPU    → OpenCL  : clEnqueueWriteBuffer
//   (B) OpenCL → CPU     : clEnqueueReadBuffer
//   (C) OpenCL → OpenCL  : clEnqueueCopyBuffer  (same device only)
//
// Peer-to-peer (different OpenCL devices) is rejected for now.
// non_blocking is accepted but ignored — always synchronous until a
// pinned-memory allocator is wired up.
// ---------------------------------------------------------------------------

at::Tensor _copy_from(const at::Tensor &self, const at::Tensor &dst, bool /*non_blocking*/)
{
    TORCH_CHECK(
        self.dtype() == dst.dtype(),
        "_copy_from: dtype mismatch: src=",
        self.dtype(),
        " dst=",
        dst.dtype()
    );
    TORCH_CHECK(
        self.numel() == dst.numel(),
        "_copy_from: numel mismatch: src=",
        self.numel(),
        " dst=",
        dst.numel()
    );
    TORCH_CHECK(self.is_contiguous(), "_copy_from: src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "_copy_from: dst must be contiguous");

    const bool src_is_cl = self.device().is_privateuseone();
    const bool dst_is_cl = dst.device().is_privateuseone();
    const size_t nbytes = static_cast<size_t>(self.numel()) * self.element_size();

    // ------------------------------------------------------------------
    // (A) CPU → OpenCL
    // ------------------------------------------------------------------
    if (!src_is_cl && dst_is_cl) {
        const auto *dst_alloc = get_cl_allocation(dst);
        cl::CommandQueue &queue = c10::opencl::get_cl_queue(dst_alloc->device);

        const cl_int err = queue.enqueueWriteBuffer(
            dst_alloc->buffer,
            CL_FALSE, // non-blocking enqueue …
            0,
            nbytes,
            self.const_data_ptr()
        );
        TORCH_CHECK(err == CL_SUCCESS, "clEnqueueWriteBuffer failed, code: ", err);
        queue.finish(); // … finish() makes it effectively synchronous
        return dst;
    }

    // ------------------------------------------------------------------
    // (B) OpenCL → CPU
    // ------------------------------------------------------------------
    if (src_is_cl && !dst_is_cl) {
        const auto *src_alloc = get_cl_allocation(self);
        cl::CommandQueue &queue = c10::opencl::get_cl_queue(src_alloc->device);

        const cl_int err =
            queue.enqueueReadBuffer(src_alloc->buffer, CL_FALSE, 0, nbytes, dst.mutable_data_ptr());
        TORCH_CHECK(err == CL_SUCCESS, "clEnqueueReadBuffer failed, code: ", err);
        queue.finish();
        return dst;
    }

    // ------------------------------------------------------------------
    // (C) OpenCL → OpenCL
    // ------------------------------------------------------------------
    if (src_is_cl && dst_is_cl) {
        const auto *src_alloc = get_cl_allocation(self);
        const auto *dst_alloc = get_cl_allocation(dst);

        TORCH_CHECK(
            src_alloc->device == dst_alloc->device,
            "_copy_from: peer-to-peer copy between OpenCL devices is not supported yet "
            "(src device=",
            src_alloc->device,
            ", dst device=",
            dst_alloc->device,
            ")"
        );

        cl::CommandQueue &queue = c10::opencl::get_cl_queue(src_alloc->device);

        const cl_int err =
            queue.enqueueCopyBuffer(src_alloc->buffer, dst_alloc->buffer, 0, 0, nbytes);
        TORCH_CHECK(err == CL_SUCCESS, "clEnqueueCopyBuffer failed, code: ", err);
        queue.finish();
        return dst;
    }

    // CPU → CPU should never reach here (handled by ATen)
    TORCH_CHECK(
        false,
        "_copy_from: unexpected device combination: src=",
        self.device(),
        " dst=",
        dst.device()
    );
}

} // namespace at::native::opencl
