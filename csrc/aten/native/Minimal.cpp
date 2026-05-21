#include "aten/native/Minimal.h"

#include <ATen/EmptyTensor.h>
#include <torch/library.h>

#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"

namespace at::native::opencl {

/**
 * Return the CLAllocation handle stored inside a tensor's DataPtr.
 *
 * CLDeviceAllocator::allocate() passes the CLAllocation* as both the
 * ctx and the data pointer of the DataPtr, so .get() gives us the handle
 * directly — not a pointer into the buffer's byte contents.
 */
static const c10::opencl::CLAllocation *get_cl_allocation(const at::Tensor &t)
{
    return static_cast<const c10::opencl::CLAllocation *>(
        t.storage().data_ptr().get()
    );
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
    const auto device = c10::device_or_default(device_opt);
    const auto dtype = c10::dtype_or_default(dtype_opt);
    TORCH_CHECK(device.is_privateuseone());
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non strided layout not supported"
    );
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory can only be on CPU"
    );
    const c10::DeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
    auto allocator = at::GetAllocator(at::kPrivateUse1);
    return at::detail::empty_generic(
        size, allocator, pu1_dks, dtype, memory_format_opt
    );
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
    const auto device = c10::device_or_default(device_opt);
    const auto dtype = c10::dtype_or_default(dtype_opt);
    TORCH_CHECK(device.is_privateuseone());
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non strided layout not supported"
    );
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory can only be on CPU"
    );
    const c10::DeviceGuard device_guard(device);
    constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
    auto allocator = at::GetAllocator(at::kPrivateUse1);
    return at::detail::empty_strided_generic(
        size, stride, allocator, pu1_dks, dtype
    );
}

at::Tensor as_strided(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset
)
{
    auto result = self.alias();
    auto *impl = result.unsafeGetTensorImpl();
    TORCH_CHECK(
        size.size() == stride.size(),
        "as_strided: size and stride must have same length"
    );
    for (const auto &s : stride) {
        TORCH_CHECK(
            s >= 0,
            "as_strided: negative stride not supported in OpenCL backend"
        );
    }
    TORCH_CHECK(
        storage_offset.has_value(), "as_strided: storage_offset is required"
    );
    TORCH_CHECK(
        storage_offset->expect_int() >= 0, "as_strided: invalid storage_offset"
    );
    impl->set_sizes_and_strides(size, stride);
    impl->set_storage_offset(storage_offset->expect_int());
    return result;
}

const at::Tensor &resize_(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format
)
{
    auto *impl = self.unsafeGetTensorImpl();

    if (impl->sizes() == C10_AS_INTARRAYREF_SLOW(size)) {
        return self;
    }

    auto storage = impl->unsafe_storage();
    TORCH_CHECK(storage, "resize_: invalid storage");

    auto *storage_impl = storage.unsafeGetStorageImpl();
    auto itemsize = impl->dtype().itemsize();
    auto storage_offset = impl->storage_offset();

    auto new_bytes_sym = at::detail::computeStorageNbytesContiguous(
        C10_AS_INTARRAYREF_SLOW(size), itemsize, storage_offset
    );

    size_t new_bytes = at::detail::computeStorageNbytesContiguous(
        C10_AS_INTARRAYREF_SLOW(size), itemsize, storage_offset
    );

    if (new_bytes > storage_impl->nbytes()) {
        auto allocator = storage_impl->allocator();
        TORCH_CHECK(allocator, "resize_: missing allocator");

        at::DataPtr new_data = allocator->allocate(new_bytes);

        std::memcpy(
            new_data.get(),
            storage_impl->data(),
            std::min(storage_impl->nbytes(), new_bytes)
        );

        storage_impl->set_data_ptr(std::move(new_data));
        storage_impl->set_nbytes(new_bytes);
    }

    impl->set_sizes_contiguous(C10_AS_INTARRAYREF_SLOW(size));
    if (memory_format.has_value()) {
        impl->empty_tensor_restride(memory_format.value());
    }

    return self;
}

at::Tensor _reshape_alias(
    const at::Tensor &self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride
)
{
    auto result = self.alias();
    auto *impl = result.unsafeGetTensorImpl();
    TORCH_CHECK(
        size.size() == stride.size(), "reshape_alias: size/stride mismatch"
    );
    impl->set_sizes_and_strides(
        C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride)
    );

    return result;
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

at::Tensor
_copy_from(const at::Tensor &self, const at::Tensor &dst, bool /*non_blocking*/)
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
    const size_t nbytes =
        static_cast<size_t>(self.numel()) * self.element_size();

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
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueWriteBuffer failed, code: ", err
        );
        queue.finish(); // … finish() makes it effectively synchronous
        return dst;
    }

    // ------------------------------------------------------------------
    // (B) OpenCL → CPU
    // ------------------------------------------------------------------
    if (src_is_cl && !dst_is_cl) {
        const auto *src_alloc = get_cl_allocation(self);
        cl::CommandQueue &queue = c10::opencl::get_cl_queue(src_alloc->device);

        const cl_int err = queue.enqueueReadBuffer(
            src_alloc->buffer, CL_FALSE, 0, nbytes, dst.mutable_data_ptr()
        );
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueReadBuffer failed, code: ", err
        );
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
            "_copy_from: peer-to-peer copy between OpenCL devices is not "
            "supported "
            "yet "
            "(src device=",
            src_alloc->device,
            ", dst device=",
            dst_alloc->device,
            ")"
        );

        cl::CommandQueue &queue = c10::opencl::get_cl_queue(src_alloc->device);

        const cl_int err = queue.enqueueCopyBuffer(
            src_alloc->buffer, dst_alloc->buffer, 0, 0, nbytes
        );
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueCopyBuffer failed, code: ", err
        );
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
