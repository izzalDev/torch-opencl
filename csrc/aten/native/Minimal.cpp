#include "aten/native/Minimal.h"

#include "runtime/CLFunctions.h"
#include <ATen/Dispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

namespace at::native::opencl {

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

at::Tensor view(const at::Tensor &self, c10::SymIntArrayRef size)
{
    TORCH_CHECK(self.is_contiguous(), "view: input tensor must be contiguous");

    auto inferred = at::infer_size_dv(size, self.sym_numel());

    auto strides_opt = at::detail::computeStride(
        self.sym_sizes(), self.sym_strides(), inferred
    );
    TORCH_CHECK(
        strides_opt.has_value(),
        "view size is not compatible with input tensor's size and strides "
        "(at least one dimension spans across two contiguous subspaces). "
        "Use .reshape(...) instead."
    );

    auto result = self.alias();
    result.unsafeGetTensorImpl()->set_sizes_and_strides(inferred, *strides_opt);
    return result;
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

    size_t new_bytes = at::detail::computeStorageNbytesContiguous(
        C10_AS_INTARRAYREF_SLOW(size), itemsize, storage_offset
    );

    if (new_bytes > storage_impl->nbytes()) {
        auto allocator = storage_impl->allocator();
        TORCH_CHECK(allocator, "resize_: missing allocator");

        at::DataPtr new_data = allocator->allocate(new_bytes);

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

at::Tensor
_copy_from(const at::Tensor &self, const at::Tensor &dst, bool /*non_blocking*/)
{
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

    if (nbytes == 0) {
        return dst;
    }

    // CPU → OpenCL
    if (!src_is_cl && dst_is_cl) {
        const auto &dst_alloc = c10::opencl::get_alloc(dst);
        const auto &queue = c10::opencl::get_cl_queue(dst_alloc.device);

        const size_t dst_offset = dst.storage_offset() * dst.element_size();

        const cl_int err = queue.enqueueWriteBuffer(
            dst_alloc.buffer,
            CL_FALSE,
            dst_offset,
            nbytes,
            self.const_data_ptr()
        );
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueWriteBuffer failed, code: ", err
        );
        queue.finish();
        return dst;
    }

    // OpenCL → CPU
    if (src_is_cl && !dst_is_cl) {
        const auto &src_alloc = c10::opencl::get_alloc(self);
        const auto &queue = c10::opencl::get_cl_queue(src_alloc.device);

        const size_t src_offset = self.storage_offset() * self.element_size();

        const cl_int err = queue.enqueueReadBuffer(
            src_alloc.buffer,
            CL_FALSE,
            src_offset,
            nbytes,
            dst.mutable_data_ptr()
        );
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueReadBuffer failed, code: ", err
        );
        queue.finish();
        return dst;
    }

    // OpenCL → OpenCL
    if (src_is_cl && dst_is_cl) {
        const auto &src_alloc = c10::opencl::get_alloc(self);
        const auto &dst_alloc = c10::opencl::get_alloc(dst);

        TORCH_CHECK(
            src_alloc.device == dst_alloc.device,
            "_copy_from: copy between OpenCL devices is not supported yet "
            "(src device=",
            src_alloc.device,
            ", dst device=",
            dst_alloc.device,
            ")"
        );

        const auto &queue = c10::opencl::get_cl_queue(src_alloc.device);

        const size_t src_offset = self.storage_offset() * self.element_size();
        const size_t dst_offset = dst.storage_offset() * dst.element_size();

        const cl_int err = queue.enqueueCopyBuffer(
            src_alloc.buffer, dst_alloc.buffer, src_offset, dst_offset, nbytes
        );
        TORCH_CHECK(
            err == CL_SUCCESS, "clEnqueueCopyBuffer failed, code: ", err
        );
        queue.finish();
        return dst;
    }

    TORCH_CHECK(
        false,
        "_copy_from: unexpected device combination: src=",
        self.device(),
        " dst=",
        dst.device()
    );
}

at::Tensor _copy_from_and_resize(const at::Tensor &self, const at::Tensor &dst)
{
    if (dst.sizes() != self.sizes()) {
        resize_(dst, self.sym_sizes(), c10::nullopt);
    }
    return _copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor &self)
{
    TORCH_CHECK(
        self.numel() == 1,
        "Tensor must have exactly 1 element to be converted to a scalar."
    );

    const auto &alloc = c10::opencl::get_alloc(self);
    const auto &queue = c10::opencl::get_cl_queue(alloc.device);

    return AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Bool, self.scalar_type(), "_local_scalar_dense", [&] {
            scalar_t value;
            cl_int err = queue.enqueueReadBuffer(
                alloc.buffer,
                CL_TRUE,
                self.storage_offset() * sizeof(scalar_t),
                sizeof(scalar_t),
                &value
            );
            TORCH_CHECK(
                err == CL_SUCCESS,
                "OpenCL enqueueReadBuffer failed with error code: ",
                err
            );
            return at::Scalar(value);
        }
    );
}

} // namespace at::native::opencl
