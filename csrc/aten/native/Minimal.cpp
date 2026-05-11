#include "aten/native/Minimal.h"
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

    TORCH_CHECK(device.is_privateuseone(), "expected opencl device, got ", device);
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non-strided layout not supported"
    );
    TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Pin memory only supported on CPU");

    const c10::DeviceGuard guard(device);

    // Hitung strides row-major
    const auto ndim = size.size();
    std::vector<int64_t> strides(ndim, 1);
    for (int i = (int)ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * size[i + 1];

    // Alokasi storage
    const size_t nbytes =
        static_cast<size_t>(c10::multiply_integers(size)) * c10::elementSize(dtype);

    auto *alloc = at::GetAllocator(at::kPrivateUse1);
    TORCH_CHECK(alloc, "OpenCL allocator belum diregister!");

    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        nbytes > 0 ? alloc->allocate(nbytes) : at::DataPtr{},
        alloc,
        true
    );

    // Buat tensor
    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype)
    );

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
    return tensor;
}

} // namespace at::native::opencl
