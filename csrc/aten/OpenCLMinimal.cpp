#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <torch/library.h>

#include "native/Minimal.h"

namespace at::openreg {

namespace {

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT WRAPPER
at::Tensor wrapper_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt
)
{
    return at::native::opencl::empty_memory_format(
        size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt
    );
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT WRAPPER

// LITERALINCLUDE START: EMPTY_STRIDED WRAPPER
at::Tensor wrapper_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt
)
{
    return at::native::opencl::empty_strided(
        size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt
    );
}
// LITERALINCLUDE END: EMPTY_STRIDED WRAPPER

// LITERALINCLUDE START: COPY_FROM WRAPPER
at::Tensor wrapper_copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking)
{
    return at::native::opencl::_copy_from(self, dst, non_blocking);
}

// LITERALINCLUDE END: COPY_FROM WRAPPER

} // namespace

// LITERALINCLUDE START: TORCH_LIBRARY_IMPL DEFAULT
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", wrapper_empty_memory_format);
    m.impl("empty_strided", wrapper_empty_strided);
    m.impl("_copy_from", wrapper_copy_from);
}
// LITERALINCLUDE END: TORCH_LIBRARY_IMPL DEFAULT

} // namespace at::openreg
