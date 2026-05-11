#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <torch/library.h>

#include "native/Minimal.h"

namespace at::openreg {

namespace {

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT WRAPPER
at::Tensor wrapper_empty_memory_format(c10::IntArrayRef size,
                                       std::optional<c10::ScalarType> dtype_opt,
                                       std::optional<c10::Layout> layout_opt,
                                       std::optional<c10::Device> device_opt,
                                       std::optional<bool> pin_memory_opt,
                                       std::optional<c10::MemoryFormat> memory_format_opt) {
  return at::native::opencl::empty_memory_format(size, dtype_opt, layout_opt, device_opt,
                                                 pin_memory_opt, memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT WRAPPER

}  // namespace

// LITERALINCLUDE START: TORCH_LIBRARY_IMPL DEFAULT
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", wrapper_empty_memory_format);
}
// LITERALINCLUDE END: TORCH_LIBRARY_IMPL DEFAULT
}  // namespace at::openreg
