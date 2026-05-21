#pragma once

#include <ATen/core/TensorBody.h>

namespace at::native::opencl {

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt
);

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt
);

at::Tensor as_strided(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset
);

const at::Tensor &resize_(
    const at::Tensor &self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format
);

at::Tensor _reshape_alias(
    const at::Tensor &self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride
);

at::Tensor
_copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking);

} // namespace at::native::opencl
