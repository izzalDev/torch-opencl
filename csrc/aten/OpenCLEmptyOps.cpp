#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/library.h>

#include "runtime/OpenCLDeviceAllocator.h"

namespace at::opencl {

namespace {

at::Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype_opt,
                         std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
                         std::optional<bool> pin_memory_opt) {
  c10::DeviceIndex device_index = device_opt->index();
  at::ScalarType dtype = dtype_opt.value_or(at::ScalarType::Float);
  auto *allocator = c10::opencl::getOpenCLAllocator(device_index);

  // Compute number of bytes needed
  int64_t numel = 1;
  for (auto s : size) numel *= s;
  size_t nbytes = static_cast<size_t>(numel) * at::elementSize(dtype);

  // Build storage
  auto storage =
      c10::Storage(c10::Storage::use_byte_size_t(), nbytes, allocator->allocate(nbytes), allocator,
                   /*resizable=*/false);

  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      at::scalarTypeToTypeMeta(dtype));

  // Apply sizes and strides
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

  return tensor;
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("empty_strided", empty_strided); }

}  // namespace at::opencl
