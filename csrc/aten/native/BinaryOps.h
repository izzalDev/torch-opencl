#pragma once
#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h>

namespace at::native::opencl {

at::Tensor &add_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha);

} // namespace at::native::opencl
