#pragma once
#include <ATen/core/TensorBody.h>
#include <c10/core/Scalar.h>

namespace at::native::opencl {

at::Tensor &zero_(at::Tensor &self);
at::Tensor &fill_(at::Tensor &self, const at::Scalar &value);

} // namespace at::native::opencl
