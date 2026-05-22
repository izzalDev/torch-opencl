#pragma once
#include <ATen/core/TensorBody.h>

namespace at::native::opencl {

at::Tensor &zero_(at::Tensor &self);

}
