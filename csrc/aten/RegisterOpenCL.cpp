#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>
#include <torch/library.h>

#include "native/BinaryOps.h"
#include "native/Fill.h"
#include "native/Minimal.h"

namespace at::opencl {

namespace {

void cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack)
{
    at::native::cpu_fallback(op, stack);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    // Minimal.h
    m.impl("empty.memory_format", at::native::opencl::empty_memory_format);
    m.impl("empty_strided", at::native::opencl::empty_strided);
    m.impl("as_strided", at::native::opencl::as_strided);
    m.impl("resize_", at::native::opencl::resize_);
    m.impl("_reshape_alias", at::native::opencl::_reshape_alias);
    m.impl("view", at::native::opencl::view);
    m.impl("_copy_from", at::native::opencl::_copy_from);
    m.impl("_local_scalar_dense", at::native::opencl::_local_scalar_dense);
    m.impl("_copy_from_and_resize", at::native::opencl::_copy_from_and_resize);

    // Fill.h
    m.impl("zero_", at::native::opencl::zero_);
    m.impl("fill_.Scalar", at::native::opencl::fill_);

    // BinaryOps.h
    m.impl("add_.Tensor", at::native::opencl::add_);

    // CPU fallback for display/diagnostic ops, not worth a dedicated kernel.
    for (const auto *op :
         {"eq.Scalar_out",
          "eq.Tensor_out",
          "abs.out",
          "ne.Scalar_out",
          "ne.Tensor_out",
          "bitwise_and.Tensor_out",
          "masked_select.out",
          "masked_select",
          "isfinite.out",
          "min.out",
          "max.out",
          "ceil.out",
          "mul.out",
          "div.out",
          "add.out",
          "sub.out",
          "min",
          "max",
          "gt.Scalar_out"}) {
        m.impl(op, torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
    }
}

} // namespace at::opencl
