#include "opencl_allocator.h"
#include "opencl_context.h"

#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace {

at::Tensor empty_strided_opencl(c10::IntArrayRef size, c10::IntArrayRef stride,
                                std::optional<at::ScalarType> dtype,
                                std::optional<at::Layout> layout,
                                std::optional<at::Device> device,
                                std::optional<bool> pin_memory) {
  auto dtype_ = c10::dtype_or_default(dtype);
  auto device_ = device.value_or(at::Device(c10::DeviceType::PrivateUse1, 0));

  auto *allocator = &OpenCLAllocator::instance();
  int64_t nelems = 1;
  for (auto s : size)
    nelems *= s;
  int64_t nbytes = nelems * c10::elementSize(dtype_);

  auto storage = c10::Storage(c10::Storage::use_byte_size_t(), nbytes,
                              allocator->allocate(nbytes), allocator,
                              /*resizable=*/false);

  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      caffe2::TypeMeta::fromScalarType(dtype_));

  at::native::setStrided(tensor, size, stride, static_cast<int64_t>(0));
  return tensor;
}

} // namespace
namespace {

at::Tensor copy_from_opencl(const at::Tensor &self, const at::Tensor &dst,
                            bool non_blocking) {
  auto &dev = OpenCLContext::instance().current_device();
  size_t nbytes = self.nbytes();

  bool src_is_cl = self.device().type() == c10::DeviceType::PrivateUse1;
  bool dst_is_cl = dst.device().type() == c10::DeviceType::PrivateUse1;

  if (src_is_cl && !dst_is_cl) {
    // Device → Host
    auto *src_buf = get_cl_buffer(self.data_ptr());
    if (!src_buf)
      throw std::runtime_error("_copy_from: src buffer tidak dikenal");
    dev.queue.enqueueReadBuffer(*src_buf, CL_TRUE, 0, nbytes, dst.data_ptr());

  } else if (!src_is_cl && dst_is_cl) {
    // Host → Device
    auto *dst_buf = get_cl_buffer(dst.data_ptr());
    if (!dst_buf)
      throw std::runtime_error("_copy_from: dst buffer tidak dikenal");
    dev.queue.enqueueWriteBuffer(*dst_buf, CL_TRUE, 0, nbytes, self.data_ptr());

  } else if (src_is_cl && dst_is_cl) {
    // Device → Device
    auto *src_buf = get_cl_buffer(self.data_ptr());
    auto *dst_buf = get_cl_buffer(dst.data_ptr());
    if (!src_buf || !dst_buf)
      throw std::runtime_error("_copy_from: buffer tidak dikenal");
    dev.queue.enqueueCopyBuffer(*src_buf, *dst_buf, 0, 0, nbytes);
    dev.queue.finish();
  }

  return dst;
}
} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  std::cerr << "[torch-opencl] registering aten::empty_strided\n";
  m.impl("empty_strided", &empty_strided_opencl);
  std::cerr << "[torch-opencl] registering aten::_copy_from\n";
  m.impl("_copy_from", &copy_from_opencl);
}
