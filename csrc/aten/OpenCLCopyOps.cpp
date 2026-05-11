#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <CL/opencl.hpp>
#include <vector>

#include "runtime/OpenCLDeviceAllocator.h"
#include "runtime/OpenCLException.h"
#include "runtime/OpenCLFunctions.h"

namespace at::opencl {

namespace {

// ---------------------------------------------------------------------------
// Helper: unwrap BufferEntry dari data pointer tensor OpenCL
// ---------------------------------------------------------------------------
const c10::opencl::BufferEntry *get_buffer_entry(const at::Tensor &t) {
  TORCH_CHECK(t.is_privateuseone(), "Expected OpenCL tensor, got ", t.device());
  TORCH_CHECK(t.is_contiguous(), "OpenCL copy requires contiguous tensors");
  return static_cast<const c10::opencl::BufferEntry *>(t.data_ptr());
}

c10::opencl::BufferEntry *get_buffer_entry_mut(at::Tensor &t) {
  TORCH_CHECK(t.is_privateuseone(), "Expected OpenCL tensor, got ", t.device());
  TORCH_CHECK(t.is_contiguous(), "OpenCL copy requires contiguous tensors");
  return static_cast<c10::opencl::BufferEntry *>(t.data_ptr());
}

// ---------------------------------------------------------------------------
// Host → Device
// ---------------------------------------------------------------------------
void copy_h2d(at::Tensor &dst, const at::Tensor &src) {
  const size_t nbytes = src.nbytes();
  if (nbytes == 0) return;

  c10::opencl::BufferEntry *dst_entry = get_buffer_entry_mut(dst);
  TORCH_CHECK(nbytes <= dst_entry->size, "copy H2D: source size (", nbytes,
              ") exceeds destination buffer size (", dst_entry->size, ")");

  cl::CommandQueue &queue = c10::opencl::get_cl_queue(dst_entry->device);
  OPENCL_CHECK(
      queue.enqueueWriteBuffer(dst_entry->buffer, CL_TRUE, 0, nbytes, src.const_data_ptr()));
}

// ---------------------------------------------------------------------------
// Device → Host
// ---------------------------------------------------------------------------
void copy_d2h(at::Tensor &dst, const at::Tensor &src) {
  const size_t nbytes = src.nbytes();
  if (nbytes == 0) return;

  const c10::opencl::BufferEntry *src_entry = get_buffer_entry(src);
  TORCH_CHECK(nbytes <= src_entry->size, "copy D2H: source buffer size (", src_entry->size,
              ") smaller than requested bytes (", nbytes, ")");

  cl::CommandQueue &queue = c10::opencl::get_cl_queue(src_entry->device);
  OPENCL_CHECK(queue.enqueueReadBuffer(src_entry->buffer, CL_TRUE, 0, nbytes, dst.data_ptr()));
}

// ---------------------------------------------------------------------------
// Device → Device
// ---------------------------------------------------------------------------
void copy_d2d(at::Tensor &dst, const at::Tensor &src) {
  const size_t nbytes = src.nbytes();
  if (nbytes == 0) return;

  const c10::opencl::BufferEntry *src_entry = get_buffer_entry(src);
  c10::opencl::BufferEntry *dst_entry = get_buffer_entry_mut(dst);

  TORCH_CHECK(nbytes <= src_entry->size, "copy D2D: source buffer too small");
  TORCH_CHECK(nbytes <= dst_entry->size, "copy D2D: destination buffer too small");

  cl::CommandQueue &queue = c10::opencl::get_cl_queue(dst_entry->device);

  if (src_entry->device == dst_entry->device) {
    OPENCL_CHECK(queue.enqueueCopyBuffer(src_entry->buffer, dst_entry->buffer, 0, 0, nbytes));
  } else {
    // Cross-device: lewat host sementara karena OpenCL tidak menjamin
    // cross-context buffer copy
    std::vector<char> tmp(nbytes);
    cl::CommandQueue &src_queue = c10::opencl::get_cl_queue(src_entry->device);
    OPENCL_CHECK(src_queue.enqueueReadBuffer(src_entry->buffer, CL_TRUE, 0, nbytes, tmp.data()));
    OPENCL_CHECK(queue.enqueueWriteBuffer(dst_entry->buffer, CL_TRUE, 0, nbytes, tmp.data()));
  }

  OPENCL_CHECK(queue.finish());
}

// ---------------------------------------------------------------------------
// Wrapper — dipanggil oleh TORCH_LIBRARY_IMPL
// ---------------------------------------------------------------------------
at::Tensor wrapper__copy_from(const at::Tensor &src, const at::Tensor &dst, bool non_blocking) {
  const bool src_opencl = src.is_privateuseone();
  const bool dst_opencl = dst.is_privateuseone();

  at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  at::Tensor dst_c = dst.is_contiguous() ? dst : dst.contiguous();

  TORCH_CHECK(src_c.sizes() == dst_c.sizes(), "_copy_from: shape mismatch ", src_c.sizes(), " vs ",
              dst_c.sizes());
  TORCH_CHECK(src_c.scalar_type() == dst_c.scalar_type(), "_copy_from: dtype mismatch ",
              src_c.scalar_type(), " vs ", dst_c.scalar_type());

  if (src_opencl && dst_opencl) {
    copy_d2d(dst_c, src_c);
  } else if (!src_opencl && dst_opencl) {
    copy_h2d(dst_c, src_c);
  } else if (src_opencl && !dst_opencl) {
    copy_d2h(dst_c, src_c);
  } else {
    TORCH_CHECK(false, "_copy_from: kombinasi device tidak didukung: ", src.device(), " -> ",
                dst.device());
  }

  return dst;
}

at::Tensor wrapper__copy_from_and_resize(const at::Tensor &src, const at::Tensor &dst) {
  return wrapper__copy_from(src, dst, /*non_blocking=*/false);
}

}  // namespace

// ---------------------------------------------------------------------------
// Registrasi ke PyTorch dispatcher
// ---------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_copy_from", wrapper__copy_from);
  m.impl("_copy_from_and_resize", wrapper__copy_from_and_resize);
}

}  // namespace at::opencl
