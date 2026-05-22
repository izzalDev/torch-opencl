#include "aten/native/Fill.h"
#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"
#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceGuard.h>

#ifdef __APPLE__
#include <vector>
#endif

namespace at::native::opencl {

static const c10::opencl::CLAllocation *get_cl_allocation(const at::Tensor &t)
{
    return static_cast<const c10::opencl::CLAllocation *>(
        t.storage().data_ptr().get()
    );
}

at::Tensor &zero_(at::Tensor &self)
{
    if (self.numel() == 0) {
        return self;
    }

    TORCH_CHECK(
        self.device().is_privateuseone(),
        "zero_: tensor must be on an OpenCL device"
    );

    const c10::DeviceGuard device_guard(self.device());

    const auto *alloc = get_cl_allocation(self);
    cl::CommandQueue &queue = c10::opencl::get_cl_queue(alloc->device);

    const auto offset = self.storage_offset() * self.element_size();
    const auto nbytes = self.numel() * self.element_size();

#ifdef __APPLE__
    // clEnqueueFillBuffer is buggy on Apple OpenCL — it reads the pointer
    // address itself as the pattern rather than the value it points to.
    // We fall back to enqueueWriteBuffer with a zero-initialized host buffer.
    std::vector<uint8_t> zeros(nbytes, 0);
    const cl_int err = queue.enqueueWriteBuffer(
        alloc->buffer, CL_FALSE, offset, nbytes, zeros.data()
    );
    TORCH_CHECK(
        err == CL_SUCCESS, "enqueueWriteBuffer failed with error code ", err
    );
#else
    const cl_uchar pattern = 0;
    const cl_int err =
        queue.enqueueFillBuffer(alloc->buffer, pattern, offset, nbytes);
    TORCH_CHECK(
        err == CL_SUCCESS, "enqueueFillBuffer failed with error code ", err
    );
#endif

    queue.finish();
    return self;
}

} // namespace at::native::opencl
