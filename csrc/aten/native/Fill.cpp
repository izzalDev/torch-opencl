#include "aten/native/Fill.h"
#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
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

at::Tensor &fill_(at::Tensor &self, const at::Scalar &value)
{
    if (self.numel() == 0) {
        return self;
    }
    const c10::DeviceGuard device_guard(self.device());
    const auto *alloc = get_cl_allocation(self);
    cl::CommandQueue &queue = c10::opencl::get_cl_queue(alloc->device);

    const auto offset = self.storage_offset() * self.element_size();
    const auto nbytes = self.numel() * self.element_size();

    cl_int err = CL_SUCCESS;

    AT_DISPATCH_ALL_TYPES(self.scalar_type(), "fill_opencl", [&] {
        scalar_t raw_value = value.to<scalar_t>();

#ifdef __APPLE__
        // clEnqueueFillBuffer is buggy on Apple OpenCL — it reads the pointer
        std::vector<scalar_t> host_buf(self.numel(), raw_value);
        err = queue.enqueueWriteBuffer(
            alloc->buffer, CL_TRUE, offset, nbytes, host_buf.data()
        );
#else
        err = queue.enqueueFillBuffer( alloc->buffer, raw_value, offset, nbytes);
#endif
    });

    TORCH_CHECK(
        err == CL_SUCCESS, "enqueueFillBuffer failed with error code ", err
    );

    queue.finish();
    return self;
}

} // namespace at::native::opencl
