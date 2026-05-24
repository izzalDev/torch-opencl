#pragma once

#include "runtime/CLDeviceAllocator.h"
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

namespace c10::opencl {

void ensure_initialized();
bool is_in_bad_fork();
DeviceIndex device_count() noexcept;

const cl::Context &get_cl_context(DeviceIndex device);
const cl::Device &get_cl_device(DeviceIndex device);
const cl::CommandQueue &get_cl_queue(DeviceIndex device);
const CLAllocation &get_alloc(const at::Tensor &tensor);

inline void check_device_index(DeviceIndex device)
{
    TORCH_CHECK(
        device >= 0 && device < device_count(),
        "Invalid OpenCL device index: ",
        device,
        ", available devices: ",
        device_count()
    );
}

} // namespace c10::opencl
