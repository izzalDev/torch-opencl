#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <CL/opencl.hpp>

namespace c10::opencl {

DeviceIndex device_count() noexcept;
bool is_in_bad_fork();
void ensure_initialized();

cl::Context &get_cl_context(DeviceIndex device);
cl::Device &get_cl_device(DeviceIndex device);
cl::CommandQueue &get_cl_queue(DeviceIndex device);

inline void check_device_index(DeviceIndex device)
{
    auto count = device_count();
    TORCH_CHECK(
        device >= 0 && device < count,
        "Invalid OpenCL device index: ",
        device,
        " (total devices: ",
        count,
        ")"
    );
}

} // namespace c10::opencl
