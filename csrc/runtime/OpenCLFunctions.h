#pragma once

#include <CL/opencl.hpp>

#include "Macros.h"
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

namespace c10::opencl {

struct DeviceContext {
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
};

OPENCL_EXPORT void initOpenCLDevices();
OPENCL_EXPORT DeviceIndex device_count() noexcept;
OPENCL_EXPORT DeviceIndex current_device();
OPENCL_EXPORT void set_device(DeviceIndex device);
OPENCL_EXPORT DeviceIndex exchange_device(DeviceIndex device);

OPENCL_EXPORT const cl::Context &get_context(DeviceIndex device);
OPENCL_EXPORT const cl::Device &get_device(DeviceIndex device);
OPENCL_EXPORT const cl::CommandQueue &get_queue(DeviceIndex device);

// Deklarasi saja — tidak ada inline, tidak ada implementasi
OPENCL_EXPORT void check_device_index(DeviceIndex device);

} // namespace c10::opencl
