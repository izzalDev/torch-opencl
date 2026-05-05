#include "runtime/OpenCLFunctions.h"
#include "CL/cl.h"
#include "runtime/OpenCLException.h"
#include <CL/opencl.hpp>
#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>
#include <vector>

namespace c10::opencl {

struct DeviceContext {
  cl::Device device;
  cl::Context *context;
  cl::CommandQueue queue;
};

c10::once_flag g_init_flag;
std::vector<DeviceContext> g_devices;
std::vector<cl::Context> g_context;

thread_local DeviceIndex tl_current_device = 0;

void doInitOpenCLDevices() {
  std::vector<cl::Platform> platforms;

  OPENCL_CHECK(cl::Platform::get(&platforms));
  TORCH_CHECK(!platforms.empty(), "No OpenCL platforms found")

  for (auto &platform : platforms) {
    std::vector<cl::Device> platform_devs;
    OPENCL_CHECK(platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devs));
    cl::Context ctx(platform_devs);
    g_context.push_back(ctx);
    for (auto &d : platform_devs) {
      DeviceContext dctx;
      cl::CommandQueue queue;
      dctx.device = d;
      dctx.context = &g_context.back();
      dctx.queue = cl::CommandQueue(g_context.back(), d);
      g_devices.push_back(dctx);
    }
  }

  TORCH_CHECK(!g_devices.empty(), "No OpenCL devices found")
}

DeviceIndex device_count() noexcept {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
  const auto count = static_cast<DeviceIndex>(g_devices.size());
  return count;
}

DeviceIndex current_device() {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
  return tl_current_device;
}

void set_device(DeviceIndex device) {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
  check_device_index(device);
  tl_current_device = device;
}

DeviceIndex exchange_device(DeviceIndex device) {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
  check_device_index(device);
  const DeviceIndex old = tl_current_device;
  tl_current_device = device;
  return old;
}

DeviceIndex maybe_exchange_device(DeviceIndex device) {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
  check_device_index(device);
  const DeviceIndex old = tl_current_device;
  tl_current_device = device;
  if (old == device) {
    return old;
  }

  tl_current_device = device;
  return old;
}
} // namespace c10::opencl
