#include "OpenCLFunctions.h"
#include "OpenCLException.h"

#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <fmt/core.h>

#include <vector>

namespace c10::opencl {

namespace {

// ----------------------------------------------------------------------------
// Global state
// ----------------------------------------------------------------------------

c10::once_flag g_init_flag;
std::vector<DeviceContext> g_devices;

thread_local DeviceIndex tl_current_device = 0;

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

void doInitOpenCLDevices() {
  std::vector<cl::Platform> platforms;
  OPENCL_CHECK(cl::Platform::get(&platforms));
  TORCH_CHECK(!platforms.empty(), "No OpenCL platforms found");

  std::vector<cl::Device> devs;

  for (const auto &platform : platforms) {
    std::vector<cl::Device> gpu_devices;

    try {
      platform.getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices);
    } catch (const cl::Error &e) {
      if (e.err() == CL_DEVICE_NOT_FOUND)
        continue;
      OPENCL_CHECK(throw e);
    }

    for (const auto &d : gpu_devices) {
      devs.push_back(d);
    }
  }

  TORCH_CHECK(!devs.empty(), "No OpenCL devices found");

  g_devices.reserve(devs.size());

  for (const auto &dev : devs) {
    cl::Context ctx(dev);
    cl::CommandQueue queue(ctx, dev, /*props=*/0);

    std::string name;
    dev.getInfo(CL_DEVICE_NAME, &name);
    TORCH_WARN("OpenCL device found: ", name);

    DeviceContext d;
    d.device = dev;
    d.context = std::move(ctx);
    d.queue = std::move(queue);
    g_devices.push_back(d);
  }
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

void initOpenCLDevices() {
  c10::call_once(g_init_flag, doInitOpenCLDevices);
}

DeviceIndex device_count() noexcept {
  try {
    initOpenCLDevices();
    const auto count = static_cast<DeviceIndex>(g_devices.size());
    return count;
  } catch (const std::exception &e) {
    TORCH_WARN("OpenCL device_count() failed: ", e.what());
    return 0;
  } catch (...) {
    TORCH_WARN("OpenCL device_count() failed: unknown error");
    return 0;
  }
}

DeviceIndex current_device() {
  initOpenCLDevices();
  return tl_current_device;
}

void set_device(DeviceIndex device) {
  initOpenCLDevices();
  check_device_index(device);
  tl_current_device = device;
}

DeviceIndex exchange_device(DeviceIndex device) {
  initOpenCLDevices();
  check_device_index(device);

  const DeviceIndex old = tl_current_device;
  tl_current_device = device;
  return old;
}

const cl::Context &get_context(DeviceIndex device) {
  initOpenCLDevices();
  check_device_index(device);
  return g_devices[device].context;
}

const cl::Device &get_device(DeviceIndex device) {
  initOpenCLDevices();
  check_device_index(device);
  return g_devices[device].device;
}

const cl::CommandQueue &get_queue(DeviceIndex device) {
  initOpenCLDevices();
  check_device_index(device);
  return g_devices[device].queue;
}

void check_device_index(DeviceIndex device) {
  TORCH_CHECK(device >= 0 && device < device_count(),
      fmt::format("OpenCL device index {} out of range [0, {})",
          static_cast<int>(device), static_cast<int>(device_count())));
}

} // namespace c10::opencl
