#include "runtime/OpenCLFunctions.h"
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

static c10::once_flag g_init_flag;
static std::vector<DeviceContext> g_devices;
static std::vector<cl::Context> g_contexts;

static thread_local DeviceIndex tl_current_device = 0;

static void initOpenCLDevices() {
    std::vector<cl::Platform> platforms;
    OPENCL_CHECK(cl::Platform::get(&platforms));
    TORCH_CHECK(!platforms.empty(), "No OpenCL platforms found");

    for (auto &platform : platforms) {
        std::vector<cl::Device> platform_devs;
        OPENCL_CHECK(platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devs));

        g_contexts.emplace_back(platform_devs);
        cl::Context &ctx = g_contexts.back();

        for (auto &dev : platform_devs) {
            g_devices.push_back({dev, &ctx, cl::CommandQueue(ctx, dev)});
        }
    }

    TORCH_CHECK(!g_devices.empty(), "No OpenCL GPU devices found");
}

static void ensureInitialized() {
    c10::call_once(g_init_flag, initOpenCLDevices);
}

DeviceIndex device_count() noexcept {
    ensureInitialized();
    return static_cast<DeviceIndex>(g_devices.size());
}

DeviceIndex current_device() {
    ensureInitialized();
    return tl_current_device;
}

void set_device(DeviceIndex device) {
    ensureInitialized();
    check_device_index(device);
    tl_current_device = device;
}

DeviceIndex maybe_exchange_device(DeviceIndex device) {
    ensureInitialized();
    check_device_index(device);
    const DeviceIndex old = tl_current_device;
    if (old != device) {
        tl_current_device = device;
    }
    return old;
}

DeviceIndex ExchangeDevice(DeviceIndex device) {
    ensureInitialized();
    check_device_index(device);
    const DeviceIndex old = tl_current_device;
    tl_current_device = device;
    return old;
}

} // namespace c10::opencl
