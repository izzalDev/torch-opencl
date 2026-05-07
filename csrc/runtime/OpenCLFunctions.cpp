// csrc/runtime/OpenCLFunctions.cpp

#include "runtime/OpenCLFunctions.h"

#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>

#include <CL/opencl.hpp>
#include <memory>
#include <vector>

#include "runtime/OpenCLException.h"

namespace c10::opencl {

struct DeviceContext {
    cl::Device device;
    std::shared_ptr<cl::Context> context;
    cl::CommandQueue queue;
};

static c10::once_flag g_init_flag;
static std::vector<DeviceContext> g_devices;

static thread_local DeviceIndex tl_current_device = 0;

static void initOpenCLDevices() {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error&) {
        return;
    }

    for (auto& platform : platforms) {
        std::vector<cl::Device> platform_devs;
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devs);
        } catch (const cl::Error&) {
            continue;
        }

        if (platform_devs.empty()) continue;

        auto ctx = std::make_shared<cl::Context>(platform_devs);

        for (auto& dev : platform_devs) {
            g_devices.push_back({dev, ctx, cl::CommandQueue(*ctx, dev)});
        }
    }
}

static void ensureInitialized() {
    c10::call_once(g_init_flag, initOpenCLDevices);
}

DeviceIndex device_count() noexcept {
    try {
        ensureInitialized();
    } catch (...) {
        return 0;
    }
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

}  // namespace c10::opencl
