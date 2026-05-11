#include "runtime/OpenCLFunctions.h"

#include <c10/core/Device.h>
#include <c10/util/CallOnce.h>

#include <CL/opencl.hpp>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
typedef int pid_t;
#define getpid _getpid
#endif

namespace c10::opencl {

struct DeviceContext {
    cl::Device device;
    std::shared_ptr<cl::Context> context;
    cl::CommandQueue queue;
};

static c10::once_flag g_init_flag;
static std::vector<DeviceContext> g_devices;
static pid_t g_original_pid = -1;

static thread_local DeviceIndex tl_current_device = 0;

static void init_opencl_device()
{
    g_original_pid = getpid();

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &) {
        return;
    }

    for (auto &platform : platforms) {
        std::vector<cl::Device> platform_devs;
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devs);
        } catch (const cl::Error &) {
            continue;
        }

        if (platform_devs.empty())
            continue;

        auto ctx = std::make_shared<cl::Context>(platform_devs);

        for (auto &dev : platform_devs) {
            try {
                g_devices.push_back({dev, ctx, cl::CommandQueue(*ctx, dev)});
            } catch (...) {
                continue;
            }
        }
    }
}

void ensure_initialized() { c10::call_once(g_init_flag, init_opencl_device); }

bool is_in_bad_fork()
{
    if (g_original_pid != -1 && getpid() != g_original_pid) {
        return true;
    }
    return false;
}

DeviceIndex device_count() noexcept
{
    try {
        ensure_initialized();
    } catch (...) {
        return 0;
    }
    return static_cast<DeviceIndex>(g_devices.size());
}

DeviceIndex current_device()
{
    ensure_initialized();
    return tl_current_device;
}

void set_device(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    tl_current_device = device;
}

DeviceIndex maybe_exchange_device(DeviceIndex device)
{
    ensure_initialized();
    if (device < 0)
        return tl_current_device;
    check_device_index(device);
    const DeviceIndex old = tl_current_device;
    if (old != device) {
        tl_current_device = device;
    }
    return old;
}

DeviceIndex exchange_device(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    const DeviceIndex old = tl_current_device;
    tl_current_device = device;
    return old;
}

cl::Context &get_cl_context(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return *g_devices[device].context;
}

cl::Device &get_cl_device(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return g_devices[device].device;
}

cl::CommandQueue &get_cl_queue(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return g_devices[device].queue;
}

} // namespace c10::opencl
