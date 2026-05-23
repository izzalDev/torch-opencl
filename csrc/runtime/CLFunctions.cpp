#include "runtime/CLFunctions.h"

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

// OpenCL contexts are shared per-platform to enable
// efficient resource sharing and reduce program build overhead.
struct DeviceContext {
    cl::Device device;
    std::shared_ptr<cl::Context> context;
    cl::CommandQueue queue;
};

static c10::once_flag g_init_flag;
static std::vector<DeviceContext> g_devices;
static pid_t g_original_pid = -1;

static void init_opencl_runtime()
{
    g_original_pid = getpid();
    std::vector<DeviceContext> tmp;

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &) {
        return;
    }

    for (auto &platform : platforms) {
        std::vector<cl::Device> platform_devs;
        try {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devs);
        } catch (const cl::Error &) {
            continue;
        }

        if (platform_devs.empty())
            continue;

        std::shared_ptr<cl::Context> ctx;
        try {
            ctx = std::make_shared<cl::Context>(platform_devs);
        } catch (const cl::Error &) {
            continue;
        }

        for (auto &dev : platform_devs) {
            try {
                tmp.push_back({dev, ctx, cl::CommandQueue(*ctx, dev)});
            } catch (const cl::Error &) {
                continue;
            }
        }
    }
    g_devices = std::move(tmp);
}

void ensure_initialized()
{
    TORCH_CHECK(!is_in_bad_fork(), "OpenCL backend cannot be used after fork");
    c10::call_once(g_init_flag, init_opencl_runtime);
}

bool is_in_bad_fork()
{
    return g_original_pid != -1 && getpid() != g_original_pid;
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
