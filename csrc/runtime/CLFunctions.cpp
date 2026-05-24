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
    const cl::Device device;
    const std::shared_ptr<const cl::Context> context;
    const cl::CommandQueue queue;
};

static c10::once_flag g_init_flag;
static std::vector<DeviceContext> g_devices;
static pid_t g_original_pid = -1;

cl_int init_opencl_runtime()
{
    g_original_pid = getpid();
    std::vector<DeviceContext> tmp;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS)
        return err;

    for (auto &platform : platforms) {
        std::vector<cl::Device> platform_devs;
        err = platform.getDevices(CL_DEVICE_TYPE_ALL, &platform_devs);
        if (err != CL_SUCCESS || platform_devs.empty())
            continue;

        auto ctx = std::make_shared<cl::Context>(
            platform_devs, nullptr, nullptr, nullptr, &err
        );
        if (err != CL_SUCCESS)
            continue;

        for (auto &dev : platform_devs) {
            cl::CommandQueue queue(*ctx, dev, 0, &err);
            if (err != CL_SUCCESS)
                continue;

            DeviceContext dc{dev, ctx, queue};
            tmp.push_back(std::move(dc));
        }
    }

    g_devices = std::move(tmp);
    return CL_SUCCESS;
}

void ensure_initialized()
{
    TORCH_CHECK(!is_in_bad_fork(), "OpenCL backend cannot be used after fork");

    c10::call_once(g_init_flag, []() { init_opencl_runtime(); });
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

const cl::Context &get_cl_context(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return *g_devices[device].context;
}

const cl::Device &get_cl_device(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return g_devices[device].device;
}

const cl::CommandQueue &get_cl_queue(DeviceIndex device)
{
    ensure_initialized();
    check_device_index(device);
    return g_devices[device].queue;
}

const c10::opencl::CLAllocation &get_alloc(const at::Tensor &tensor)
{
    return *static_cast<const c10::opencl::CLAllocation *>(
        tensor.storage().data_ptr().get()
    );
}

} // namespace c10::opencl
