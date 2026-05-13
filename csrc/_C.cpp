#include <nanobind/nanobind.h>

#include <c10/core/Device.h>

#include "runtime/CLFunctions.h"
#include "runtime/CLGuard.h"

namespace {

c10::opencl::CLGuardImpl g_guard;

c10::DeviceIndex get_device() { return g_guard.getDevice().index(); }

void set_device(c10::DeviceIndex device)
{
    g_guard.setDevice(c10::Device(c10::DeviceType::PrivateUse1, device));
}

c10::DeviceIndex exchange_device(c10::DeviceIndex device)
{
    return g_guard.exchangeDevice(c10::Device(c10::DeviceType::PrivateUse1, device)).index();
}

} // namespace

NB_MODULE(_C, m)
{
    m.doc() = "torch_opencl C extension module for PyTorch OpenCL backend.";

    m.def("is_in_bad_fork", &c10::opencl::is_in_bad_fork);
    m.def("init", &c10::opencl::ensure_initialized);

    m.def("get_device_count", &c10::opencl::device_count);

    m.def("get_device", &get_device);
    m.def("set_device", &set_device);
    m.def("exchange_device", &exchange_device);
}
