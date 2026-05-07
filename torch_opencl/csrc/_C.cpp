// torch_opencl/csrc/_C.cpp

#include <ATen/Context.h>
#include <nanobind/nanobind.h>
#include <runtime/OpenCLFunctions.h>
#include <torch/csrc/utils/device_lazy_init.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "OpenCL backend for PyTorch — low-level C++ bindings.";

    m.def("_init", []() {
        torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
        at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);
    });

    m.def("_is_in_bad_fork",
          []() { return torch::utils::is_device_in_bad_fork(at::kPrivateUse1); });

    m.def("_get_device_count", []() {
        torch::utils::register_fork_handler_for_device_init(at::kPrivateUse1);
        return c10::opencl::device_count();
    });

    m.def("_get_device", []() { return static_cast<int32_t>(c10::opencl::current_device()); });

    m.def("_set_device",
          [](int32_t device) { c10::opencl::set_device(static_cast<c10::DeviceIndex>(device)); });

    m.def("_exchange_device", [](int32_t device) {
        if (device < 0) return static_cast<int32_t>(-1);
        return static_cast<int32_t>(
            c10::opencl::ExchangeDevice(static_cast<c10::DeviceIndex>(device)));
    });

    m.def("_maybe_exchange_device", [](int32_t device) {
        if (device < 0) return static_cast<int32_t>(-1);
        return static_cast<int32_t>(
            c10::opencl::maybe_exchange_device(static_cast<c10::DeviceIndex>(device)));
    });
}
