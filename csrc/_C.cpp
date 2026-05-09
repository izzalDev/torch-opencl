#include <c10/core/Device.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "runtime/OpenCLFunctions.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_C, m) {
  m.doc() = "torch_opencl C extension module for PyTorch OpenCL backend.";

  m.def("is_in_bad_fork", &c10::opencl::is_in_bad_fork);
  m.def("init", &c10::opencl::ensure_initialized);
  m.def("get_device_count", &c10::opencl::device_count);
  m.def("get_device", &c10::opencl::current_device);
  m.def("set_device", &c10::opencl::set_device);
  m.def("exchange_device", &c10::opencl::exchange_device);
  m.def("maybe_exchange_device", &c10::opencl::maybe_exchange_device);
}
