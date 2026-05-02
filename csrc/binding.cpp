#include "_core.h"
#include "opencl_hooks.h"
#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
	register_opencl_backend();

  nb::class_<OpenCLDevice>(m, "OpenCLDeviceProperties")
      .def_ro("name", &OpenCLDevice::name)
      .def_ro("total_memory", &OpenCLDevice::total_memory)
      .def_ro("multi_processor_count", &OpenCLDevice::compute_units)
      .def_ro("major", &OpenCLDevice::major)
      .def_ro("minor", &OpenCLDevice::minor)
      .def_ro("l2_cache_size", &OpenCLDevice::l2_cache_size)
      .def("__repr__", [](const OpenCLDevice &d) {
        return fmt::format(
            "OpenCLDeviceProperties(name='{}', major={}, minor={}, "
            "total_memory={}MB, multi_processor_count={}, l2_cache_size={}MB)",
            d.name, d.major, d.minor, d.total_memory / (1 << 20),
            d.compute_units, d.l2_cache_size / (1 << 20));
      });

  m.def("is_available", &is_available);
  m.def("device_count", &device_count);
  m.def("device_name", &device_name, nb::arg("device_index") = 0);
  m.def("current_device", &current_device);
  m.def("set_device", &set_device, nb::arg("device"));
  m.def("synchronize", &synchronize, nb::arg("device_index") = 0);
  m.def("get_device_properties", &get_device_properties,
        nb::arg("device_index") = 0);
}
