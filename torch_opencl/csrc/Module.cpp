#include <nanobind/nanobind.h>
#include <runtime/OpenCLFunctions.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "OpenCL backend for PyTorch — API similar to torch.cuda";
    m.def("device_count", &c10::opencl::device_count,
          "Return the number of available OpenCL GPU devices.");
}
