#include <nanobind/nanobind.h>
#include <runtime/OpenCLFunctions.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "OpenCL backend untuk PyTorch, API mirip torch.cuda";
	m.def("_device_count", &c10::opencl::device_count);
}
