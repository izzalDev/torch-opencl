#include <nanobind/nanobind.h>
#include <_core.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
  m.doc() = "OpenCL backend untuk PyTorch, API mirip torch.cuda";
  m.def("is_available", &torch_opencl::is_available,
        "Kembalikan True jika minimal ada satu GPU OpenCL yang tersedia.");
}
