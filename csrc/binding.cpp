#include "_core.h"
#include <nanobind/nanobind.h>

NB_MODULE(_core, m) {
  m.doc() = "OpenCL backend untuk PyTorch, API mirip torch.cuda";
  m.def("is_available", &is_available,
        "Kembalika True jika minimal ada satu GPU OpenCL yang tersedia.");
}
