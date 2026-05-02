#include <nanobind/nanobind.h>

namespace nb = nanobind;

int add(int a, int b) {
    return a + b;
}

NB_MODULE(_C, m) {
    m.doc() = "Dummy torch-opencl module";

    m.def("add", &add, "Add two integers");
}
