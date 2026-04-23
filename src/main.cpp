#include <nanobind/nanobind.h>
#include <torch/torch.h>

namespace nb = nanobind;

int add(int a, int b) { return a + b; }

NB_MODULE(_core, m) {
    m.doc() = "Core extension module built with nanobind and PyTorch.";

    m.def("add", &add, nb::arg("a"), nb::arg("b"),
          "Add two integers.\n\n"
          ":param a: First integer.\n"
          ":param b: Second integer.\n"
          ":return: Sum of a and b.");
}
