#include "OpenCLException.h"
#include "CL/opencl.hpp"
#include <fmt/format.h>
#include <string>

namespace c10::opencl {

std::string clErrorMsg(const cl::Error &e) {
  return fmt::format("OpenCL error in {} — code {}", e.what(), e.err());
}

void clCheckFail(const cl::Error &e, const char *file, uint32_t line,
                 const char *func) {
  throw c10::Error({func, file, line}, clErrorMsg(e));
}

} // namespace c10::opencl
