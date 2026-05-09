#include "runtime/OpenCLException.h"

#include <string>

namespace c10::opencl {

std::string clErrorMsg(const cl::Error& e) {
  return std::string("OpenCL error in ") + e.what() + " — code " + std::to_string(e.err());
}

void clCheckFail(const cl::Error& e, const char* file, uint32_t line, const char* func) {
  throw c10::Error({func, file, line}, clErrorMsg(e));
}

}  // namespace c10::opencl
