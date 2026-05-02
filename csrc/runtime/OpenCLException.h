#pragma once

#include <CL/opencl.hpp>
#include <c10/util/Exception.h>

void clCheckFail(
    const char *func, const char *file, uint32_t line, const char *msg = "");

// Wraps a cl::Error-throwing expression into a c10::Error.
#define OPENCL_CHECK(EXPR)                                                     \
  do {                                                                         \
    try {                                                                      \
      (EXPR);                                                                  \
    } catch (const cl::Error &_e) {                                            \
      clCheckFail(__func__, __FILE__, static_cast<uint32_t>(__LINE__),         \
          (std::string(#EXPR) + ": " + _e.what() + " (" +                      \
              std::to_string(_e.err()) + ")")                                  \
              .c_str());                                                       \
    }                                                                          \
  } while (0)
