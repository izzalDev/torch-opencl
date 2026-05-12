#pragma once

#include <c10/util/Exception.h>

#include <CL/opencl.hpp>
#include <cstdint>
#include <string>

namespace c10::opencl {

std::string clErrorMsg(const cl::Error &e);
void clCheckFail(const cl::Error &e, const char *file, uint32_t line, const char *func);

} // namespace c10::opencl

#define OPENCL_CHECK(EXPR)                                                                         \
    do {                                                                                           \
        try {                                                                                      \
            (EXPR);                                                                                \
        } catch (const cl::Error &__e) {                                                           \
            ::c10::opencl::clCheckFail(__e, __FILE__, static_cast<uint32_t>(__LINE__), __func__);  \
        }                                                                                          \
    } while (0)
