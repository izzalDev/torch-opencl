#pragma once

#ifdef _WIN32
#define OPENCL_EXPORT __declspec(dllexport)
#else
#define OPENCL_EXPORT __attribute__((visibility("default")))
#endif
