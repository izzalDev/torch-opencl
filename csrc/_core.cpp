#include "_core.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif // __APPLE__
#include <vector>

namespace torch_opencl {
bool is_available() {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  for (auto &platform : platforms) {
    cl_uint num_devices = 0;
    cl_int dev_err =
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (dev_err == CL_SUCCESS && num_devices > 0) {
      return true;
    }
  }

  return false;
}
} // namespace torch_opencl
