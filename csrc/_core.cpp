#include "_core.h"
#include <CL/opencl.hpp>
#include <vector>

bool is_available() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (auto &platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (!devices.empty()) {
      return true;
    }
  }
  return false;
}

int device_count() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  int count = 0;
  for (auto &platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    count += static_cast<int>(devices.size());
  }
  return count;
}
