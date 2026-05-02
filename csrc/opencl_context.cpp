#include "opencl_context.h"
#include <CL/opencl.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

OpenCLContext &OpenCLContext::instance() {
  static OpenCLContext ctx;
  return ctx;
}

OpenCLContext::OpenCLContext() {
  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (...) {
    return;
  }

  for (auto &platform : platforms) {
    std::vector<cl::Device> devs;
    try {
      platform.getDevices(CL_DEVICE_TYPE_GPU, &devs);
    } catch (...) {
      continue;
    }
    for (auto &dev : devs) {
      try {
        cl::Context ctx(dev);
        cl::CommandQueue queue(ctx, dev);

        OpenCLDevice d;
        std::string version = dev.getInfo<CL_DEVICE_VERSION>();
        std::sscanf(version.c_str(), "OpenCL %d.%d", &d.major, &d.minor);
        d.index = static_cast<int>(devices_.size());
        d.name = dev.getInfo<CL_DEVICE_NAME>();
        d.total_memory = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        d.compute_units = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        d.l2_cache_size = dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
        d.device = std::move(dev);
        d.context = std::move(ctx);
        d.queue = std::move(queue);
        devices_.push_back(std::move(d));
      } catch (...) {
        continue;
      }
    }
  }
}

void OpenCLContext::check_index(int index) const {
  if (index < 0 || index >= static_cast<int>(devices_.size()))
    throw std::out_of_range(
        "opencl device index " + std::to_string(index) +
        " out of range (device_count=" + std::to_string(devices_.size()) + ")");
}

bool OpenCLContext::is_available() const {
  return !devices_.empty();
}

int OpenCLContext::device_count() const {
  return static_cast<int>(devices_.size());
}

const std::string OpenCLContext::device_name(int index) {
  check_index(index);
  return devices_[index].name;
}

void OpenCLContext::set_device(int index) {
  check_index(index);
  current_device_index_ = index;
}

OpenCLDevice& OpenCLContext::current_device(){
  return devices_[current_device_index_];
}

const OpenCLDevice OpenCLContext::current_device() const {
  return devices_[current_device_index_];
}
