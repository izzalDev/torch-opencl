#pragma once
#include <CL/opencl.hpp>
#include <string>
#include <vector>

struct OpenCLDevice {
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;

  std::string name;
  cl_ulong total_memory;
  cl_uint compute_units;
  cl_ulong l2_cache_size;
  int major;
  int minor;
};

class OpenCLContext {
public:
  static OpenCLContext &instance();
  bool is_available() const;
  int device_count() const;
  const std::string device_name(int index);
  OpenCLDevice &get(int index);
  const OpenCLDevice &get(int index) const;
  OpenCLContext(const OpenCLContext &) = delete;
  OpenCLContext &operator=(const OpenCLContext &) = delete;
  void set_device(int index);
  int current_device() const;

private:
  OpenCLContext();
  void check_index(int index) const;
  std::vector<OpenCLDevice> devices_;
  int current_device_index_ = 0;
};
