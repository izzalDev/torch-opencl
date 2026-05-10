#pragma once

#include <c10/core/Allocator.h>

#include <CL/opencl.hpp>
#include <mutex>
#include <unordered_map>

namespace c10::opencl {

struct BufferEntry {
  cl::Buffer buffer;
  size_t size;
  DeviceIndex device;
};

class OpenCLAllocator : public at::Allocator {
 public:
  explicit OpenCLAllocator(DeviceIndex device) : device_(device) {}

  at::DataPtr allocate(size_t size) override;
  at::DeleterFnPtr raw_deleter() const override { return &OpenCLAllocator::Delete; }
  void copy_data(void* dest, const void* src, std::size_t count) const override;

 private:
  static void Delete(void* ctx);

  DeviceIndex device_;
  std::unordered_map<void*, BufferEntry*> buffers_;
  mutable std::mutex mutex_;
};

OpenCLAllocator* getOpenCLAllocator(DeviceIndex device);

}  // namespace c10::opencl
