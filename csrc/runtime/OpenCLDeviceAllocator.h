#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>

#include <CL/opencl.hpp>

namespace c10::opencl {

// ---------------------------------------------------------------------
// Opaque OpenCL allocation handle
// Stored inside at::DataPtr
// ---------------------------------------------------------------------
struct OpenCLBufferHandle {
  cl::Buffer buffer;
  c10::DeviceIndex device;
  size_t size;
};

// ---------------------------------------------------------------------
// OpenCL device allocator
// ---------------------------------------------------------------------
class OpenCLDeviceAllocator final : public c10::DeviceAllocator {
 public:
  OpenCLDeviceAllocator() = default;
  ~OpenCLDeviceAllocator() override = default;

  OpenCLDeviceAllocator(const OpenCLDeviceAllocator &) = delete;
  OpenCLDeviceAllocator &operator=(const OpenCLDeviceAllocator &) = delete;

  // Allocate device memory
  at::DataPtr allocate(size_t nbytes) override;

  // Raw deleter used by DataPtr
  at::DeleterFnPtr raw_deleter() const override;

  // Generic byte copy helper
  void copy_data(void *dest, const void *src, std::size_t count) const override;

  // Device allocator state
  bool initialized() override;

  // Cache management (no-op for now)
  void emptyCache(MempoolId_t mempool_id = {0, 0}) override;

  // Stream tracking (no-op for now)
  void recordStream(const at::DataPtr &ptr, c10::Stream stream) override;

  // Memory statistics (empty for now)
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override;

  void resetAccumulatedStats(c10::DeviceIndex device) override;

  void resetPeakStats(c10::DeviceIndex device) override;
};

}  // namespace c10::opencl
