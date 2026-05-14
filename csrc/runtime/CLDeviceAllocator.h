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
struct CLAllocation {
    cl::Buffer buffer;
    c10::DeviceIndex device;
    size_t size;
    CLAllocation(cl::Buffer buf, c10::DeviceIndex dev, size_t sz)
        : buffer(std::move(buf)), device(dev), size(sz)
    {
    }
};

// ---------------------------------------------------------------------
// OpenCL device allocator
// ---------------------------------------------------------------------
class CLDeviceAllocator final : public c10::DeviceAllocator {
  public:
    CLDeviceAllocator() = default;
    ~CLDeviceAllocator() override = default;

    CLDeviceAllocator(const CLDeviceAllocator &) = delete;
    CLDeviceAllocator &operator=(const CLDeviceAllocator &) = delete;

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

    // Memory statistics
    c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override;

    void resetAccumulatedStats(c10::DeviceIndex device) override;

    void resetPeakStats(c10::DeviceIndex device) override;
};

} // namespace c10::opencl
