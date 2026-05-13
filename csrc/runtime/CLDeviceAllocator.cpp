#include "runtime/CLDeviceAllocator.h"

#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <c10/core/Device.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <mutex>
#include <torch/headeronly/core/DeviceType.h>
#include <vector>

#include "runtime/CLFunctions.h"

namespace c10::opencl {

namespace {

struct DeviceStatsTracker {
    int64_t current_allocated{0};
    int64_t peak_allocated{0};
    int64_t num_allocs{0};
    int64_t num_frees{0};
    int64_t num_ooms{0};
};

/**
 * Provides access to the per-device statistics tracker.
 * Uses lazy initialization to match the actual OpenCL device count.
 */
DeviceStatsTracker &get_tracker(int64_t device_index)
{
    static std::vector<DeviceStatsTracker> trackers;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
        int count = device_count();
        trackers.resize(count > 0 ? count : 1);
    });

    return trackers.at(device_index);
}

/**
 * Deleter function for DataPtr.
 * Decrements allocation tracking as memory is returned to the driver.
 */
void deleteHandle(void *ptr)
{
    auto *alloc = static_cast<CLAllocation *>(ptr);
    if (alloc) {
        auto &tracker = get_tracker(alloc->device);
        tracker.current_allocated -= alloc->size;
        tracker.num_frees++;
        delete alloc;
    }
}

} // namespace

at::DataPtr CLDeviceAllocator::allocate(size_t nbytes)
{
    c10::impl::VirtualGuardImpl guard(at::kPrivateUse1);
    const auto device_index = guard.getDevice().index();

    TORCH_CHECK(
        device_index >= 0 && device_index < static_cast<int64_t>(device_count()),
        "allocate(): invalid device_index: ",
        device_index
    );

    auto &tracker = get_tracker(device_index);

    if (nbytes == 0) {
        auto handle = std::make_unique<CLAllocation>(cl::Buffer{}, device_index, 0);
        auto *raw = handle.release();
        return {raw, raw, &deleteHandle, at::Device(at::kPrivateUse1, device_index)};
    }

    cl::Context &context = get_cl_context(device_index);

    cl_int err = CL_SUCCESS;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, nbytes, nullptr, &err);

    if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        tracker.num_ooms++;

    TORCH_CHECK(
        err == CL_SUCCESS,
        "OpenCL clCreateBuffer failed with error code: ",
        err,
        ". Potential Out of Memory (OOM) on GPU."
    );

    tracker.num_allocs++;
    tracker.current_allocated += nbytes;

    if (tracker.current_allocated > tracker.peak_allocated)
        tracker.peak_allocated = tracker.current_allocated;

    auto handle = std::make_unique<CLAllocation>(std::move(buffer), device_index, nbytes);
    auto *raw = handle.release();

    return {raw, raw, &deleteHandle, at::Device(at::kPrivateUse1, device_index)};
}

at::DeleterFnPtr CLDeviceAllocator::raw_deleter() const { return &deleteHandle; }

void CLDeviceAllocator::copy_data(void *dest, const void *src, std::size_t count) const
{
    const auto *src_handle = static_cast<const CLAllocation *>(src);
    auto *dest_handle = static_cast<CLAllocation *>(dest);

    TORCH_CHECK(
        src_handle->device == dest_handle->device, "copy_data: peer-to-peer copy not supported"
    );

    cl::CommandQueue &queue = get_cl_queue(src_handle->device);
    cl_int err = queue.enqueueCopyBuffer(src_handle->buffer, dest_handle->buffer, 0, 0, count);
    TORCH_CHECK(err == CL_SUCCESS, "clEnqueueCopyBuffer failed");

    queue.finish();
}

bool CLDeviceAllocator::initialized() { return device_count() > 0; }

void CLDeviceAllocator::emptyCache(MempoolId_t)
{
    // No-op: caching is not implemented yet.
}

void CLDeviceAllocator::recordStream(const at::DataPtr &, c10::Stream)
{
    // No-op: using synchronous command queues for now.
}

c10::CachingDeviceAllocator::DeviceStats
CLDeviceAllocator::getDeviceStats(c10::DeviceIndex device_index)
{
    c10::CachingDeviceAllocator::DeviceStats stats;
    auto &tracker = get_tracker(device_index);

    int64_t current_val = tracker.current_allocated;
    int64_t peak_val = tracker.peak_allocated;

    stats.allocated_bytes[0].current = current_val;
    stats.allocated_bytes[0].peak = peak_val;
    stats.num_device_alloc = tracker.num_allocs;
    stats.num_device_free = tracker.num_frees;

    stats.reserved_bytes[0].current = current_val;
    stats.reserved_bytes[0].peak = peak_val;
    stats.num_ooms = tracker.num_ooms;

    return stats;
}

void CLDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex device_index)
{
    auto &tracker = get_tracker(device_index);
    tracker.num_allocs = 0;
    tracker.num_frees = 0;
}

void CLDeviceAllocator::resetPeakStats(c10::DeviceIndex device_index)
{
    auto &tracker = get_tracker(device_index);
    tracker.peak_allocated = tracker.current_allocated;
}

} // namespace c10::opencl

static c10::opencl::CLDeviceAllocator global_cl_allocator;

static bool register_allocator [[maybe_unused]] = []() {
    at::SetAllocator(at::kPrivateUse1, &global_cl_allocator);
    return true;
}();
