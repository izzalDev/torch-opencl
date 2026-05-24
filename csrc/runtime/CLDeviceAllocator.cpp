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

static std::vector<c10::CachingDeviceAllocator::DeviceStats> g_all_stats;
static std::once_flag g_stats_init_flag;

void ensure_stats_initialized()
{
    std::call_once(g_stats_init_flag, []() {
        int count = device_count();
        g_all_stats.resize(count > 0 ? count : 1);
    });
}

c10::CachingDeviceAllocator::DeviceStats &get_stats(int64_t device_index)
{
    ensure_stats_initialized();
    return g_all_stats[device_index];
}

void track_allocation(int64_t device_index, size_t nbytes)
{
    auto &stats = get_stats(device_index);
    stats.num_device_alloc++;
    stats.allocated_bytes[0].current += nbytes;
    stats.reserved_bytes[0].current += nbytes;

    if (stats.allocated_bytes[0].current > stats.allocated_bytes[0].peak) {
        stats.allocated_bytes[0].peak = stats.allocated_bytes[0].current;
        stats.reserved_bytes[0].peak = stats.allocated_bytes[0].current;
    }
}

void track_deallocation(int64_t device_index, size_t nbytes)
{
    auto &stats = get_stats(device_index);
    stats.allocated_bytes[0].current -= nbytes;
    stats.reserved_bytes[0].current -= nbytes;
    stats.num_device_free++;
}

void deleteHandle(void *ptr)
{
    auto *alloc = static_cast<CLAllocation *>(ptr);
    if (alloc) {
        track_deallocation(alloc->device, alloc->size);
        delete alloc;
    }
}

} // namespace

at::DataPtr CLDeviceAllocator::allocate(size_t nbytes)
{
    c10::impl::VirtualGuardImpl guard(at::kPrivateUse1);
    const auto device_index = guard.getDevice().index();

    TORCH_CHECK(
        device_index >= 0 &&
            device_index < static_cast<int64_t>(device_count()),
        "allocate(): invalid device_index: ",
        device_index
    );

    if (nbytes == 0) {
        auto handle =
            std::make_unique<CLAllocation>(cl::Buffer{}, device_index, 0);
        auto *raw = handle.release();
        return {
            raw, raw, &deleteHandle, at::Device(at::kPrivateUse1, device_index)
        };
    }

    const auto &context = get_cl_context(device_index);

    cl_int err = CL_SUCCESS;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, nbytes, nullptr, &err);

    if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
        auto &stats = get_stats(device_index);
        stats.num_ooms++;
    }

    TORCH_CHECK(
        err == CL_SUCCESS,
        "OpenCL clCreateBuffer failed with error code: ",
        err,
        ". Potential Out of Memory (OOM) on GPU."
    );

    track_allocation(device_index, nbytes);

    auto handle =
        std::make_unique<CLAllocation>(std::move(buffer), device_index, nbytes);
    auto *raw = handle.release();

    return {
        raw, raw, &deleteHandle, at::Device(at::kPrivateUse1, device_index)
    };
}

at::DeleterFnPtr CLDeviceAllocator::raw_deleter() const
{
    return &deleteHandle;
}

void CLDeviceAllocator::copy_data(
    void *dest, const void *src, std::size_t count
) const
{
    const auto *src_handle = static_cast<const CLAllocation *>(src);
    auto *dest_handle = static_cast<CLAllocation *>(dest);

    TORCH_CHECK(
        src_handle->device == dest_handle->device,
        "copy_data: peer-to-peer copy not supported"
    );

    const auto &queue = get_cl_queue(src_handle->device);
    cl_int err = queue.enqueueCopyBuffer(
        src_handle->buffer, dest_handle->buffer, 0, 0, count
    );
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
    return get_stats(device_index);
}

void CLDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex device_index)
{
    auto &stats = get_stats(device_index);
    stats.num_device_alloc = 0;
    stats.num_device_free = 0;
}

void CLDeviceAllocator::resetPeakStats(c10::DeviceIndex device_index)
{
    auto &stats = get_stats(device_index);
    stats.allocated_bytes[0].peak = stats.allocated_bytes[0].current;
    stats.reserved_bytes[0].peak = stats.allocated_bytes[0].current;
}

} // namespace c10::opencl

static c10::opencl::CLDeviceAllocator global_cl_allocator;

static bool register_allocator [[maybe_unused]] = []() {
    at::SetAllocator(at::kPrivateUse1, &global_cl_allocator);
    return true;
}();
