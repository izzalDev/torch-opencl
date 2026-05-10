#include "runtime/OpenCLDeviceAllocator.h"

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <memory>
#include <mutex>
#include <vector>

#include "runtime/OpenCLException.h"
#include "runtime/OpenCLFunctions.h"

namespace c10::opencl {

at::DataPtr OpenCLAllocator::allocate(size_t size) {
  if (size == 0) {
    return at::DataPtr(nullptr, nullptr, &OpenCLAllocator::Delete,
                       at::Device(at::DeviceType::PrivateUse1, device_));
  }

  cl::Buffer buffer;
  OPENCL_CHECK(buffer = cl::Buffer(get_cl_context(device_), CL_MEM_READ_WRITE, size));

  auto *entry = new BufferEntry{std::move(buffer), size, device_};

  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[static_cast<void *>(entry)] = entry;
  }

  return at::DataPtr(static_cast<void *>(entry), static_cast<void *>(entry),
                     &OpenCLAllocator::Delete, at::Device(at::DeviceType::PrivateUse1, device_));
}

void OpenCLAllocator::Delete(void *ctx) {
  if (ctx == nullptr) return;

  auto *entry = static_cast<BufferEntry *>(ctx);
  DeviceIndex device = entry->device;

  OpenCLAllocator *allocator = getOpenCLAllocator(device);

  {
    std::lock_guard<std::mutex> lock(allocator->mutex_);
    allocator->buffers_.erase(ctx);
    delete entry;
  }
}

void OpenCLAllocator::copy_data(void *dest, const void *src, std::size_t count) const {
  if (count == 0) return;

  TORCH_CHECK(dest != nullptr && src != nullptr, "copy_data: null pointer");

  auto *src_entry = static_cast<const BufferEntry *>(src);
  auto *dest_entry = static_cast<BufferEntry *>(dest);

  TORCH_CHECK(src_entry->device == dest_entry->device, "copy_data: source device (",
              src_entry->device, ") != dest device (", dest_entry->device,
              "). Cross-device copy not supported here.");

  TORCH_CHECK(count <= src_entry->size, "copy_data: count (", count,
              ") exceeds source buffer size (", src_entry->size, ")");
  TORCH_CHECK(count <= dest_entry->size, "copy_data: count (", count,
              ") exceeds dest buffer size (", dest_entry->size, ")");

  cl::CommandQueue &queue = get_cl_queue(device_);
  OPENCL_CHECK(queue.enqueueCopyBuffer(src_entry->buffer, dest_entry->buffer, 0, 0, count));
  OPENCL_CHECK(queue.finish());
}

static std::vector<std::unique_ptr<OpenCLAllocator>> g_allocators;
static std::once_flag g_allocator_init_flag;

static void init_allocators() {
  ensure_initialized();
  DeviceIndex count = device_count();
  g_allocators.reserve(count);
  for (DeviceIndex i = 0; i < count; ++i) {
    g_allocators.push_back(std::make_unique<OpenCLAllocator>(i));
  }
}

OpenCLAllocator *getOpenCLAllocator(DeviceIndex device) {
  std::call_once(g_allocator_init_flag, init_allocators);

  TORCH_CHECK(device >= 0 && device < static_cast<DeviceIndex>(g_allocators.size()),
              "getOpenCLAllocator: invalid device index ", device);

  return g_allocators[device].get();
}

}  // namespace c10::opencl
