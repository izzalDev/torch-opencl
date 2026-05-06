#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace c10::opencl {

DeviceIndex device_count() noexcept;
DeviceIndex current_device();
void set_device(DeviceIndex device);
DeviceIndex maybe_exchange_device(DeviceIndex to_device);
DeviceIndex ExchangeDevice(DeviceIndex device);

static inline void check_device_index(int64_t device) {
  TORCH_CHECK(device >= 0 && device < c10::opencl::device_count(),
              "The device index is out of range. It must be in [0, ",
              static_cast<int>(c10::opencl::device_count()), "), but got ",
              static_cast<int>(device), ".");
}

} // namespace c10::opencl
