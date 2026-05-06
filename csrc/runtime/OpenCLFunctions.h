#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Macros.h>

namespace c10::opencl {

DeviceIndex device_count() noexcept;
DeviceIndex current_device();
void set_device(DeviceIndex device);
DeviceIndex maybe_exchange_device(DeviceIndex device);
DeviceIndex ExchangeDevice(DeviceIndex device);

static inline void check_device_index(int64_t device) {
    TORCH_CHECK(
        device >= 0 && device < c10::opencl::device_count(),
        "Device index out of range [0, ", static_cast<int>(device_count()),
        "), got ", static_cast<int>(device));
}

} // namespace c10::opencl
