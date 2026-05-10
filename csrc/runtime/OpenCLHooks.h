#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/StorageImpl.h>

#include "OpenCLDeviceAllocator.h"
#include "OpenCLFunctions.h"

namespace c10::opencl {
struct OpenCLHooksInterface : public at::PrivateUse1HooksInterface {
  OpenCLHooksInterface() {};
  ~OpenCLHooksInterface() override = default;

  void init() const override {
    // Initialize OpenCL runtime if needed
    // This is called when PyTorch first accesses the device
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override { return true; }

  bool isBuilt() const override {
    // This extension is compiled as part of the OpenCL test extension.
    return true;
  }

  bool isAvailable() const override {
    // Consider OpenReg available if there's at least one device reported.
    return device_count() > 0;
  }

  DeviceIndex deviceCount() const override { return device_count(); }

  void setCurrentDevice(DeviceIndex device) const override { set_device(device); }

  DeviceIndex getCurrentDevice() const override { return current_device(); }

  DeviceIndex exchangeDevice(DeviceIndex device) const override { return exchange_device(device); }

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
    // Only exchange if the requested device is valid; otherwise, no-op and return current
    auto count = device_count();
    if (device < 0 || device >= count) {
      return getCurrentDevice();
    }
    return exchangeDevice(device);
  }

  at::Allocator *getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  bool isPinnedPtr(const void *data) const override { return false; }

  at::Device getDeviceFromPtr(void *data) const override {
    TORCH_CHECK(data != nullptr, "getDeviceFromPtr: null pointer");
    auto *entry = static_cast<const BufferEntry *>(data);
    return at::Device(at::DeviceType::PrivateUse1, entry->device);
  }
};

}  // namespace c10::opencl
