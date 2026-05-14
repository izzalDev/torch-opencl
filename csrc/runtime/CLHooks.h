#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"
#include "runtime/CLGuard.h"

namespace c10::opencl {

struct OpenCLHooksInterface : public at::PrivateUse1HooksInterface {
    OpenCLHooksInterface() = default;
    ~OpenCLHooksInterface() override = default;

    void init() const override { ensure_initialized(); }

    // Return whether the specified device has an initialized primary context.
    bool hasPrimaryContext(DeviceIndex device_index) const override
    {
        return device_index >= 0 && device_index < device_count();
    }

    // Return whether the backend was compiled into the current build.
    bool isBuilt() const override { return true; }

    bool isAvailable() const override { return device_count() > 0; }

    DeviceIndex deviceCount() const override { return device_count(); }

    void setCurrentDevice(DeviceIndex device) const override
    {
        guard_.setDevice(Device(DeviceType::PrivateUse1, device));
    }

    DeviceIndex getCurrentDevice() const override { return guard_.getDevice().index(); }

    // Exchange the current device and return the previous device index.
    DeviceIndex exchangeDevice(DeviceIndex device) const override
    {
        return guard_.exchangeDevice(Device(DeviceType::PrivateUse1, device)).index();
    }

    // Exchange the device only if the requested index is valid.
    DeviceIndex maybeExchangeDevice(DeviceIndex device) const override
    {
        const auto count = device_count();
        if (device < 0 || device >= count) {
            return getCurrentDevice();
        }
        return exchangeDevice(device);
    }

    // Return the pinned host memory allocator associated with this backend.
    at::Allocator *getPinnedMemoryAllocator() const override
    {
        return at::getHostAllocator(at::kPrivateUse1);
    }

    // Return whether the pointer refers to pinned host memory.
    bool isPinnedPtr(const void *data) const override { return false; }

    // Return the device associated with an allocation pointer.
    at::Device getDeviceFromPtr(void *data) const override
    {
        TORCH_CHECK(data != nullptr, "getDeviceFromPtr: null pointer");
        auto *handle = static_cast<const CLAllocation *>(data);
        return at::Device(at::kPrivateUse1, handle->device);
    }

  private:
    mutable CLGuardImpl guard_;
};

} // namespace c10::opencl
