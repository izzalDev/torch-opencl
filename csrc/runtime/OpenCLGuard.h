#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "runtime/OpenCLFunctions.h"

namespace c10::opencl {

struct OpenCLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;

    OpenCLGuardImpl() = default;
    explicit OpenCLGuardImpl(DeviceType t)
    {
        TORCH_CHECK(
            t == static_type, "OpenCLGuardImpl initialized with non-PrivateUse1 DeviceType: ", t
        );
    }

    DeviceType type() const override { return static_type; }

    Device exchangeDevice(Device d) const override
    {
        TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);
        auto old = exchange_device(d.index());
        return Device(static_type, old);
    }

    Device getDevice() const override { return Device(static_type, current_device()); }

    void setDevice(Device d) const override
    {
        TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);
        set_device(d.index());
    }

    void uncheckedSetDevice(Device d) const noexcept override { set_device(d.index()); }

    DeviceIndex deviceCount() const noexcept override { return device_count(); }

    void synchronizeDevice(const DeviceIndex device_index) const override
    {
        get_cl_queue(device_index).finish();
    }

    Stream getStream(Device d) const noexcept override { return Stream(Stream::UNSAFE, d, 0); }

    Stream getDefaultStream(Device d) const override { return getStream(d); }

    Stream exchangeStream(Stream s) const noexcept override { return getStream(s.device()); }

    bool queryStream(const Stream &) const override { return false; }

    void synchronizeStream(const Stream &stream) const override
    {
        get_cl_queue(stream.device_index()).finish();
    }
};

} // namespace c10::opencl
