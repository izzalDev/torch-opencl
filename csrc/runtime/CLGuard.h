#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10::opencl {

struct CLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;

    CLGuardImpl();

    explicit CLGuardImpl(DeviceType t);

    DeviceType type() const override;

    Device exchangeDevice(Device d) const override;

    Device getDevice() const override;

    void setDevice(Device d) const override;

    void uncheckedSetDevice(Device d) const noexcept override;

    DeviceIndex deviceCount() const noexcept override;

    void synchronizeStream(const Stream &stream) const override;

    void synchronizeDevice(DeviceIndex device_index) const override;

    /**
     * OpenCL backend currently exposes a single implicit queue per device.
     * Stream APIs therefore map to a synthetic default stream.
     */
    Stream getStream(Device d) const noexcept override { return Stream(Stream::UNSAFE, d, 0); }

    /**
     * No dedicated stream state exists.
     * Return the synthetic default stream unchanged.
     */
    Stream exchangeStream(Stream s) const noexcept override { return s; }
};

} // namespace c10::opencl
