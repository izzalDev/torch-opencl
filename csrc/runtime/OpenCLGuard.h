#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10::opencl {

struct OpenCLGuardImpl : public c10::impl::DeviceGuardImplInterface {
  DeviceType type() const override;
  Device exchangeDevice(Device d) const override;
  Device getDevice() const override;
  void setDevice(Device d) const override;
  void uncheckedSetDevice(Device d) const noexcept override;
  c10::Stream getStream(Device d) const noexcept override;
  c10::Stream exchangeStream(c10::Stream) const noexcept override;
  DeviceIndex deviceCount() const noexcept override;
  bool queryStream(const c10::Stream &) const override;
  void synchronizeStream(const c10::Stream &stream) const override;
};

} // namespace c10::opencl
