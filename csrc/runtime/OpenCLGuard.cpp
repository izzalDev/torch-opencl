#include "OpenCLGuard.h"
#include "OpenCLFunctions.h"

namespace c10::opencl {

DeviceType OpenCLGuardImpl::type() const {
  return c10::DeviceType::PrivateUse1;
}

Device OpenCLGuardImpl::exchangeDevice(Device d) const {
  Device prev(type(), current_device());
  set_device(d.index());
  return prev;
}

Device OpenCLGuardImpl::getDevice() const {
  return Device(type(), current_device());
}

void OpenCLGuardImpl::setDevice(Device d) const {
  set_device(d.index());
}

void OpenCLGuardImpl::uncheckedSetDevice(Device d) const noexcept {
  set_device(d.index());
}

c10::Stream OpenCLGuardImpl::getStream(Device d) const noexcept {
  return c10::Stream(c10::Stream::UNSAFE, d, 0);
}

c10::Stream OpenCLGuardImpl::exchangeStream(c10::Stream) const noexcept {
  return getStream(getDevice());
}

DeviceIndex OpenCLGuardImpl::deviceCount() const noexcept {
  return device_count();
}

bool OpenCLGuardImpl::queryStream(const c10::Stream &) const {
  return false;
}

void OpenCLGuardImpl::synchronizeStream(const c10::Stream &stream) const {
  get_queue(stream.device().index()).finish();
}

} // namespace c10::opencl
