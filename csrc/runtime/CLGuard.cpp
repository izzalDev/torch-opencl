#include "runtime/CLGuard.h"
#include "runtime/CLFunctions.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10::opencl {

namespace {

// Thread-local current device state used by DeviceGuard semantics.
thread_local DeviceIndex tl_current_device = 0;

} // namespace

CLGuardImpl::CLGuardImpl() = default;

// Construct a guard implementation bound to PrivateUse1.
CLGuardImpl::CLGuardImpl(DeviceType t)
{
    TORCH_CHECK(t == static_type, "CLGuardImpl initialized with non-PrivateUse1 DeviceType: ", t);
}

// Return the backend device type handled by this guard implementation.
DeviceType CLGuardImpl::type() const { return static_type; }

// Exchange the current thread-local device and return the previous device.
Device CLGuardImpl::exchangeDevice(Device d) const
{
    TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);
    check_device_index(d.index());
    const auto old = tl_current_device;
    tl_current_device = d.index();
    return Device(static_type, old);
}

// Return the current thread-local device.
Device CLGuardImpl::getDevice() const
{
    check_device_index(tl_current_device);
    return Device(static_type, tl_current_device);
}

// Set the current thread-local device with validation.
void CLGuardImpl::setDevice(Device d) const
{
    TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);
    check_device_index(d.index());
    tl_current_device = d.index();
}

// Set the current device without validation checks.
void CLGuardImpl::uncheckedSetDevice(Device d) const noexcept { tl_current_device = d.index(); }

// Return the number of available OpenCL devices.
DeviceIndex CLGuardImpl::deviceCount() const noexcept { return device_count(); }

// Synchronize the OpenCL queue associated with the stream device.
void CLGuardImpl::synchronizeStream(const Stream &stream) const
{
    get_cl_queue(stream.device_index()).finish();
}

// Synchronize all queued work on the specified device.
void CLGuardImpl::synchronizeDevice(DeviceIndex device_index) const
{
    get_cl_queue(device_index).finish();
}

} // namespace c10::opencl

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::opencl::CLGuardImpl);
