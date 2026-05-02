#include "opencl_hooks.h"
#include "opencl_allocator.h"
#include "opencl_context.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

// ── DeviceGuard ─────────────────────────────────────────────────────────────

struct OpenCLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    auto &ctx = OpenCLContext::instance();
    int old = ctx.current_device().index;
    ctx.set_device(d.index());
    return c10::Device(static_type, old);
  }

  c10::Device getDevice() const override {
    return {static_type,
            (c10::DeviceIndex)OpenCLContext::instance().current_device().index};
  }

  void setDevice(c10::Device d) const override {
    OpenCLContext::instance().set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    OpenCLContext::instance().set_device(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, getDevice());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return (c10::DeviceIndex)OpenCLContext::instance().device_count();
  }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenCLGuardImpl);

// ── PrivateUse1 Hooks ────────────────────────────────────────────────────────

struct OpenCLHooks final : public at::PrivateUse1HooksInterface {
  bool hasPrimaryContext(c10::DeviceIndex) const override {
    return OpenCLContext::instance().is_available();
  }

  bool isPinnedPtr(const void *) const override {
    return false;
  }
};

// ── Entry point ──────────────────────────────────────────────────────────────

void register_opencl_backend() {
  c10::register_privateuse1_backend("opencl");

  c10::SetAllocator(c10::DeviceType::PrivateUse1, &OpenCLAllocator::instance(),
                    /*priority=*/100);

  at::RegisterPrivateUse1HooksInterface(new OpenCLHooks());
}
