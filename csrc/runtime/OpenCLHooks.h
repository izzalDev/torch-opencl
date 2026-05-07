#pragma once
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {
namespace opencl {

struct OpenCLHooks : public at::PrivateUse1HooksInterface {
    OpenCLHooks() = default;
    ~OpenCLHooks() override = default;

    const at::Generator& getDefaultGenerator(at::DeviceIndex device_index) const override;
    at::Device getDeviceFromPtr(void* data) const override;
    // Perbaikan: tambahkan const
    bool isPinnedPtr(const void* data) const override;
};

}  // namespace opencl
}  // namespace at
