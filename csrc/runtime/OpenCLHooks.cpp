#include "OpenCLHooks.h"

#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace at {
namespace opencl {

const at::Generator& OpenCLHooks::getDefaultGenerator(at::DeviceIndex device_index) const {
    static auto gen = at::make_generator<at::CPUGeneratorImpl>();
    return gen;
}

at::Device OpenCLHooks::getDeviceFromPtr(void* data) const {
    return at::Device(at::kPrivateUse1, 0);
}

bool OpenCLHooks::isPinnedPtr(const void* data) const {
    return false;
}

struct OpenCLHooksRegisterer {
    OpenCLHooksRegisterer() {
        at::RegisterPrivateUse1HooksInterface(new OpenCLHooks());
    }
};

static OpenCLHooksRegisterer registerer;

}  // namespace opencl
}  // namespace at
