#include "OpenCLHooks.h"

// LITERALINCLUDE START: OPENCL HOOK REGISTER
namespace c10::opencl {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenCLHooksInterface());

  return true;
}();

}  // namespace c10::opencl
// LITERALINCLUDE END: OPENCL HOOK REGISTER
