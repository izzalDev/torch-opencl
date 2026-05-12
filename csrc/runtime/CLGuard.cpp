#include "runtime/CLGuard.h"

namespace c10::opencl {

// LITERALINCLUDE START: OPENCL GUARD REGISTRATION
C10_REGISTER_GUARD_IMPL(PrivateUse1, CLGuardImpl);
// LITERALINCLUDE END: OPENCL GUARD REGISTRATION

} // namespace c10::opencl
