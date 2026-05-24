#include "aten/native/BinaryOps.h"
#include "runtime/CLDeviceAllocator.h"
#include "runtime/CLFunctions.h"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <type_traits>

namespace at::native::opencl {

static const c10::opencl::CLAllocation *get_cl_allocation(const at::Tensor &t)
{
    return static_cast<const c10::opencl::CLAllocation *>(
        t.storage().data_ptr().get()
    );
}

static std::string get_opencl_type_name(at::ScalarType scalar_type)
{
    switch (scalar_type) {
        case at::ScalarType::Float: return "float";
        case at::ScalarType::Double: return "double";
        case at::ScalarType::Int: return "int";
        case at::ScalarType::Long: return "long";
        case at::ScalarType::Short: return "short";
        case at::ScalarType::Char: return "char";
        case at::ScalarType::Byte: return "uchar";
        case at::ScalarType::Bool: return "uchar";
        default:
            TORCH_CHECK(false, "Unsupported scalar type for OpenCL: ", scalar_type);
    }
}

template <typename T>
struct ToCLType {
    using type = T;
};

template <>
struct ToCLType<bool> {
    using type = cl_uchar;
};

static cl::Kernel get_add_kernel(c10::DeviceIndex device_index, at::ScalarType scalar_type)
{
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, cl::Kernel> kernel_cache;

    std::string type_name = get_opencl_type_name(scalar_type);
    std::string cache_key = std::to_string(device_index) + "_" + type_name;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = kernel_cache.find(cache_key);
    if (it != kernel_cache.end()) {
        return it->second;
    }

    cl::Context &context = c10::opencl::get_cl_context(device_index);
    cl::Device &device = c10::opencl::get_cl_device(device_index);

    std::string pragma_fp64 = "";
    if (scalar_type == at::ScalarType::Double) {
        pragma_fp64 = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }

    std::string source = pragma_fp64 +
        "__kernel void add_kernel(\n"
        "    __global " + type_name + "* self,\n"
        "    const ulong self_offset,\n"
        "    __global const " + type_name + "* other,\n"
        "    const ulong other_offset,\n"
        "    const " + type_name + " alpha,\n"
        "    const ulong n)\n"
        "{\n"
        "    size_t id = get_global_id(0);\n"
        "    if (id < n) {\n"
        "        self[self_offset + id] += alpha * other[other_offset + id];\n"
        "    }\n"
        "}\n";

    cl::Program program(context, source);
    try {
        program.build({device});
    } catch (const cl::Error &) {
        std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        TORCH_CHECK(false, "Failed to build OpenCL add_kernel. Build log:\n", build_log);
    }

    cl::Kernel kernel(program, "add_kernel");
    kernel_cache[cache_key] = kernel;
    return kernel;
}

at::Tensor &add_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    if (self.numel() == 0) {
        return self;
    }

    TORCH_CHECK(
        self.device().is_privateuseone() && other.device().is_privateuseone(),
        "add_: tensors must be on OpenCL devices"
    );
    TORCH_CHECK(
        self.device() == other.device(),
        "add_: tensors must be on the same OpenCL device"
    );
    TORCH_CHECK(
        self.sizes() == other.sizes(),
        "add_: shape mismatch: ", self.sizes(), " vs ", other.sizes()
    );
    TORCH_CHECK(
        self.scalar_type() == other.scalar_type(),
        "add_: dtype mismatch: ", self.scalar_type(), " vs ", other.scalar_type()
    );

    // Fallback if not contiguous
    if (!self.is_contiguous() || !other.is_contiguous()) {
        int64_t self_storage_numel = static_cast<int64_t>(self.storage().nbytes() / self.element_size());
        auto self_flat = at::as_strided(self, {self_storage_numel}, {1}, 0);
        auto self_storage_cpu = at::empty({self_storage_numel}, self.options().device(at::kCPU));
        self_storage_cpu.copy_(self_flat);
        auto self_cpu = at::as_strided(self_storage_cpu, self.sizes(), self.strides(), self.storage_offset());

        int64_t other_storage_numel = static_cast<int64_t>(other.storage().nbytes() / other.element_size());
        auto other_flat = at::as_strided(other, {other_storage_numel}, {1}, 0);
        auto other_storage_cpu = at::empty({other_storage_numel}, other.options().device(at::kCPU));
        other_storage_cpu.copy_(other_flat);
        auto other_cpu = at::as_strided(other_storage_cpu, other.sizes(), other.strides(), other.storage_offset());

        self_cpu.add_(other_cpu, alpha);

        self_flat.copy_(self_storage_cpu);
        return self;
    }

    const c10::DeviceGuard device_guard(self.device());
    const auto *self_alloc = get_cl_allocation(self);
    const auto *other_alloc = get_cl_allocation(other);
    auto &queue = c10::opencl::get_cl_queue(self_alloc->device);

    const auto numel = self.numel();

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "add__opencl", [&]() {
        using cl_type = typename ToCLType<scalar_t>::type;
        scalar_t raw_alpha = alpha.to<scalar_t>();
        cl_type cl_alpha = static_cast<cl_type>(raw_alpha);

        cl::Kernel kernel = get_add_kernel(self_alloc->device, self.scalar_type());

        cl_int err = CL_SUCCESS;
        err = kernel.setArg(0, self_alloc->buffer);
        TORCH_CHECK(err == CL_SUCCESS, "setArg 0 failed: ", err);
        err = kernel.setArg(1, static_cast<cl_ulong>(self.storage_offset()));
        TORCH_CHECK(err == CL_SUCCESS, "setArg 1 failed: ", err);
        err = kernel.setArg(2, other_alloc->buffer);
        TORCH_CHECK(err == CL_SUCCESS, "setArg 2 failed: ", err);
        err = kernel.setArg(3, static_cast<cl_ulong>(other.storage_offset()));
        TORCH_CHECK(err == CL_SUCCESS, "setArg 3 failed: ", err);
        err = kernel.setArg(4, cl_alpha);
        TORCH_CHECK(err == CL_SUCCESS, "setArg 4 failed: ", err);
        err = kernel.setArg(5, static_cast<cl_ulong>(numel));
        TORCH_CHECK(err == CL_SUCCESS, "setArg 5 failed: ", err);

        cl_int err_launch = queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(numel),
            cl::NullRange
        );
        TORCH_CHECK(err_launch == CL_SUCCESS, "enqueueNDRangeKernel failed: ", err_launch);
    });

    queue.finish();
    return self;
}

} // namespace at::native::opencl
