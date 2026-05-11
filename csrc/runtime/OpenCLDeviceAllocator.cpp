#include "runtime/OpenCLDeviceAllocator.h"

#include <CL/cl.h>
#include <c10/core/Device.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/headeronly/core/DeviceType.h>

#include <CL/opencl.hpp>
#include <cstring>

#include "runtime/OpenCLFunctions.h"

namespace c10::opencl {
namespace {

void deleteHandle(void *ptr) { delete static_cast<OpenCLBufferHandle *>(ptr); }

} // namespace

// ─────────────────────────────────────────────
// allocate()
//
// Cara kerja:
// 1. Tanya guard "device mana yang aktif sekarang?"
// 2. Ambil cl::Context milik device itu
// 3. Buat cl::Buffer (memori di GPU) sebesar nbytes
// 4. Bungkus dalam OpenCLBufferHandle (struct kita)
// 5. Kembalikan DataPtr — berisi pointer ke handle,
//    deleter-nya deleteHandle, dan device tag-nya
//
// DataPtr adalah "smart pointer" PyTorch yang tahu
// di device mana data ini hidup + cara menghapusnya.
// ─────────────────────────────────────────────
at::DataPtr OpenCLDeviceAllocator::allocate(size_t nbytes)
{
    c10::impl::VirtualGuardImpl guard(at::kPrivateUse1);
    const auto device_index = guard.getDevice().index();
    cl::Context &context = get_cl_context(device_index);

    // CL_MEM_READ_WRITE: buffer bisa dibaca dan ditulis kernel
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, std::max(nbytes, size_t(1)));

    auto *handle = new OpenCLBufferHandle{std::move(buffer), device_index, nbytes};

    // DataPtr(data, ctx, deleter, device)
    // - data    : pointer yang dikembalikan ke pengguna (handle kita)
    // - ctx     : pointer yang dikirim ke deleter
    // - deleter : fungsi yang dipanggil saat tensor dihapus
    // - device  : tag untuk PyTorch tahu ini di device mana
    return {handle, handle, &deleteHandle, at::Device(at::kPrivateUse1, device_index)};
}

// ─────────────────────────────────────────────
// raw_deleter()
//
// PyTorch kadang perlu tahu fungsi deleter secara
// eksplisit (misal untuk clone storage). Kembalikan
// deleteHandle yang sama dengan yang dipakai DataPtr.
// ─────────────────────────────────────────────
at::DeleterFnPtr OpenCLDeviceAllocator::raw_deleter() const { return &deleteHandle; }

// ─────────────────────────────────────────────
// copy_data()
//
// Dipanggil PyTorch saat perlu menyalin data antar
// storage di device yang sama (misal saat .clone()).
// Untuk OpenCL yang benar seharusnya pakai
// clEnqueueCopyBuffer, tapi memcpy cukup untuk
// bootstrap awal karena handle kita adalah struct
// di host memory yang membungkus cl::Buffer.
//
// TODO: ganti dengan clEnqueueCopyBuffer bila
// copy data GPU-to-GPU nyata dibutuhkan.
// ─────────────────────────────────────────────
void OpenCLDeviceAllocator::copy_data(void *dest, const void *src, std::size_t count) const
{
    std::memcpy(dest, src, count);
}

// ─────────────────────────────────────────────
// initialized()
//
// Dipakai PyTorch untuk cek apakah backend sudah
// siap. Cukup cek ada tidaknya device OpenCL.
// ─────────────────────────────────────────────
bool OpenCLDeviceAllocator::initialized() { return device_count() > 0; }

// ─────────────────────────────────────────────
// emptyCache() — no-op
//
// CUDA punya memory pool/cache yang bisa di-flush.
// OpenCL kita belum punya caching allocator,
// setiap allocate() langsung ke driver OpenCL.
// ─────────────────────────────────────────────
void OpenCLDeviceAllocator::emptyCache(MempoolId_t) {}

// ─────────────────────────────────────────────
// recordStream() — no-op
//
// Dipakai CUDA untuk pastikan memori tidak dibebaskan
// sebelum operasi di stream selesai.
// OpenCL kita pakai satu queue per device dan
// belum punya multi-stream, jadi no-op dulu.
// ─────────────────────────────────────────────
void OpenCLDeviceAllocator::recordStream(const at::DataPtr &, c10::Stream) {}

// ─────────────────────────────────────────────
// getDeviceStats() — kosong
//
// Statistik memori (allocated, reserved, dll).
// Kembalikan struct kosong — belum ditracking.
// ─────────────────────────────────────────────
c10::CachingDeviceAllocator::DeviceStats OpenCLDeviceAllocator::getDeviceStats(c10::DeviceIndex)
{
    return {};
}

void OpenCLDeviceAllocator::resetAccumulatedStats(c10::DeviceIndex) {}
void OpenCLDeviceAllocator::resetPeakStats(c10::DeviceIndex) {}

} // namespace c10::opencl

// ─────────────────────────────────────────────
// Registrasi global (di luar namespace)
//
// Static initializer jalan saat .so di-load Python.
// Urutan:
// 1. global_opencl_allocator dibuat (konstruktor trivial)
// 2. Lambda dijalankan → at::SetAllocator mendaftarkan
//    allocator kita ke slot PrivateUse1 di registry PyTorch
// 3. Sekarang at::GetAllocator(at::kPrivateUse1) akan
//    mengembalikan &global_opencl_allocator
// ─────────────────────────────────────────────
static c10::opencl::OpenCLDeviceAllocator global_opencl_allocator;

static bool register_allocator [[maybe_unused]] = []() {
    at::SetAllocator(at::kPrivateUse1, &global_opencl_allocator);
    return true;
}();
