#include "aten/native/Minimal.h"

#include <torch/library.h>

namespace at::native::opencl {

at::Tensor empty_memory_format(c10::IntArrayRef size, std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt,
                               std::optional<c10::MemoryFormat> memory_format_opt) {
  // ── 1. Resolve parameter optional ke nilai konkret ──────────────────────
  // device_or_default → kalau device_opt kosong, pakai device aktif saat ini
  // dtype_or_default  → kalau dtype_opt kosong, pakai float32
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);

  // ── 2. Validasi precondition ─────────────────────────────────────────────
  // Fungsi ini hanya boleh dipanggil untuk device PrivateUse1 (alias "opencl")
  TORCH_CHECK(device.is_privateuseone(), "empty_memory_format: expected opencl device, got ",
              device);
  // OpenCL backend kita hanya support layout Strided (dense tensor biasa)
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  // Pin memory hanya relevan untuk CPU↔GPU transfer, tidak untuk OpenCL kita
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Pin memory can only be on CPU");

  // ── 3. Set device aktif selama scope fungsi ini ──────────────────────────
  // DeviceGuard memastikan operasi berikutnya menggunakan device yang diminta,
  // dan otomatis restore device sebelumnya saat keluar scope (RAII).
  const c10::DeviceGuard device_guard(device);

  // ── 4. Hitung contiguous strides ─────────────────────────────────────────
  // Contiguous = elemen tersimpan rapat di memori, row-major (C order).
  // Rumus: stride[i] = stride[i+1] * size[i+1], stride[last] = 1
  //
  // Contoh size = [2, 3, 4]:
  //   stride[2] = 1
  //   stride[1] = 1 * 4 = 4
  //   stride[0] = 4 * 3 = 12
  //
  // Artinya untuk akses elemen [i,j,k]: offset = i*12 + j*4 + k*1
  const auto ndim = size.size();
  std::vector<int64_t> strides(ndim);
  if (ndim > 0) {
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; --i) strides[i] = strides[i + 1] * size[i + 1];
  }

  // ── 5. Hitung total byte yang dibutuhkan ─────────────────────────────────
  // elementSize(dtype) → ukuran satu elemen dalam byte (float32 = 4, int8 = 1, dst)
  // Gunakan c10::multiply_integers untuk overflow-safe multiply
  const int64_t nelems = c10::multiply_integers(size);
  const size_t nbytes = static_cast<size_t>(nelems) * c10::elementSize(dtype);

  // ── 6. Alokasi memori via allocator PrivateUse1 ──────────────────────────
  // GetAllocator mengembalikan allocator yang sudah kita register lewat
  // at::SetAllocator(kPrivateUse1, &global_opencl_allocator).
  // allocate() membuat cl::Buffer di GPU dan membungkusnya dalam DataPtr.
  auto *allocator = at::GetAllocator(at::kPrivateUse1);
  TORCH_CHECK(allocator != nullptr, "OpenCL allocator belum diregister!");

  // ── 7. Buat StorageImpl ──────────────────────────────────────────────────
  // StorageImpl adalah objek yang memiliki raw memory (DataPtr).
  // Semua TensorImpl yang berbagi storage yang sama = view satu sama lain.
  //
  // use_byte_size_t() → flag bahwa ukuran dinyatakan dalam bytes (bukan elemen)
  // resizable=true    → storage boleh di-resize nanti (misal oleh resize_())
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes,
      nbytes > 0 ? allocator->allocate(nbytes) : at::DataPtr{}, allocator,
      /*resizable=*/true);

  // ── 8. Buat TensorImpl ───────────────────────────────────────────────────
  // TensorImpl = metadata tensor (shape, stride, dtype, device, dll)
  //              + pointer ke StorageImpl di atas.
  //
  // DispatchKeySet(PrivateUse1) → memberi tahu dispatcher bahwa tensor ini
  // adalah milik backend opencl, bukan CPU/CUDA.
  //
  // scalarTypeToTypeMeta → konversi ScalarType (enum) ke TypeMeta (type info)
  constexpr auto pu1_dks = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage), pu1_dks,
                                                         c10::scalarTypeToTypeMeta(dtype));

  // ── 9. Set shape dan strides ─────────────────────────────────────────────
  // TensorImpl baru lahir tanpa shape — kita set manual di sini.
  // Setelah ini tensor.sizes() == size dan tensor.strides() == strides.
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);

  return tensor;
}

}  // namespace at::native::opencl
