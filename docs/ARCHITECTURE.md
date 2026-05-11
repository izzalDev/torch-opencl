# Arsitektur torch-opencl

Dokumen ini menjelaskan **bagaimana** dan **mengapa** setiap bagian project ini
dirancang seperti ini, sehingga Anda bisa langsung paham ketika membuka kembali
kode ini.

## Konsep Inti: PyTorch PrivateUse1

PyTorch menyediakan mekanisme `PrivateUse1` — sebuah "slot" device generik yang
bisa diklaim oleh siapapun untuk membuat custom backend tanpa memodifikasi PyTorch
itu sendiri. torch-opencl menggunakan slot ini dan menamainya `"opencl"`.

```
PyTorch Dispatcher
      │
      ├── cpu          → ATen CPU kernels
      ├── cuda         → ATen CUDA kernels
      ├── mps          → ATen Metal kernels
      └── PrivateUse1  ←── kita daftarkan di sini sebagai "opencl"
```

Setelah `rename_privateuse1_backend("opencl")` dipanggil, PyTorch mengenal
`torch.device("opencl")` dan meroute semua operasi ke implementasi kita.

## Alur: `tensor.to("opencl")`

Ini adalah alur paling fundamental. Memahami ini berarti memahami seluruh backend.

```
Python: t.to("opencl")
         │
         ▼
PyTorch dispatcher → aten::empty_strided (PrivateUse1)
         │
         ▼
[OpenCLEmptyOps.cpp] wrapper__empty_strided()
  1. Resolve device index (default: current_device())
  2. Hitung nbytes = numel × sizeof(dtype)
  3. getOpenCLAllocator(device)->allocate(nbytes)
         │
         ▼
  [OpenCLDeviceAllocator.cpp] OpenCLAllocator::allocate()
    - cl::Buffer(context, CL_MEM_READ_WRITE, size)  ← alokasi di GPU
    - Bungkus dalam BufferEntry{buffer, size, device_index}
    - Return at::DataPtr yang menunjuk ke BufferEntry*
         │
         ▼
  Tensor baru terbentuk dengan storage di OpenCL device
         │
         ▼
PyTorch dispatcher → aten::_copy_from (PrivateUse1)
         │
         ▼
[OpenCLCopyOps.cpp] wrapper__copy_from()
  - Deteksi arah: CPU→GPU, GPU→CPU, atau GPU→GPU
  - H2D: queue.enqueueWriteBuffer(dst_buffer, src_cpu_ptr)
  - D2H: queue.enqueueReadBuffer(src_buffer, dst_cpu_ptr)
  - D2D: queue.enqueueCopyBuffer(src_buffer, dst_buffer)
```

## Komponen C++

### `runtime/OpenCLFunctions` — Inti Runtime

File paling fundamental. Menyimpan satu global `vector<DeviceContext>`:

```cpp
struct DeviceContext {
  cl::Device device;
  std::shared_ptr<cl::Context> context;  // shared antar device satu platform
  cl::CommandQueue queue;                // satu queue per device
};
```

**Inisialisasi lazy** via `c10::call_once` — hanya dijalankan sekali. Prosesnya:

1. Enumerate semua OpenCL platform
2. Untuk setiap platform, ambil semua GPU device
3. Buat satu shared `cl::Context` per platform
4. Buat `cl::CommandQueue` per device

**Thread-local device index**: `tl_current_device` adalah thread-local, sehingga
setiap thread Python punya "current device" sendiri — konsisten dengan perilaku CUDA.

### `runtime/OpenCLDeviceAllocator` — Manajemen Memori

Kunci desain: cara PyTorch mengakses memori GPU kita:

```
PyTorch menyimpan → void* data_ptr → BufferEntry* → cl::Buffer
```

PyTorch tidak tahu tentang `cl::Buffer`. Yang ia pegang hanyalah `void*`
yang menunjuk ke `BufferEntry`. Setiap kali perlu mengakses buffer OpenCL:

```cpp
auto* entry = static_cast<BufferEntry*>(tensor.data_ptr());
cl::Buffer& buf = entry->buffer;
```

`BufferEntry` dilacak dalam `unordered_map<void*, BufferEntry*>` per allocator
untuk memastikan cleanup yang benar. `Delete` callback dipanggil PyTorch saat
storage tidak lagi dibutuhkan.

### `aten/OpenCLEmptyOps` — Alokasi Tensor

**Wajib ada.** PyTorch memanggil `empty_strided` setiap kali perlu membuat
tensor baru di device (saat `.to()`, `.clone()`, output dari operasi, dll.).
Tanpa ini, semua operasi gagal dengan `NotImplementedError`.

```cpp
allocator->allocate(nbytes)            // → cl::Buffer di GPU
→ c10::Storage(...)                    // bungkus dalam PyTorch Storage
→ make_tensor<TensorImpl>(storage)     // buat TensorImpl
→ set_sizes_and_strides(size, stride)  // terapkan shape
```

### `aten/OpenCLCopyOps` — Transfer Data

Tiga jalur transfer, semua sinkron (`CL_TRUE`):

| Jalur                    | Fungsi OpenCL        | Keterangan                       |
| ------------------------ | -------------------- | -------------------------------- |
| CPU → GPU (H2D)          | `enqueueWriteBuffer` | Tulis dari RAM ke cl::Buffer     |
| GPU → CPU (D2H)          | `enqueueReadBuffer`  | Baca dari cl::Buffer ke RAM      |
| GPU → GPU (same device)  | `enqueueCopyBuffer`  | Copy antar buffer langsung       |
| GPU → GPU (cross-device) | Read + Write via tmp | OpenCL tidak jamin cross-context |

Transfer sinkron dipilih untuk kesederhanaan. Async membutuhkan manajemen
event/fence yang kompleks — kandidat optimasi di masa depan.

### `runtime/OpenCLGuard` — Device Context Switching

Mengimplementasikan `DeviceGuardImplInterface` PyTorch. Dipakai PyTorch secara
internal untuk memastikan operasi dijalankan di device yang benar.

```cpp
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenCLGuardImpl);
```

### `runtime/OpenCLHooks` — Integrasi Level Tinggi

`PrivateUse1HooksInterface` adalah antarmuka yang PyTorch gunakan untuk
bertanya tentang kemampuan device kita: `isAvailable()`, `deviceCount()`,
`getDeviceFromPtr()`, dll. Registrasi via static initializer saat `.so` di-load.

## Komponen Python

### `torch_opencl/__init__.py` — Registrasi Backend

Tiga langkah registrasi yang harus berurutan:

```python
rename_privateuse1_backend("opencl")             # 1. Beri nama "opencl"
torch._register_device_module("opencl", opencl)  # 2. Pasang torch.opencl
generate_methods_for_privateuse1_backend(...)    # 3. Generate .opencl() di Tensor
```

Setelah ini, `tensor.opencl()`, `tensor.is_opencl`, dll. tersedia secara otomatis.

## Aliran Registrasi (saat `import torch_opencl`)

```
import torch_opencl
    │
    ├── Load _C.so → static initializers C++ dijalankan:
    │     ├── OpenCLHooks: RegisterPrivateUse1HooksInterface(...)
    │     └── OpenCLGuard: C10_REGISTER_GUARD_IMPL(PrivateUse1, ...)
    │
    └── __init__.py:
          ├── rename_privateuse1_backend("opencl")
          ├── torch._register_device_module("opencl", opencl_module)
          └── generate_methods_for_privateuse1_backend(for_storage=True)

Setelah selesai:
  ✓ torch.device("opencl") valid
  ✓ torch.opencl.is_available() tersedia
  ✓ tensor.to("opencl") berfungsi
  ✓ tensor.opencl() / tensor.is_opencl tersedia
```

## Operator yang Diimplementasikan

| Operator                      | File                 | Keterangan               |
| ----------------------------- | -------------------- | ------------------------ |
| `aten::empty_strided`         | `OpenCLEmptyOps.cpp` | Alokasi tensor baru      |
| `aten::_copy_from`            | `OpenCLCopyOps.cpp`  | Transfer H2D, D2H, D2D   |
| `aten::_copy_from_and_resize` | `OpenCLCopyOps.cpp`  | Delegasi ke `_copy_from` |

Semua operator komputasi (add, matmul, dll.) belum diimplementasikan.

## Menambahkan Operator Baru

```cpp
// csrc/aten/OpenCLMyOp.cpp
#include <torch/library.h>
#include "runtime/OpenCLFunctions.h"

namespace at::opencl {
namespace {

at::Tensor wrapper__my_op(const at::Tensor& self) {
  // implementasi dengan OpenCL kernel
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("my_op", wrapper__my_op);
}

}  // namespace at::opencl
```

File baru di `csrc/` otomatis diambil CMake via `GLOB_RECURSE` —
tidak perlu edit `CMakeLists.txt`.

## Keterbatasan Saat Ini

| Keterbatasan               | Dampak                           | Solusi ke Depan                    |
| -------------------------- | -------------------------------- | ---------------------------------- |
| Transfer sinkron           | Blocking CPU selama copy         | Async + cl::Event                  |
| Satu queue per device      | Tidak ada multi-stream           | Pool of queues                     |
| Tidak ada kernel komputasi | Hanya transfer yang jalan di GPU | Implementasi kernel OpenCL         |
| Cross-device copy via host | Performa buruk untuk multi-GPU   | Peer-to-peer jika platform sama    |
| Pinned memory tidak nyata  | `isPinnedPtr` selalu false       | Implementasi CL_MEM_ALLOC_HOST_PTR |
