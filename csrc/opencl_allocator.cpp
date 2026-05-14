#include "opencl_allocator.h"
#include "opencl_context.h"

#include <CL/opencl.hpp>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// Tempat simpan objek buffer supaya nggak kena destroy (karena OpenCL 1.2 butuh
// ini)
static std::unordered_map<const void *, cl::Buffer> g_buffers;
// Biar nggak crash kalau ada banyak thread alokasi barengan
static std::mutex g_mutex;

// Masukkin buffer ke map
void register_cl_buffer(void *key, cl::Buffer buf) {
  std::lock_guard lock(g_mutex);
  g_buffers.emplace(key, std::move(buf));
}

// Ambil balik objek buffernya pakai alamat pointernya
cl::Buffer *get_cl_buffer(const void *ptr) {
  std::lock_guard lock(g_mutex);
  auto it = g_buffers.find(ptr);
  return it != g_buffers.end() ? &it->second : nullptr;
}

// Hapus buffer dari map (otomatis bebasin memori di GPU)
static void delete_cl_buffer(void *ptr) {
  std::lock_guard lock(g_mutex);
  g_buffers.erase(ptr);
}

// Singleton biar allocatornya cuma satu di seluruh app
OpenCLAllocator &OpenCLAllocator::instance() {
  static OpenCLAllocator alloc;
  return alloc;
}

// Fungsi buat pesen memori di GPU
c10::DataPtr OpenCLAllocator::allocate(size_t nbytes) {
  auto &dev = OpenCLContext::instance().current_device();

  // Minimal alokasi 1 byte biar OpenCL nggak ngambek
  size_t size = std::max(nbytes, size_t(1));

  // Bikin buffer di GPU
  cl::Buffer buf(dev.context, CL_MEM_READ_WRITE, size);

  // Pakai handle internal buffer buat jadi "pointer palsu" buat PyTorch
  void *key = static_cast<void *>(buf());

  // Daftarin ke map biar objek 'buf' nggak mati pas fungsi ini kelar
  register_cl_buffer(key, std::move(buf));

  // Balikin DataPtr yang udah ditempel fungsi delete_cl_buffer
  c10::Device c10_dev(c10::DeviceType::PrivateUse1, dev.index);
  return c10::DataPtr(key, key, &delete_cl_buffer, c10_dev);
}

c10::DeleterFnPtr OpenCLAllocator::raw_deleter() const {
  return &delete_cl_buffer;
}

// Fungsi buat nyalin data antar buffer di GPU
void OpenCLAllocator::copy_data(void *dest, const void *src,
                                std::size_t count) const {
  auto &dev = OpenCLContext::instance().current_device();

  // Cari objek asli dari pointer dest dan src
  auto *src_buf = get_cl_buffer(src);
  auto *dest_buf = get_cl_buffer(dest);

  if (!src_buf || !dest_buf) {
    throw std::runtime_error("copy_data: pointer nggak dikenal sama OpenCL");
  }

  // Suruh GPU copy datanya
  dev.queue.enqueueCopyBuffer(*src_buf, *dest_buf, 0, 0, count);

  // Tunggu sampai copy selesai (synchronous)
  dev.queue.finish();
}
