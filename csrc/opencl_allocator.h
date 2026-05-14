#pragma once
#include <CL/opencl.hpp>
#include <c10/core/Allocator.h>

cl::Buffer *get_cl_buffer(const void *data_ptr);
void register_cl_buffer(void *key, cl::Buffer buf);

class OpenCLAllocator final : public c10::Allocator {
public:
  static OpenCLAllocator &instance();
  c10::DataPtr allocate(size_t nbytes) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void *dest, const void *src, std::size_t count) const override;
};
