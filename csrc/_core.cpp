#include "_core.h"

#include "opencl_context.h"

bool is_available() {
  return OpenCLContext::instance().is_available();
}

int device_count() {
  return OpenCLContext::instance().device_count();
}

std::string device_name(int index) {
  return OpenCLContext::instance().device_name(index);
}

int current_device() {
  return OpenCLContext::instance().current_device();
}

void set_device(int index) {
  OpenCLContext::instance().set_device(index);
}

const OpenCLDevice &get_device_properties(int index) {
  return OpenCLContext::instance().get(index);
}

void synchronize(int index) {
  OpenCLContext::instance().get(index).queue.finish();
}
