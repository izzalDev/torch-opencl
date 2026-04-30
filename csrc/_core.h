#pragma once
#include "opencl_context.h"

bool is_available();
int device_count();
std::string device_name(int index);
int current_device();
void set_device(int index);
const OpenCLDevice &get_device_properties(int index);
void synchronize(int index);
