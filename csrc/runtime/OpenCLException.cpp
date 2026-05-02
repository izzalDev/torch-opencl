#include "OpenCLException.h"

void clCheckFail(
    const char *func, const char *file, uint32_t line, const char *msg) {
  throw c10::Error({func, file, line}, msg);
}
