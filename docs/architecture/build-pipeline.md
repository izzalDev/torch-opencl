# Build Pipeline

## Build Architecture

```mermaid
flowchart TD
    A[Python Package] --> B[CMake]
    B --> C[C++ Compilation]
    C --> D[OpenCL Linking]
    D --> E[PyTorch Extension]
    E --> F[Python Import]
```

## Komponen Build

### CMake

Digunakan untuk:

- compile backend
- link OpenCL
- setup include path

### Conan

Dependency manager untuk:

- OpenCL headers
- helper library

### PyTorch Extension

Digunakan agar extension dapat di-load langsung dari Python.

## Output

Build menghasilkan:

- shared library (`.so` / `.pyd`)
- Python package
