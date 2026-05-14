# Architecture

Dokumentasi arsitektur internal `torch-opencl`.

Backend ini mengintegrasikan OpenCL ke dalam ekosistem PyTorch menggunakan
mekanisme `PrivateUse1` sehingga tensor OpenCL dapat diperlakukan seperti
backend native lain (`cuda`, `mps`, dll).

## Fokus Arsitektur

- Integrasi dengan dispatcher PyTorch
- Runtime OpenCL abstraction
- Memory allocator berbasis `cl::Buffer`
- Device management
- Build dan extension loading
- Runtime execution flow

## Struktur Dokumentasi

| Dokumen                | Deskripsi                   |
| ---------------------- | --------------------------- |
| System Overview        | Gambaran umum arsitektur    |
| Runtime Flow           | Alur runtime tensor         |
| Runtime Layer          | Manajemen context dan queue |
| Allocator Design       | Integrasi allocator OpenCL  |
| Dispatcher Integration | Integrasi PrivateUse1       |
| Kernel Execution       | Eksekusi kernel OpenCL      |
| Build Pipeline         | Pipeline build dan linking  |
| Testing Strategy       | Strategi testing backend    |
