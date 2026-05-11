# Allocator Design

## Tujuan

Menghubungkan allocator PyTorch dengan memory OpenCL.

## Arsitektur Allocator

```mermaid
flowchart TD
    A[PyTorch Tensor] --> B[at::Allocator]
    B --> C[OpenCLDeviceAllocator]
    C --> D[cl::Buffer]
```

## Masalah Utama

PyTorch menggunakan model:

```cpp
void*
```

Sedangkan OpenCL menggunakan object:

```cpp
cl::Buffer
```

Karena itu backend memerlukan wrapper internal.

## Buffer Wrapper

```cpp
struct BufferEntry {
    cl::Buffer buffer;
    size_t size;
    int device;
};
```

## Lifecycle

```mermaid
sequenceDiagram
    participant T as Tensor
    participant A as Allocator
    participant B as Buffer

    T->>A: allocate
    A->>B: create cl::Buffer
    B-->>A: buffer handle
    A-->>T: DataPtr
```

## Release Strategy

Ketika reference tensor habis:

- DataPtr destructor dipanggil
- cl::Buffer direlease
- memory GPU dibebaskan
