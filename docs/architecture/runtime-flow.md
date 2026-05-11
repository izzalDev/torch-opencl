# Runtime Flow

## Flow Tensor Creation

```mermaid
sequenceDiagram
    participant P as Python
    participant D as Dispatcher
    participant A as Allocator
    participant O as OpenCL Runtime

    P->>D: tensor.to("opencl")
    D->>A: allocate()
    A->>O: cl::Buffer
    O-->>A: device buffer
    A-->>D: DataPtr
    D-->>P: OpenCL Tensor
```

## Tahapan Runtime

### 1. Tensor Dispatch

Ketika tensor dipindahkan ke device `opencl`,
dispatcher PyTorch mencari kernel dan allocator
yang sesuai untuk backend `PrivateUse1`.

### 2. Memory Allocation

Allocator membuat `cl::Buffer`
menggunakan context aktif.

### 3. Tensor Storage Binding

Buffer kemudian dibungkus menjadi `at::DataPtr`
agar kompatibel dengan storage system PyTorch.

### 4. Kernel Execution

Operator tensor akan memanggil implementasi backend OpenCL.

## Synchronization

Backend menggunakan command queue OpenCL
untuk sinkronisasi operasi.

```mermaid
flowchart LR
    A[Tensor Op] --> B[Command Queue]
    B --> C[OpenCL Kernel]
    C --> D[Device Execution]
```
