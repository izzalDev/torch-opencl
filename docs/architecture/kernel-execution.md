# Kernel Execution

## Pipeline Eksekusi

```mermaid
sequenceDiagram
    participant P as PyTorch Op
    participant K as OpenCL Kernel
    participant Q as Command Queue
    participant G as GPU

    P->>K: launch kernel
    K->>Q: enqueueNDRangeKernel
    Q->>G: execute
    G-->>Q: complete
```

## Kernel Lifecycle

1. Build source OpenCL
2. Compile kernel
3. Create kernel object
4. Bind tensor buffers
5. Dispatch NDRange

## Buffer Binding

Tensor storage diterjemahkan menjadi:

- input buffer
- output buffer
- scalar arguments

## Synchronization Model

OpenCL backend menggunakan:

- blocking transfer
- explicit queue synchronization
- event-based execution
