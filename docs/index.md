---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Torch OpenCL"
  text: "OpenCL Backend for PyTorch"
  tagline: "Experimental PyTorch backend using OpenCL and PrivateUse1 integration."
  actions:
    - theme: brand
      text: Architecture
      link: /architecture/

    - theme: alt
      text: GitHub
      link: https://github.com/izzalDev/torch-opencl

features:
  - title: PyTorch Integration
    details: Integrates OpenCL directly into the PyTorch dispatcher using the PrivateUse1 backend system.

  - title: OpenCL Runtime
    details: Runtime abstraction layer for OpenCL platforms, devices, contexts, and command queues.

  - title: Tensor Backend
    details: Supports tensor allocation, tensor transfer, and tensor execution using OpenCL buffers.

  - title: Custom Allocator
    details: Custom allocator integration between PyTorch storage and OpenCL memory objects.

  - title: Kernel Execution
    details: Executes OpenCL kernels through the PyTorch dispatcher and backend runtime pipeline.

  - title: Integration Testing
    details: Backend validation is performed using full integration testing against real runtime execution.
---
