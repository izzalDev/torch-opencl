cmake_minimum_required(VERSION 3.18)

# Project name is injected by scikit-build-core from pyproject.toml [project].name
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # Generate compile_commands.json for IDEs
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Required for shared libraries (.so/.dylib)

include(cmake/CPM.cmake)

# Download and build OpenCL-SDK from KhronosGroup.
# Disable tests, docs, samples and build as static to keep the wheel self-contained.
CPMAddPackage(
    NAME OpenCL-SDK
    GITHUB_REPOSITORY KhronosGroup/OpenCL-SDK
    VERSION 2025.07.23
    EXCLUDE_FROM_ALL YES
    OPTIONS
        "BUILD_TESTING OFF"
        "BUILD_DOCS OFF"
        "BUILD_EXAMPLES OFF"
        "BUILD_SHARED_LIBS OFF"
        "OPENCL_SDK_BUILD_SAMPLES OFF"
        "OPENCL_SDK_BUILD_UTILITIES OFF"
        "CL_HPP_ENABLE_EXCEPTIONS ON"
)

# Development: Python headers + lib needed by scikit-build-core's two-step find_package
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Silence the "Kineto not found" warning printed by TorchConfig.cmake
set(kineto_LIBRARY "NOT_REQUIRED" CACHE STRING "Silence kineto warning")

find_package(Torch CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(csrc)              # C++ core library (libtorch_opencl)
add_subdirectory(torch_opencl/csrc) # Python binding module (_C)
