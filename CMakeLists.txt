cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/CPM.cmake)

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

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(kineto_LIBRARY "NOT_REQUIRED" CACHE STRING "Silence kineto warning")
find_package(Torch CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(csrc)
add_subdirectory(torch_opencl/csrc)
