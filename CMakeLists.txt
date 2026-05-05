cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()

include(cmake/CPM.cmake)

cpmaddpackage(
  NAME
  OpenCL-SDK
  GITHUB_REPOSITORY
  KhronosGroup/OpenCL-SDK
  VERSION
  2025.07.23
  OPTIONS
  "BUILD_TESTING OFF"
  "BUILD_DOCS OFF"
  "BUILD_EXAMPLES OFF"
  "BUILD_SHARED_LIBS OFF"
  "OPENCL_SDK_BUILD_SAMPLES OFF"
  "OPENCL_SDK_BUILD_UTILITIES OFF"
  "CL_HPP_ENABLE_EXCEPTIONS ON")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(csrc)
add_subdirectory(torch_opencl/csrc)
