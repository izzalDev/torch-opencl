cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(OpenCLICDLoader REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
find_package(OpenCLHeaders REQUIRED)
find_package(fmt REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
)
nanobind_add_module(_C ${SOURCE_FILES})

target_compile_definitions(_C PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS
)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(_C PRIVATE
    ${TORCH_LIBRARIES}
    OpenCL::HeadersCpp
    fmt::fmt
)

if(APPLE)
    target_link_libraries(_C PRIVATE "-framework OpenCL")
else()
    target_link_libraries(_C PRIVATE OpenCL::OpenCL)
endif()

nanobind_add_stub(
    _C_stub
    MODULE _C
    OUTPUT _C.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_C>
    DEPENDS _C
    MARKER_FILE py.typed
)

install(TARGETS _C
    DESTINATION ${SKBUILD_PROJECT_NAME}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${SKBUILD_PROJECT_NAME}
)
