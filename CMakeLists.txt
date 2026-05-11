cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)

set(MODULE_NAME    _C)
set(STUB_TARGET    ${MODULE_NAME}_stub)
set(STUB_FILE      ${MODULE_NAME}.pyi)
set(PKG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME})

if(SKBUILD_STATE STREQUAL "editable")
    set(INSTALL_DIR ${PKG_SOURCE_DIR})
else()
    set(INSTALL_DIR ${SKBUILD_PROJECT_NAME})
endif()

if(APPLE)
    set(CL_VERSION 120)
    set(OPENCL_LIBS "-framework OpenCL")
else()
    set(CL_VERSION 300)
    find_package(OpenCL REQUIRED)
    set(OPENCL_LIBS OpenCL::OpenCL)
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS csrc/*.cpp)
nanobind_add_module(${MODULE_NAME} ${SOURCES})

target_compile_definitions(${MODULE_NAME} PRIVATE
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_TARGET_OPENCL_VERSION=${CL_VERSION}
    CL_HPP_TARGET_OPENCL_VERSION=${CL_VERSION}
)
target_include_directories(${MODULE_NAME} PRIVATE csrc)
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)

target_link_libraries(${MODULE_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    OpenCL::HeadersCpp
    OpenCL::Headers
    ${OPENCL_LIBS}
)

if(MSVC)
    target_compile_options(${MODULE_NAME} PRIVATE /wd4267)
endif()

install(TARGETS ${MODULE_NAME} DESTINATION ${INSTALL_DIR})

if(NOT WIN32)
    nanobind_add_stub(
        ${STUB_TARGET}
        MODULE      ${MODULE_NAME}
        OUTPUT      ${STUB_FILE}
        PYTHON_PATH $<TARGET_FILE_DIR:${MODULE_NAME}>
        DEPENDS     ${MODULE_NAME}
        MARKER_FILE py.typed
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${STUB_FILE}
        ${CMAKE_CURRENT_BINARY_DIR}/py.typed
        DESTINATION ${INSTALL_DIR}
    )
endif()
