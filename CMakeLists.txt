cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#12.1.0")
CPMAddPackage("gh:KhronosGroup/OpenCL-Headers@2025.07.22")
CPMAddPackage("gh:KhronosGroup/OpenCL-ICD-Loader@2025.07.22")
CPMAddPackage("gh:KhronosGroup/OpenCL-CLHPP@2025.07.22")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(OpenCL REQUIRED)
find_package(nanobind CONFIG REQUIRED)

set(OCL_VER $<IF:$<PLATFORM_ID:Darwin>,120,300>)

file(GLOB_RECURSE SOURCE_FILES "csrc/*.cpp")

nanobind_add_module(_C ${SOURCE_FILES})

target_compile_definitions(_C PRIVATE
				CL_TARGET_OPENCL_VERSION=${OCL_VER}
				CL_HPP_TARGET_OPENCL_VERSION=${OCL_VER}
				CL_HPP_MINIMUM_OPENCL_VERSION=120
				CL_HPP_ENABLE_EXCEPTIONS
)

target_include_directories(_C PRIVATE csrc/runtime include)

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
