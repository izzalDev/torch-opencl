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

nanobind_add_module(_core
				csrc/binding.cpp
				csrc/_core.cpp
				csrc/opencl_context.cpp
				csrc/opencl_allocator.cpp
				csrc/opencl_hooks.cpp
				csrc/opencl_ops.cpp
)

target_compile_definitions(_core PRIVATE
				CL_TARGET_OPENCL_VERSION=120
				CL_HPP_TARGET_OPENCL_VERSION=120
				CL_HPP_MINIMUM_OPENCL_VERSION=120
				CL_HPP_ENABLE_EXCEPTIONS
)

target_include_directories(_core PRIVATE
				csrc/include
)

target_link_libraries(_core PRIVATE
				${TORCH_LIBRARIES}
				OpenCL::HeadersCpp
				fmt::fmt
)

if(APPLE)
	target_link_libraries(_core PRIVATE "-framework OpenCL")
else()
	target_link_libraries(_core PRIVATE OpenCL::OpenCL)
endif()

nanobind_add_stub(
				_core_stub
				MODULE _core
				OUTPUT _core.pyi
				PYTHON_PATH $<TARGET_FILE_DIR:_core>
				DEPENDS _core
				MARKER_FILE py.typed
)

install(TARGETS _core
				DESTINATION ${SKBUILD_PROJECT_NAME}
)

install(FILES
				${CMAKE_CURRENT_BINARY_DIR}/_core.pyi
				${CMAKE_CURRENT_BINARY_DIR}/py.typed
				DESTINATION ${SKBUILD_PROJECT_NAME}
)
