cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)

nanobind_add_module(_core src/main.cpp)

target_compile_features(_core PRIVATE cxx_std_17)
target_link_libraries(_core PRIVATE ${TORCH_LIBRARIES})

nanobind_add_stub(
  _core_stub
  MODULE _core
  OUTPUT _core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core
  MARKER_FILE py.typed
)

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/py.typed
  DESTINATION ${SKBUILD_PROJECT_NAME}
)
