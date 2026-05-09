cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)

set(MODULE_NAME    _C)
set(STUB_TARGET    ${MODULE_NAME}_stub)
set(STUB_FILE      ${MODULE_NAME}.pyi)
set(PKG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${SKBUILD_PROJECT_NAME})

if(SKBUILD_STATE STREQUAL "editable")
    set(INSTALL_DIR ${PKG_SOURCE_DIR})
else()
    set(INSTALL_DIR ${SKBUILD_PROJECT_NAME})
endif()

nanobind_add_module(${MODULE_NAME} csrc/${MODULE_NAME}.cpp)
target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)
target_link_libraries(${MODULE_NAME} PRIVATE ${TORCH_LIBRARIES})

nanobind_add_stub(
    ${STUB_TARGET}
    MODULE      ${MODULE_NAME}
    OUTPUT      ${STUB_FILE}
    PYTHON_PATH $<TARGET_FILE_DIR:${MODULE_NAME}>
    DEPENDS     ${MODULE_NAME}
    MARKER_FILE py.typed
)

install(TARGETS ${MODULE_NAME} DESTINATION ${INSTALL_DIR})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${STUB_FILE}
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${INSTALL_DIR}
)
