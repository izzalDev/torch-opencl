set(LIBRARY_NAME _C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

nanobind_add_module(${LIBRARY_NAME} Module.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)

nanobind_add_stub(
    _C_stub
    MODULE _C
    OUTPUT _C.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_C>
    DEPENDS torch_opencl _C
    MARKER_FILE py.typed
)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi
    DESTINATION ${SKBUILD_PROJECT_NAME}
)
