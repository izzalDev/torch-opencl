set(LIBRARY_NAME _C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

nanobind_add_module(${LIBRARY_NAME} _C.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})

# import the package during stub generation. On Windows, importing _C requires
# torch to be imported first (via __init__.py) to register PyTorch DLL paths.
# scikit-build-core already packages __init__.py into the wheel separately.
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../__init__.py
    DESTINATION ${SKBUILD_PROJECT_NAME}
)

nanobind_add_stub(
    _C_stub
    INSTALL_TIME
    MODULE torch_opencl._C
    OUTPUT ${SKBUILD_PROJECT_NAME}/_C.pyi
    PYTHON_PATH .
    MARKER_FILE ${SKBUILD_PROJECT_NAME}/py.typed
)
