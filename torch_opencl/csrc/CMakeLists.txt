set(LIBRARY_NAME _C)

# Development.Module: only Python headers needed; no need to link libpython
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Creates the Python extension module (_C.cpython-*.so/.dylib) via nanobind
nanobind_add_module(${LIBRARY_NAME} _C.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)

# RPATH so the dynamic linker finds libtorch_opencl at runtime without
# LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
#   macOS -> @loader_path (relative to the .so being loaded)
#   Linux -> $ORIGIN      (same meaning, different syntax)
set_target_properties(${LIBRARY_NAME} PROPERTIES
    INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})

# nanobind_add_stub needs __init__.py present at install time to import
# torch_opencl._C. On Windows this also registers PyTorch DLL search paths.
# We copy all .py files from the package directory, excluding csrc.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
    DESTINATION ${SKBUILD_PROJECT_NAME}
    FILES_MATCHING
    PATTERN "*.py"
		PATTERN "csrc" EXCLUDE
)

# Generate _C.pyi type stubs and a py.typed marker (PEP 561) at install time
nanobind_add_stub(
    _C_stub
    INSTALL_TIME
    MODULE torch_opencl._C
    OUTPUT ${SKBUILD_PROJECT_NAME}/_C.pyi
    PYTHON_PATH .
    MARKER_FILE ${SKBUILD_PROJECT_NAME}/py.typed
)
