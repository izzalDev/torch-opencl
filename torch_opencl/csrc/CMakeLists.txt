CPMAddPackage("gh:wjakob/nanobind@2.12.0")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(LIBRARY_NAME _C)

nanobind_add_module(${LIBRARY_NAME} Module.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)

if(APPLE)
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

nanobind_add_stub(
    ${LIBRARY_NAME}_stub
    MODULE ${LIBRARY_NAME}
    OUTPUT ${LIBRARY_NAME}.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:${LIBRARY_NAME}>
    DEPENDS ${LIBRARY_NAME}
    MARKER_FILE py.typed
)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.pyi
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${SKBUILD_PROJECT_NAME}
)
