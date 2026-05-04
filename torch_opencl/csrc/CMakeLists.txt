set(LIBRARY_NAME _C)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

nanobind_add_module(${LIBRARY_NAME} Module.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)
target_link_directories(${LIBRARY_NAME} PRIVATE ${SKBUILD_PROJECT_NAME}/lib)

set_target_properties(torch_opencl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:${LIBRARY_NAME}>
)

nanobind_add_stub(
    ${LIBRARY_NAME}_stub
    MODULE ${LIBRARY_NAME}
    OUTPUT ${LIBRARY_NAME}.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:${LIBRARY_NAME}>
    DEPENDS ${LIBRARY_NAME} torch_opencl
    MARKER_FILE py.typed
)

install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})

install(TARGETS torch_opencl
    LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
    RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}.pyi
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION ${SKBUILD_PROJECT_NAME}
)
