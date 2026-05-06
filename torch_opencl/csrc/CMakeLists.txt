set(LIBRARY_NAME _C)

find_package( Python COMPONENTS Interpreter Development.Module REQUIRED)

nanobind_add_module(${LIBRARY_NAME} Module.cpp)
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_opencl)

install( TARGETS ${LIBRARY_NAME}  DESTINATION  ${SKBUILD_PROJECT_NAME})


