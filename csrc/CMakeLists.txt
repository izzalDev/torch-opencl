set(LIBRARY_NAME torch_opencl)

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(APPLE)
  set(CL_VERSION 120)
  target_link_libraries(${LIBRARY_NAME} PRIVATE "-framework OpenCL")
else()
  set(CL_VERSION 300)
  target_link_libraries(${LIBRARY_NAME} PRIVATE OpenCL::OpenCL)
  set_target_properties(${LIBRARY_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS
                                                   ON)
endif()

target_compile_definitions(
  ${LIBRARY_NAME}
  PRIVATE CL_HPP_ENABLE_EXCEPTIONS CL_HPP_MINIMUM_OPENCL_VERSION=120
          CL_TARGET_OPENCL_VERSION=${CL_VERSION}
          CL_HPP_TARGET_OPENCL_VERSION=${CL_VERSION})

target_link_libraries(${LIBRARY_NAME} PUBLIC torch PRIVATE OpenCL::HeadersCpp)
install(
  TARGETS ${LIBRARY_NAME}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
