set(LIBRARY_NAME torch_opencl)

# Collect all .cpp files recursively; re-run cmake if new files are added
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

# PUBLIC: headers are also exposed to targets that link this library (e.g. _C)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(APPLE)
  # macOS ships OpenCL as a system framework; max supported version is 1.2
  set(CL_VERSION 120)
  target_link_libraries(${LIBRARY_NAME} PRIVATE "-framework OpenCL")
else()
  # Linux/Windows use the KhronosGroup OpenCL-SDK fetched by CPM
  set(CL_VERSION 300)
  target_link_libraries(${LIBRARY_NAME} PRIVATE OpenCL::OpenCL)
  set_target_properties(
    ${LIBRARY_NAME}
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON # Auto-export all symbols on
                                             # Windows
  )
endif()

target_compile_definitions(
  ${LIBRARY_NAME}
  PRIVATE CL_HPP_ENABLE_EXCEPTIONS # Enable C++ exceptions in cl2.hpp
          CL_HPP_MINIMUM_OPENCL_VERSION=120 # Suppress deprecation for pre-1.2
                                            # API
          CL_TARGET_OPENCL_VERSION=${CL_VERSION} # Platform OpenCL version
          CL_HPP_TARGET_OPENCL_VERSION=${CL_VERSION} # C++ wrapper OpenCL
                                                     # version
)

target_link_libraries(
  ${LIBRARY_NAME}
  PUBLIC torch # Propagated to _C so it doesn't need to re-link torch
  PRIVATE OpenCL::HeadersCpp # C++ OpenCL headers, only needed at compile time
)

# Install alongside _C so that RPATH @loader_path/$ORIGIN can find it at runtime
install(TARGETS ${LIBRARY_NAME} DESTINATION ${SKBUILD_PROJECT_NAME})
