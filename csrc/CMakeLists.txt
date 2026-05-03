# 1. Setup OpenCL SDK
CPMAddPackage(
    NAME OpenCL-SDK
    GITHUB_REPOSITORY KhronosGroup/OpenCL-SDK
    VERSION 2025.07.23
    OPTIONS
        "OPENCL_SDK_BUILD_SAMPLES OFF"
        "OPENCL_SDK_BUILD_UTILITIES OFF"
)

find_package(Torch CONFIG REQUIRED)

# 2. Definisi Target (Harus di atas set_target_properties)
set(LIBRARY_NAME torch_opencl)
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

# 3. Pengaturan Properti (PENTING untuk Windows/MSVC)
if(WIN32)
    set_target_properties(${LIBRARY_NAME} PROPERTIES 
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# 4. Konfigurasi Target
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBRARY_NAME} PUBLIC 
    ${TORCH_LIBRARIES} 
    OpenCL::Headers
    OpenCL::OpenCL
)

# 5. Install
install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${SKBUILD_PROJECT_NAME}
    LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
    RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
)
