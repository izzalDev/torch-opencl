set(LIBRARY_NAME torch_opencl)
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library)

if(APPLE)
	target_link_libraries(${LIBRARY_NAME} PRIVATE "-framework OpenCL")
else()
	target_link_libraries(${LIBRARY_NAME} PRIVATE OpenCL::OpenCL)
	set_target_properties(${LIBRARY_NAME} PROPERTIES 
		WINDOWS_EXPORT_ALL_SYMBOLS ON 
	)
endif()
