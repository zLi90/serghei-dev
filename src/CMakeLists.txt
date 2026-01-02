# Set source code variable 
file(GLOB LIB_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

add_executable(${CMAKE_PROJECT_NAME} ${LIB_HEADERS} driver.cpp)

set_target_properties(
	${CMAKE_PROJECT_NAME}
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE MPI::MPI_CXX) 					# Link MPI

find_package(Kokkos)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Kokkos::kokkos) 				# Link Kokkos
if(SERGHEI_RE_MODEL)
    find_package(KokkosKernels)
	target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE     Kokkos::kokkoskernels)
endif()

# Conditionally link PnetCDF if available
if(SERGHEI_HAS_PNETCDF)
	if (NOT PnetCDF_C_FOUND)
		# use for local PnetCDF builds PkgConfig
		target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PnetCDF_INCLUDE_DIRS}) 	# Link PNetCDF includes
		target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${PnetCDF_LINK_LIBRARIES}) 	  	# Link PNetCDF library
	else ()
		# use for system PnetCDF builds FindPnetCDF.cmake
		target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PnetCDF_C_INCLUDE_DIRS}) # Link PNetCDF includes
		target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${PnetCDF_C_LIBRARIES}) 	  	# Link PNetCDF library
	endif()
endif()

add_custom_command(TARGET serghei POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:serghei> ${PROJECT_SOURCE_DIR}/bin/serghei
    COMMENT "SERGHEI executable in ${PROJECT_SOURCE_DIR}/bin/serghei"
)
