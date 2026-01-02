#   This is the CMakeLists file for building and compiling SERGHEI
#   To use, open a terminal, cd to the SERGHEI directory and type 
#   cmake -S ./ -B build	
#   Then, type "make" to compile SERGHEI
#   Note: If you changed any cmake options herein, before rebuilding, delete "CMakeCache.txt" to erase old cmake settings

cmake_minimum_required (VERSION 3.20)
#IF(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
#  MESSAGE(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
#  CMAKE_POLICY(SET CMP0074 NEW)
#ENDIF()

#   If needed, specify the CXX compiler here, otherwise comment out these lines
set(CMAKE_C_COMPILER gcc-14)
set(CMAKE_CXX_COMPILER g++-14)

# Name the project
project (serghei LANGUAGES CXX C)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../install/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../install/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../install/bin)

# Enable color output for messages
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")
set(Gray        "${Esc}[90m")
set(DebugColour ${Magenta})
set(EnableColour ${Green})
set(DisableColour ${Yellow})
set(ncColour ${Cyan})
set(MiscColour ${Blue})
set(DefaultColour ${Gray}) # colour used for default options

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

# Specify if testing is required for build
if(NOT DEFINED SERGHEI_ENABLE_TESTS)
  set(SERGHEI_ENABLE_TESTS OFF)
endif()

#   Specify the path where Kokkos is built
set(Kokkos_ROOT "/Users/zhili/Codes/kokkos-install")

#   Specify the path where KokkosKernels is built
set(KokkosKernels_ROOT "/Users/zhili/Codes/kokkoskernels-install")

#   Specify if using GPU or not
set(Enable_GPU "OFF")

#	If using GPU, specify the architecture
if (Enable_GPU)
	if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES 80)
	endif()
endif()


#include (cmake/LibFind.cmake)
#include (cmake/LibCheck.cmake)
#include (cmake/FindPnetCDF.cmake)

# Find SERGHEI dependencies 
find_package(MPI REQUIRED COMPONENTS CXX)



# Option to enable/disable PNETCDF support
option(SERGHEI_ENABLE_PNETCDF "Enable PnetCDF support for NetCDF I/O" OFF)

if(SERGHEI_ENABLE_PNETCDF)
	# use for system PnetCDF builds
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
	include (cmake/LibFind.cmake)
	include (cmake/LibCheck.cmake)
	include (cmake/FindPnetCDF.cmake)

	find_package(PnetCDF QUIET COMPONENTS C)

	if (NOT PnetCDF_C_FOUND)
		message(STATUS "PnetCDF was not FOUND via find_package, falling back to pkgconfig")

		# use for local PnetCDF builds
		find_package(PkgConfig REQUIRED)
		# set environment variable pointing to the pnetcdf

		if (DEFINED PnetCDF_DIR)
			set(ENV{PKG_CONFIG_PATH} "${PnetCDF_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
			set(ENV{LD_LIBRARY_PATH} "${PnetCDF_DIR}/lib:$ENV{LD_LIBRARY_PATH}")
		endif()
		
		pkg_check_modules(PnetCDF QUIET pnetcdf)
	endif()

	if(PnetCDF_C_FOUND OR PnetCDF_FOUND)
		set(SERGHEI_HAS_PNETCDF ON)
		add_compile_definitions(SERGHEI_HAS_PNETCDF=1)
		message(STATUS ${EnableColour} "PnetCDF support enabled" ${ColourReset})
	else()
		set(SERGHEI_HAS_PNETCDF OFF)
		message(STATUS ${DisableColour} "PnetCDF NOT FOUND - NetCDF I/O disabled, using VTK output" ${ColourReset})
	endif()
else()
	set(SERGHEI_HAS_PNETCDF OFF)
	message(STATUS ${DisableColour} "PnetCDF support disabled by user - using VTK output" ${ColourReset})
endif()



execute_process(COMMAND git describe COMMAND tr -d '\n' OUTPUT_VARIABLE serghei_git_version)
add_compile_definitions(SERGHEI_GIT_VERSION="${serghei_git_version}")
message(STATUS "SERGHEI version: " ${serghei_git_version})

message(STATUS ${ColourBold} "SERGHEI OPTIONS" ${ColourReset} 
	" ("
	${DefaultColour} "Default " ${ColourReset} "| "
	${EnableColour} "Enabled " ${ColourReset} "| "
	${DisableColour} "Disabled " ${ColourReset} "| "
	${DebugColour} "Debug"
	${ColourReset}
	")"
)

option(SERGHEI_SWE_RE "Enable coupled shallow water and subsurface models" OFF)
if(SERGHEI_SWE_RE)
	add_compile_definitions(SERGHEI_SWE_RE=1)
	message(STATUS ${EnableColour} "Coupled shallow water and subsurface models enabled" ${ColourReset})
	set(SERGHEI_SWE_MODEL ON)
	set(SERGHEI_RE_MODEL ON)
else()
	add_compile_definitions(SERGHEI_SWE_RE=0)
	message(STATUS ${DefaultColour} "Coupled shallow water and subsurface models disabled" ${ColourReset})
endif()

option(SERGHEI_RE_MODEL "Enable subsurface model" OFF)
if(SERGHEI_RE_MODEL)
	add_compile_definitions(SERGHEI_RE_MODEL=1)
	if(NOT SERGHEI_SWE_RE)
		set(SERGHEI_SWE_MODEL OFF)
	endif()
	message(STATUS ${EnableColour} "Subsurface model enabled" ${ColourReset})
  # Specify the path where KokkosKernels is built
  #set(KokkosKernels_ROOT "/Users/zli/Codes/kokkoskernels4-omp")
endif()

option(SERGHEI_SWE_MODEL "Enable shallow water model" ON)
if(SERGHEI_SWE_MODEL)
	add_compile_definitions(SERGHEI_SWE_MODEL=1)
	message(STATUS ${DefaultColour} "Shallow water model enabled" ${ColourReset})
else()
	add_compile_definitions(SERGHEI_SWE_MODEL=0)
	message(STATUS ${DisableColour} "Shallow water model disabled" ${ColourReset})
endif()


option(SERGHEI_LPT "Enable particle tracking" OFF)
if(SERGHEI_LPT)
	add_compile_definitions(SERGHEI_LPT=1)
	message(STATUS ${EnableColour} "Particle tracking enabled" ${ColourReset})
endif()

option(SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION "Enable not hydrodynamic evolution" OFF)
if(SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION)
	add_compile_definitions(SERGHEI_HYDRODYNAMIC_NOT_EVOLUTION=1)
	message(STATUS ${EnableColour} "Not hydrodynamic evolution enabled" ${ColourReset})
endif()

option(SERGHEI_VEGETATION_MODEL "Enable vegetation model" OFF)
if(SERGHEI_VEGETATION_MODEL)
	add_compile_definitions(SERGHEI_VEGETATION_MODEL=1)
	message(STATUS ${EnableColour} "Vegetation model enabled" ${ColourReset})
endif()

option(SERGHEI_TOOLS "Enable additional tools" ON)
if(SERGHEI_TOOLS)
	message(STATUS ${DefaultColour} "Tools enabled" ${ColourReset})
else()
	message(STATUS ${DisableColour} "Tools disabled" ${ColourReset})
	add_compile_definitions(SERGHEI_TOOLS=0)
endif()

option(SERGHEI_FRICTION_MODEL "Enable friction model" ON)   
if(SERGHEI_FRICTION_MODEL)
	message(STATUS ${DefaultColour} "Friction model enabled (Manning default)" ${ColourReset})
else()
	message(STATUS ${DisableColour} "Friction model disabled" ${ColourReset})
	add_compile_definitions(SERGHEI_FRICTION_MODEL=0)
endif()

option(SERGHEI_FRICTION_CENTERED "Enabled centered friction discretisation" ON)   
if(SERGHEI_FRICTION_CENTERED)
	message(STATUS ${EnableColour} "Centered friction discretisation enabled" ${ColourReset})
	add_compile_definitions(SERGHEI_POINTWISE_FRICTION=1)
else()
	message(STATUS ${DefaultColour} "Centered friction discretisation disabled" ${ColourReset})
endif()

option(SERGHEI_FRICTION_DARCY_WEISBACH "Enable Darcy-Weisbach friction model" OFF)   
if(SERGHEI_FRICTION_DARCY_WEISBACH)
	add_compile_definitions(SERGHEI_FRICTION_MODEL=2)
	message(STATUS ${EnableColour} "Enabling Darcy-Weisbach friction model" ${ColourReset})
endif()

option(SERGHEI_FRICTION_CHEZY "Enable Chezy friction model" OFF)   
if(SERGHEI_FRICTION_CHEZY)
	add_compile_definitions(SERGHEI_FRICTION_MODEL=3)
	message(STATUS ${EnableColour} "Enabling Chezy friction model" ${ColourReset})
endif()


option(SERGHEI_MESH_UNIFORM "Set uniform cartesian grid mesh" ON)
if(SERGHEI_MESH_UNIFORM)
	message(STATUS ${DefaultColour} "Mesh set to uniform cartesian grid" ${ColourReset})
else()
	message(SEND_ERROR "Non-uniform mesh not available")
endif()

# Arithmetic precision options
option(SERGHEI_FP32 "Set single precision computations" OFF)
if(SERGHEI_FP32)
	message(STATUS ${MiscColour} "Single precision set" ${ColourReset})
	add_compile_definitions(SERGHEI_REAL=1)
	set(SERGHEI_NC_FLOAT ON)
	set(SERGHEI_NC_DOUBLE OFF)
else()
	message(STATUS ${DefaultColour} "Double precision set" ${ColourReset})
	add_compile_definitions(SERGHEI_REAL=2)
endif()

# I/O options
option(SERGHEI_MAXFLOOD "Enable maximum flood statistics" OFF)
if(SERGHEI_MAXFLOOD)
	message(STATUS ${EnableColour} "Enabling maximum flood statistics" ${ColourReset})
	add_compile_definitions(SERGHEI_MAXFLOOD=1)
endif()

option(SERGHEI_WRITE_HZ "Write water surface elevation output" OFF)
if(SERGHEI_WRITE_HZ)
	message(STATUS ${EnableColour} "Writing water surface elevation output" ${ColourReset})
	add_compile_definitions(SERGHEI_WRITE_HZ=1)
endif()

option(SERGHEI_INPUT_NETCDF "Enable NetCDF input files" OFF)
if(SERGHEI_INPUT_NETCDF)
	message(STATUS ${EnableColour} "NetCDF input enabled" ${ColourReset})
	add_compile_definitions(SERGHEI_INPUT_NETCDF=1)
endif()

option(SERGHEI_WRITE_SUBDOMS "Enable writing subdomain indices in NetCDF output" OFF)
if(SERGHEI_WRITE_SUBDOMS)
	message(STATUS ${EnableColour} "Subdomain indices enabled in NetCDF output" ${ColourReset})
	add_compile_definitions(SERGHEI_WRITE_SUBDOMS=1)
endif()

# NetCDF options
# A useful option for large problems is NC_64BIT_DATA.
option(SERGHEI_NC_CLOBBER  "Set NetCDF mode to NC_CLOBBER" ON)

option(SERGHEI_NC_64BIT_DATA  "Set NetCDF mode to NC_64BIT_DATA" OFF)
if(SERGHEI_NC_64BIT_DATA)
	add_compile_definitions(SERGHEI_NC_MODE=NC_64BIT_DATA)
	message(STATUS ${ncColour} "NetCDF mode set to NC_64BIT_DATA" ${ColourReset})
	set(SERGHEI_NC_CLOBBER OFF)
endif()

option(SERGHEI_NC_ENABLE_NAN "Enable writing NaN for cells outside the domain" ON)
if(SERGHEI_NC_ENABLE_NAN)
	message(STATUS ${DefaultColour} "NetCDF NaN enabled" ${ColourReset})
else()
	message(STATUS ${MiscColour} "NetCDF NaN disabled" ${ColourReset})
	add_compile_definitions(SERGHEI_NC_ENABLE_NAN=0)
endif()

# debugging options
option(SERGHEI_DEBUG_PARALLEL_DECOMPOSITION "Enable parallel decomposition debugging" OFF)
if(SERGHEI_DEBUG_PARALLEL_DECOMPOSITION)
	add_compile_definitions(SERGHEI_DEBUG_PARALLEL_DECOMPOSITION=1)
	message(STATUS ${DebugColour}   "Parallel decomposition debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_WORKFLOW "Enable workflow debugging" OFF)
if(SERGHEI_DEBUG_WORKFLOW)
	add_compile_definitions(SERGHEI_DEBUG_WORKFLOW=1)
	message(STATUS ${DebugColour} "Workflow debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_KOKKOS_SETUP  "Enable Kokkos setup debugging" OFF)
if(SERGHEI_DEBUG_KOKKOS_SETUP)
	add_compile_definitions(SERGHEI_DEBUG_KOKKOS_SETUP=1)
	message(STATUS ${DebugColour}  "Kokkos setup debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_BOUNDARY  "Enable boundary debugging" OFF)
if(SERGHEI_DEBUG_BOUNDARY)
	add_compile_definitions(SERGHEI_DEBUG_BOUNDARY=1)
	message(STATUS ${DebugColour}  "Boundary debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_DT "Enable timestep debugging" OFF)
if(SERGHEI_DEBUG_DT)
	add_compile_definitions(SERGHEI_DEBUG_DT=1)
	message(STATUS ${DebugColour}  "Time stepping debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_TOOLS  "Enable tools debugging" OFF)
if(SERGHEI_DEBUG_TOOLS)
	add_compile_definitions(SERGHEI_DEUBG_TOOLS=1)
	message(STATUS ${DebugColour}  "Tools debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_MASS_CONS "Enable mass conservation debugging" OFF)
if(SERGHEI_DEBUG_MASS_CONS)
	add_compile_definitions(SERGHEI_DEBUG_MASS_CONS=1)
	message(STATUS ${DebugColour}  "Mass conservation debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_INFILTRATION "Enable infiltration debugging" OFF)
if(SERGHEI_DEBUG_INFILTRATION)
	add_compile_definitions(SERGHEI_DEBUG_INFILTRATION=1)
	message(STATUS ${DebugColour}  "Infiltration debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()	

option(SERGHEI_DEBUG_MPI  "Enable MPI debugging" OFF)
if(SERGHEI_DEBUG_MPI)
	add_compile_definitions(SERGHEI_DEBUG_MPI=1)
	message(STATUS ${DebugColour}  "MPI debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_INPUT_NETCDF  "Enable NetCDF input debugging" OFF)
if(SERGHEI_DEBUG_INPUT_NETCDF)
	add_compile_definitions(SERGHEI_DEBUG_INPUT_NETCDF=1)
	message(STATUS ${DebugColour}  "NetCDF input debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SERGHEI_DEBUG_OUTPUT  "Enable output debugging" OFF)
if(SERGHEI_DEBUG_OUTPUT)
	add_compile_definitions(SERGHEI_DEBUG_OUTPUT=1)
	message(STATUS ${DebugColour}  "Output debugging enabled" ${ColourReset})
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	message(STATUS ${DebugColour} "Enabling debug mode due to enabled debugging flags" ${ColourReset})
endif()

# Add the subdirectory where the source codes are stored
add_subdirectory(src)
