cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Parse version from pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_TOML)
string(REGEX MATCH "version = \"([^\"]+)\"" _ "${PYPROJECT_TOML}")
set(TABASCAL_VERSION "${CMAKE_MATCH_1}")

project(tabascal LANGUAGES CXX VERSION "${TABASCAL_VERSION}")
set(TABASCAL_SO_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})


# set default build type to RELEASE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo"
	)
endif()


# set language and standard
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)

include(CMakeDependentOption)

# Options
set(TABASCAL_GPU "OFF" CACHE STRING "GPU backend")
set_property(CACHE TABASCAL_GPU PROPERTY STRINGS
	"OFF" "CUDA" "ROCM"
	)

set(CMAKE_CUDA_ARCHITECTURES "80;90a-real;90" CACHE STRING "CUDA Architectures")

set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1030;gfx1100" CACHE STRING "HIP Architectures")

set(Python_EXECUTABLE "python3" CACHE STRING "The python interpreter")

# Options combination check
set(TABASCAL_CUDA OFF)
set(TABASCAL_ROCM OFF)
if(TABASCAL_GPU)
	if(TABASCAL_GPU STREQUAL "CUDA")
		set(TABASCAL_CUDA ON)
	elseif(TABASCAL_GPU STREQUAL "ROCM")
		set(TABASCAL_ROCM ON)
	else()
		message(FATAL_ERROR "Invalid GPU backend")
	endif()
endif()

# Detect jaxlib include directory (override with -DJAXLIB_HOME=...)
if(NOT DEFINED JAXLIB_HOME)
	execute_process(
		COMMAND ${Python_EXECUTABLE} -c
			"import jaxlib, os; print(os.path.dirname(jaxlib.__file__))"
		OUTPUT_VARIABLE JAXLIB_HOME
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE _jaxlib_result
	)
	if(NOT _jaxlib_result EQUAL 0)
		message(FATAL_ERROR "Failed to locate jaxlib via python3. "
			"Install jaxlib or pass -DJAXLIB_HOME=<path>.")
	endif()
endif()
message(STATUS "Using JAXLIB_HOME: ${JAXLIB_HOME}")


set(TABASCAL_CPU_SOURCES
  ./tabascal/components/ffi/rfi_kernel.cpp
  ./tabascal/components/ffi/rfi_jvp_kernel.cpp
  ./tabascal/components/ffi/rfi_transpose_kernel.cpp
)

set(TABASCAL_GPU_SOURCES
  ./tabascal/components/ffi/rfi_kernel_gpu.cu
  ./tabascal/components/ffi/rfi_jvp_kernel_gpu.cu
  ./tabascal/components/ffi/rfi_transpose_kernel_gpu.cu
)

add_library(tabascal SHARED ${TABASCAL_CPU_SOURCES})
target_include_directories(tabascal PRIVATE ${JAXLIB_HOME}/include)

target_compile_options(tabascal PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-return-type>
  $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-attributes>
)

if(TABASCAL_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set_source_files_properties(${TABASCAL_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_library(tabascal_cuda SHARED ${TABASCAL_GPU_SOURCES})
  target_include_directories(tabascal_cuda PRIVATE ${JAXLIB_HOME}/include)
  target_link_libraries(tabascal_cuda PRIVATE CUDA::cudart)
  target_compile_options(tabascal_cuda PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=940>
	$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2473>
  )
endif()

if(TABASCAL_ROCM)
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)

  set_source_files_properties(${TABASCAL_GPU_SOURCES} PROPERTIES LANGUAGE HIP)

  add_library(tabascal_hip SHARED ${TABASCAL_GPU_SOURCES})
  target_include_directories(tabascal_hip PRIVATE ${JAXLIB_HOME}/include)
  target_link_libraries(tabascal_hip PRIVATE hip::host)
endif()
