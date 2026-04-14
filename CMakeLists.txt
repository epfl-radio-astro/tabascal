cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Parse version from pyproject.toml
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_TOML)
string(REGEX MATCH "\nversion = \"([^\"]+)\"" _ "${PYPROJECT_TOML}")
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
option(TABASCAL_CPU "Build CPU extension" ON)
option(TABASCAL_CUDA "Build CUDA extension" OFF)
option(TABASCAL_ROCM "Build ROCm extension" OFF)

set(CMAKE_CUDA_ARCHITECTURES "80;90a-real;90" CACHE STRING "CUDA Architectures")
set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx1030;gfx1100" CACHE STRING "HIP Architectures")

set(Python_EXECUTABLE "python3" CACHE STRING "The python interpreter")

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

if(TABASCAL_CPU)
  add_library(tabascal SHARED ${TABASCAL_CPU_SOURCES})
  target_include_directories(tabascal PRIVATE ${JAXLIB_HOME}/include)

  target_compile_options(tabascal PRIVATE
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-return-type>
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-attributes>
  )
endif()

if(TABASCAL_CUDA)
  enable_language(CUDA)
  # find toolkit after language is enabled to ensure version matching
  find_package(CUDAToolkit REQUIRED)

  set_source_files_properties(${TABASCAL_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_library(tabascal_cuda SHARED ${TABASCAL_GPU_SOURCES})
  # set_target_properties(tabascal_cuda PROPERTIES OUTPUT_NAME "tabascal_cuda_${CUDAToolkit_VERSION_MAJOR}")
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

# Normalize shared-library suffix across platforms so Python loader finds
# libtabascal*.so on macOS as well as Linux.
if(TARGET tabascal)
  set_target_properties(tabascal PROPERTIES SUFFIX ".so")
endif()
if(TARGET tabascal_cuda)
  set_target_properties(tabascal_cuda PROPERTIES SUFFIX ".so")
endif()
if(TARGET tabascal_hip)
  set_target_properties(tabascal_hip PROPERTIES SUFFIX ".so")
endif()

# Install rules so scikit-build-core stages libraries into the wheel at the
# location where tabascal/components/ffi/rfi_vis_op.py loads them from.
if(TARGET tabascal)
  install(TARGETS tabascal
	LIBRARY DESTINATION tabascal/components/ffi
	RUNTIME DESTINATION tabascal/components/ffi)
endif()
if(TARGET tabascal_cuda)
  install(TARGETS tabascal_cuda
	LIBRARY DESTINATION tabascal/components/ffi
	RUNTIME DESTINATION tabascal/components/ffi)
endif()
if(TARGET tabascal_hip)
  install(TARGETS tabascal_hip
	LIBRARY DESTINATION tabascal/components/ffi
	RUNTIME DESTINATION tabascal/components/ffi)
endif()
