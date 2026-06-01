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
set(CMAKE_CXX_STANDARD 20)
set(CUDA_STANDARD 20)
set(CUDA_STANDARD_REQUIRED ON)

include(CMakeDependentOption)
include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

include(CheckLanguage)

# Options
option(TABASCAL_CPU "Build CPU extension" ON)

# GPU backends auto-enable when a usable toolchain is discoverable in the build
# environment, unless the user forces a value with -DTABASCAL_CUDA=ON/OFF (or the
# TABASCAL_CUDA env var, mapped to the cache define in pyproject.toml). This lets
# `pixi install -e cuda12` and a `pip install .` on a CUDA-capable host build the
# GPU kernels out of the box, while CPU-only hosts (no nvcc) stay CPU-only.
#
# The default flips ON only when BOTH the compiler (nvcc) AND the CUDA toolkit
# (cudart, found via CUDAToolkit) are present. A stray nvcc without the toolkit
# leaves the default OFF so a previously-working CPU build never starts failing;
# the find_package(... REQUIRED) error below is reserved for explicit opt-in.
if(NOT DEFINED TABASCAL_CUDA)
  check_language(CUDA)
  set(_TABASCAL_CUDA_DEFAULT OFF)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      message(STATUS "CUDA toolkit detected (${CMAKE_CUDA_COMPILER}); enabling CUDA extension")
      set(_TABASCAL_CUDA_DEFAULT ON)
    else()
      message(STATUS "nvcc found but CUDAToolkit incomplete; CUDA extension stays OFF "
                     "(use -DTABASCAL_CUDA=ON to force)")
    endif()
  endif()
else()
  set(_TABASCAL_CUDA_DEFAULT ${TABASCAL_CUDA})
endif()
option(TABASCAL_CUDA "Build CUDA extension" ${_TABASCAL_CUDA_DEFAULT})

# Mirror the same toolchain-presence gating for ROCm/HIP.
if(NOT DEFINED TABASCAL_ROCM)
  check_language(HIP)
  set(_TABASCAL_ROCM_DEFAULT OFF)
  if(CMAKE_HIP_COMPILER)
    find_package(hip CONFIG QUIET)
    if(hip_FOUND)
      message(STATUS "HIP toolkit detected (${CMAKE_HIP_COMPILER}); enabling ROCm extension")
      set(_TABASCAL_ROCM_DEFAULT ON)
    else()
      message(STATUS "HIP compiler found but hip package incomplete; ROCm extension stays OFF "
                     "(use -DTABASCAL_ROCM=ON to force)")
    endif()
  endif()
else()
  set(_TABASCAL_ROCM_DEFAULT ${TABASCAL_ROCM})
endif()
option(TABASCAL_ROCM "Build ROCm extension" ${_TABASCAL_ROCM_DEFAULT})

option(TABASCAL_BUNDLED_LIBS "Use all bundled libraries" ON)
cmake_dependent_option(TABASCAL_BUNDLED_HIGHWAY "Use bundled highway lib" ON "TABASCAL_BUNDLED_LIBS" OFF)
option(TABASCAL_MULTI_ARCH "Build kernels for multiple CPU architectues with dynamic dispatch. When disabled, arch flags should be set through CMAKE_CXX_FLAGS." ON)

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
  ./tabascal/components/ffi/util_gpu.cu
)

if(TABASCAL_CPU)
  add_library(tabascal SHARED ${TABASCAL_CPU_SOURCES})
  target_include_directories(tabascal PRIVATE ${JAXLIB_HOME}/include ./tabascal/components/ffi)

  target_compile_options(tabascal PRIVATE
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-return-type>
	$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-attributes>
  )

  if(TABASCAL_BUNDLED_HIGHWAY)
	# add google highway
	set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
	set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "")
	set(HWY_ENABLE_INSTALL OFF CACHE BOOL "")
	set(HWY_ENABLE_TESTS OFF CACHE BOOL "")
	set(HWY_FORCE_STATIC_LIBS ON CACHE BOOL "")
	set(HWY_ENABLE_CONTRIB ON CACHE BOOL "")
	FetchContent_Declare(
	  hwy
	  URL https://github.com/google/highway/archive/refs/tags/1.4.0.tar.gz
	  URL_MD5 9d335797777e17f827c7980b8313a34b
	)
	FetchContent_MakeAvailable(hwy)
	if(NOT TARGET hwy::hwy)
	  add_library(hwy::hwy ALIAS hwy)
	endif()
  else()
	find_package(hwy CONFIG REQUIRED)
  endif()
  target_link_libraries(tabascal PRIVATE hwy::hwy)

  if(TABASCAL_MULTI_ARCH)
	target_compile_definitions(tabascal PUBLIC -DTABASCAL_MULTI_ARCH)
  endif()
endif()

if(TABASCAL_CUDA)
  enable_language(CUDA)
  # find toolkit after language is enabled to ensure version matching
  find_package(CUDAToolkit REQUIRED)

  set_source_files_properties(${TABASCAL_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_library(tabascal_cuda SHARED ${TABASCAL_GPU_SOURCES})
  # set_target_properties(tabascal_cuda PROPERTIES OUTPUT_NAME "tabascal_cuda_${CUDAToolkit_VERSION_MAJOR}")
  target_include_directories(tabascal_cuda PRIVATE ${JAXLIB_HOME}/include ./tabascal/components/ffi)
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
  target_include_directories(tabascal_hip PRIVATE ${JAXLIB_HOME}/include ./tabascal/components/ffi)
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
