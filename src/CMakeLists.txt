# Native build configuration rewritten for Futurism.
# Replaces build support inherited through Kubism from KMCUDA.
# Pinned sources and file comparisons: docs/SOURCE_AUDIT.md.
# Source attribution and license scope: docs/PROVENANCE.md.

cmake_minimum_required(VERSION 3.10)
project(futurism LANGUAGES C CXX)

# Keep FindCUDA for the CUDA 11.4 / CMake 3.x Jetson build environment.
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(CUDA_ARCH "87" CACHE STRING "CUDA compute capability (Jetson AGX Orin: 87)")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Werror -DCUDA_ARCH=${CUDA_ARCH} ${OpenMP_CXX_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS
     "-arch=sm_${CUDA_ARCH}" "--std=c++14"
     "-Xptxas=-O3" "-Xptxas=-v"
     "-D_MWAITXINTRIN_H_INCLUDED" "-D_FORCE_INLINES" "-lineinfo")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-G" "-g")
endif()

cuda_add_library(futurism SHARED
    kmcuda.cc kmcuda.h kmeans.cu kmeans_cpu.cc kmeans_cpu.h
    knn.cu transpose.cu wrappers.h private.h fp_abstraction.h
    tricks.cuh metric_abstraction.h power_metrics.h)
target_link_libraries(futurism ${CUDA_curand_LIBRARY} OpenMP::OpenMP_CXX Threads::Threads)
