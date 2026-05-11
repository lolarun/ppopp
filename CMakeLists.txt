cmake_minimum_required(VERSION 3.21)
project(ppopp27_pagerank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Safety net for single-config generators (Make, Ninja). Without this, an
# unspecified build type means no -O flags on host code — a common silent
# benchmarking footgun.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── GPU backend ───────────────────────────────────────────────────────────────
# Mirrors the convention from sibling Paper 2.1 repo (asplos-27). One source
# tree builds for either NVIDIA (nvcc + hip_cuda_compat.h shim, no ROCm needed)
# or AMD (native ROCm hipcc).
#
# -DGPU_BACKEND=CUDA  : NVIDIA via nvcc
# -DGPU_BACKEND=ROCM  : AMD via ROCm hipcc
# -DGPU_BACKEND=NONE  : (reserved for future CPU-only utilities; no kernel target)
set(GPU_BACKEND "NONE" CACHE STRING "GPU backend: CUDA / ROCM / NONE")

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(USE_GPU TRUE)
    add_compile_definitions(USE_GPU)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Primary target A10 (sm_86); also handles A100 (80), L20/4090 (89), H100 (90), T4 (75)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA arch (default 86 = A10)")
    endif()
    message(STATUS "GPU backend: CUDA (nvcc) arch=${CMAKE_CUDA_ARCHITECTURES}")
elseif(GPU_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
    find_package(hip CONFIG REQUIRED)
    set(USE_GPU TRUE)
    add_compile_definitions(USE_GPU)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        # Primary target MI300X VF (gfx942); also handles MI200 (gfx90a)
        set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING "HIP arch (default gfx942 = MI300)")
    endif()
    message(STATUS "GPU backend: ROCM (hipcc) arch=${CMAKE_HIP_ARCHITECTURES}")
else()
    set(USE_GPU FALSE)
    message(STATUS "GPU backend: NONE (no GPU targets built)")
endif()

# ── Core (header-only) ────────────────────────────────────────────────────────
add_library(core INTERFACE)
target_include_directories(core INTERFACE ${CMAKE_SOURCE_DIR}/src)

# ── PageRank kernel + driver ─────────────────────────────────────────────────
if(USE_GPU)
    add_executable(pagerank src/pagerank/pagerank.hip)
    target_link_libraries(pagerank PRIVATE core)
    if(GPU_BACKEND STREQUAL "CUDA")
        # Compile .hip as CUDA; nvcc sees __CUDACC__ → hip_cuda_compat.h kicks in
        set_source_files_properties(src/pagerank/pagerank.hip PROPERTIES LANGUAGE CUDA)
        target_link_libraries(pagerank PRIVATE CUDA::cudart)
    else()
        set_source_files_properties(src/pagerank/pagerank.hip PROPERTIES LANGUAGE HIP)
        # hip::host (not hip::device) for executable: avoids leaking
        # --offload-arch into plain C++ compile commands (gotcha from Paper 2.1).
        target_link_libraries(pagerank PRIVATE hip::host)
    endif()
endif()
