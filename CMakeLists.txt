cmake_minimum_required(VERSION 3.21)
project(gpu_graph_drift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── GPU backend ───────────────────────────────────────────────────────────────
set(GPU_BACKEND "NONE" CACHE STRING "GPU backend: CUDA / ROCM / NONE")

option(RELAX_ATOMICS "E12.c: drop atomic CAS in d[] updates (SSSP experiment only)" OFF)
if(RELAX_ATOMICS)
    add_compile_definitions(RELAX_ATOMICS)
    message(WARNING "RELAX_ATOMICS=ON: d[] updates are non-atomic (E12.c experiment binary)")
endif()

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(USE_GPU TRUE)
    add_compile_definitions(USE_GPU)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA arch (default 86 = A10)")
    endif()
    message(STATUS "GPU backend: CUDA (nvcc) arch=${CMAKE_CUDA_ARCHITECTURES}")
elseif(GPU_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
    find_package(hip CONFIG REQUIRED)
    set(USE_GPU TRUE)
    add_compile_definitions(USE_GPU)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING "HIP arch (default gfx942 = MI300)")
    endif()
    message(STATUS "GPU backend: ROCM (hipcc) arch=${CMAKE_HIP_ARCHITECTURES}")
else()
    set(USE_GPU FALSE)
    message(STATUS "GPU backend: NONE (no GPU targets built)")
endif()

# ── Algorithm selection ───────────────────────────────────────────────────────
option(BUILD_PAGERANK "Build PageRank algorithms" ON)
option(BUILD_SSSP     "Build SSSP algorithms and harness" ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(OpenMP QUIET)
find_package(nlohmann_json 3.10 QUIET)
find_package(Catch2 3 QUIET)

# ── Core (header-only) ───────────────────────────────────────────────────────
add_library(core INTERFACE)
target_include_directories(core INTERFACE ${CMAKE_SOURCE_DIR}/src)

# ── PageRank kernels ─────────────────────────────────────────────────────────
if(USE_GPU AND BUILD_PAGERANK)
    add_executable(pagerank src/pagerank/pagerank.hip)
    target_link_libraries(pagerank PRIVATE core)

    add_executable(pagerank_pull src/pagerank/pagerank_pull.hip)
    target_link_libraries(pagerank_pull PRIVATE core)

    add_executable(pagerank_pull_v2 src/pagerank/pagerank_pull_v2.hip)
    target_link_libraries(pagerank_pull_v2 PRIVATE core)

    if(GPU_BACKEND STREQUAL "CUDA")
        set_source_files_properties(src/pagerank/pagerank.hip PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/pagerank/pagerank_pull.hip PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/pagerank/pagerank_pull_v2.hip PROPERTIES LANGUAGE CUDA)
        target_link_libraries(pagerank PRIVATE CUDA::cudart)
        target_link_libraries(pagerank_pull PRIVATE CUDA::cudart)
        target_link_libraries(pagerank_pull_v2 PRIVATE CUDA::cudart)
    else()
        set_source_files_properties(src/pagerank/pagerank.hip PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/pagerank/pagerank_pull.hip PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/pagerank/pagerank_pull_v2.hip PROPERTIES LANGUAGE HIP)
        target_link_libraries(pagerank PRIVATE hip::host)
        target_link_libraries(pagerank_pull PRIVATE hip::host)
        target_link_libraries(pagerank_pull_v2 PRIVATE hip::host)
    endif()
endif()

# ── IO (SSSP loaders) ────────────────────────────────────────────────────────
if(BUILD_SSSP)
add_library(io STATIC
    src/io/dimacs_loader.cpp
    src/io/gap_loader.cpp
    src/io/snap_loader.cpp
    src/io/rmat_generator.cpp
    src/io/csr_io.cpp
)
target_link_libraries(io PUBLIC core)

# ── SSSP ─────────────────────────────────────────────────────────────────────
set(SSSP_SOURCES
    src/sssp/cpu_dijkstra.cpp
    src/sssp/cpu_delta_stepping.cpp
)

if(USE_GPU)
    list(APPEND SSSP_SOURCES
        src/sssp/delta_stepping.hip
        src/sssp/bellman_ford.hip
        src/sssp/async_push_sssp.hip)
    if(GPU_BACKEND STREQUAL "CUDA")
        set_source_files_properties(src/sssp/delta_stepping.hip PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/sssp/bellman_ford.hip PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(src/sssp/async_push_sssp.hip PROPERTIES LANGUAGE CUDA)
    else()
        set_source_files_properties(src/sssp/delta_stepping.hip PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/sssp/bellman_ford.hip PROPERTIES LANGUAGE HIP)
        set_source_files_properties(src/sssp/async_push_sssp.hip PROPERTIES LANGUAGE HIP)
    endif()
endif()

add_library(sssp STATIC ${SSSP_SOURCES})
target_link_libraries(sssp PUBLIC core)

# ── Verifier ─────────────────────────────────────────────────────────────────
add_library(verifier STATIC
    src/verifier/cpu_verifier.cpp
)
target_link_libraries(verifier PUBLIC core)
if(OpenMP_CXX_FOUND)
    target_link_libraries(verifier PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── SSSP harness (run_sssp) ──────────────────────────────────────────────────
if(nlohmann_json_FOUND)
    add_executable(run_sssp
        src/harness/main.cpp
        src/harness/log_writer.cpp
        src/harness/error_injector.cpp
    )
    target_link_libraries(run_sssp PRIVATE io sssp verifier nlohmann_json::nlohmann_json)
    if(USE_GPU)
        if(GPU_BACKEND STREQUAL "CUDA")
            target_link_libraries(run_sssp PRIVATE CUDA::cudart)
        else()
            target_link_libraries(run_sssp PRIVATE hip::host)
        endif()
    endif()
else()
    message(STATUS "nlohmann_json not found — skipping run_sssp target")
endif()

endif() # BUILD_SSSP

# ── Tests ────────────────────────────────────────────────────────────────────
if(Catch2_FOUND AND BUILD_SSSP)
    include(CTest)
    include(Catch)

    add_executable(tests
        tests/test_csr.cpp
        tests/test_dijkstra_small.cpp
        tests/test_verifier_soundness.cpp
        tests/test_delta_stepping_cpu.cpp
        tests/test_csr_io.cpp
    )
    target_link_libraries(tests PRIVATE io sssp verifier Catch2::Catch2WithMain)
    catch_discover_tests(tests)
else()
    message(STATUS "Catch2 not found — skipping tests target")
endif()
