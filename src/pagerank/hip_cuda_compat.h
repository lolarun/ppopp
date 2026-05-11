#pragma once
// Thin HIP-to-CUDA compatibility shim.
// Included automatically when building with -DGPU_BACKEND=CUDA (no ROCm needed).
// Maps hip* API calls to their cuda* equivalents so pagerank.hip compiles
// unchanged with nvcc.
//
// Mirrors src/sssp/hip_cuda_compat.h in the sibling Paper 2.1 repo
// (asplos-27); kept independent rather than shared so the two repos stay
// physically separate per project policy.

#include <cuda_runtime.h>

// Error type and constants
using hipError_t = cudaError_t;
constexpr cudaError_t hipSuccess = cudaSuccess;
inline const char* hipGetErrorString(cudaError_t e) { return cudaGetErrorString(e); }
inline cudaError_t hipGetLastError() { return cudaGetLastError(); }

// Memory — template overload matches HIP's hipMalloc<T>(T**, size_t) signature
// (CUDA's cudaMalloc only accepts void**, HIP accepts any T**)
template<typename T>
inline cudaError_t hipMalloc(T** p, size_t sz) {
    return cudaMalloc(reinterpret_cast<void**>(p), sz);
}
inline cudaError_t hipFree(void* p) { return cudaFree(p); }
inline cudaError_t hipMemset(void* p, int v, size_t sz) { return cudaMemset(p, v, sz); }

// Memcpy direction aliases
constexpr cudaMemcpyKind hipMemcpyHostToDevice   = cudaMemcpyHostToDevice;
constexpr cudaMemcpyKind hipMemcpyDeviceToHost   = cudaMemcpyDeviceToHost;
constexpr cudaMemcpyKind hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

inline cudaError_t hipMemcpy(void* dst, const void* src, size_t sz, cudaMemcpyKind k) {
    return cudaMemcpy(dst, src, sz, k);
}

// Sync
inline cudaError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }

// Device properties — used by harness to record device.name in JSONL metadata
using hipDeviceProp_t = cudaDeviceProp;
inline cudaError_t hipGetDeviceProperties(hipDeviceProp_t* p, int dev) {
    return cudaGetDeviceProperties(p, dev);
}

// Kernel launch: hipLaunchKernelGGL(kernel, grid, block, shm, stream, args...)
//   → kernel<<<grid, block, shm, stream>>>(args...)
#define hipLaunchKernelGGL(kernel, grid, block, shm, stream, ...) \
    (kernel)<<<(grid), (block), (shm), (stream)>>>(__VA_ARGS__)
