#pragma once
// gpu_atomics.hpp — shared GPU atomic helpers for SSSP relaxations.
//
// Extracted from delta_stepping.hip so bellman_ford.hip (and any future GPU
// SSSP variant) can reuse the same primitives.  Behaviour is identical to the
// originals — see the comments in-line for the RELAX_ATOMICS experiment
// branches.

#if defined(__NVCC__) || defined(__CUDACC__)
#  include "../core/hip_cuda_compat.h"   // HIP → CUDA shim for nvcc builds
#else
#  include <hip/hip_runtime.h>   // native HIP for ROCm builds
#endif
#include "../core/csr_sssp.h"

// ── FP atomicMin via CAS ──────────────────────────────────────────────────────
// E12.c experiment: when RELAX_ATOMICS is defined, the CAS loop is replaced by
// a plain non-atomic load + non-atomic store (early-out preserved).  This
// breaks the min-reduction invariant under concurrent writers: last-writer
// wins regardless of value, so d[] becomes non-deterministic across runs and
// across vendors.  Used to demonstrate that atomic CAS — not the algorithm
// itself — is what makes Δ-stepping's d[] reduction-order-independent.

#ifdef RELAX_ATOMICS
__device__ inline float atomicMinFP(float* addr, float val) {
    float cur = *addr;
    if (cur <= val) return cur;
    *addr = val;
    return val;
}

__device__ inline double atomicMinFP(double* addr, double val) {
    double cur = *addr;
    if (cur <= val) return cur;
    *addr = val;
    return val;
}
#else
__device__ inline float atomicMinFP(float* addr, float val) {
    unsigned* a = reinterpret_cast<unsigned*>(addr);
    unsigned  old = *a, assumed;
    do {
        assumed = old;
        if (__uint_as_float(old) <= val) break;
        old = atomicCAS(a, assumed, __float_as_uint(val));
    } while (assumed != old);
    return __uint_as_float(old);
}

__device__ inline double atomicMinFP(double* addr, double val) {
    unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long  old = *a, assumed;
    do {
        assumed = old;
        if (__longlong_as_double((long long)old) <= val) break;
        old = atomicCAS(a, assumed, (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double((long long)old);
}
#endif

// ── FP32 packed (d, pi) — in-flight emission via 64-bit CAS ──────────────────
// Eliminates the dual-atomic race in kernel_relax. FP64 cannot use this
// (would need 96-bit atomic) and falls back to post-hoc reconstruct_pi.

struct alignas(8) DPi32 {
    float d;
    vid_t pi;
};

__device__ inline unsigned long long pack_dpi32(float d, vid_t pi) {
    // Layout matches DPi32 on little-endian: low 32 = d, high 32 = pi
    return (unsigned long long)__float_as_uint(d)
         | ((unsigned long long)pi << 32);
}
__device__ inline DPi32 unpack_dpi32(unsigned long long packed) {
    DPi32 r;
    r.d  = __uint_as_float((unsigned)(packed & 0xFFFFFFFFu));
    r.pi = (vid_t)(packed >> 32);
    return r;
}

// Atomically update (d[v], pi[v]) at *p with (nd, u) on strict improvement.
// Returns 1 if updated, 0 otherwise.
//
// NOTE: no tie-break (nd == c.d → no update). FP32 rounding can make
// d[u]+w == d[v] for paths where d[u] is not actually a strict ancestor of
// v in the SSSP tree; applying the smallest-vid tie-break in that case
// creates pi cycles on large graphs (e.g. usa_road, 23.9M vertices).
// Strict-only update is cycle-free because every successful write requires
// nd < c.d, so pi[v] always points to a vertex with smaller d (a valid
// ancestor). Cross-platform pi may differ in FP-equal cases (which is
// already documented as acceptable in docs/ts/01_sssp_and_emission.md §5.3).
__device__ inline int atomicRelaxDPi32(DPi32* p, float nd, vid_t u) {
#ifdef RELAX_ATOMICS
    // E12.c: non-atomic relaxed update — early-out preserved, CAS loop dropped.
    auto* p64 = reinterpret_cast<unsigned long long*>(p);
    unsigned long long cur = *p64;
    DPi32 c = unpack_dpi32(cur);
    if (nd >= c.d) return 0;
    *p64 = pack_dpi32(nd, u);
    return 1;
#else
    auto* p64 = reinterpret_cast<unsigned long long*>(p);
    unsigned long long cur = *p64, assumed;
    do {
        assumed = cur;
        DPi32 c = unpack_dpi32(assumed);
        if (nd >= c.d) return 0;
        unsigned long long new_packed = pack_dpi32(nd, u);
        cur = atomicCAS(p64, assumed, new_packed);
    } while (assumed != cur);
    return 1;
#endif
}
