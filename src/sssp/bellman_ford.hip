// bellman_ford.hip — GPU Bellman-Ford, HIP unified (NVIDIA + AMD)
//
// Algorithm: classical Bellman-Ford with edge-parallel relaxation.
//
// Each iteration:
//   For every vertex u with d[u] < INF, relax all out-edges (u→v, w):
//     attempt d[v] := min(d[v], d[u] + w) via atomicMinFP / atomicRelaxDPi32.
//   Set any_updated when any relaxation succeeds; terminate early when no
//   iteration produces an update.  Loop is capped at n_vertices iterations
//   (worst case for non-negative weights).
//
// Compared to delta_stepping.hip: no bucket structure, no Δ, no light/heavy
// split — every active vertex's full edge list is scanned every iteration.
// This is high-contention by design: it stresses the same atomic primitives
// (atomicMinFP for d, atomicRelaxDPi32 for packed (d,pi)) that Δ-stepping
// uses, so cross-algorithm d_hash agreement under strict atomics validates
// the §II.E claim that atomic discipline — not the algorithm — is the
// reduction-order-independence enabler.
//
// Build:
//   NVIDIA: hipcc --offload-arch=sm_80 -O2 bellman_ford.hip ...
//   AMD:    hipcc --offload-arch=gfx90a -O2 bellman_ford.hip ...

#if defined(__NVCC__) || defined(__CUDACC__)
#  include "../core/hip_cuda_compat.h"   // HIP → CUDA shim for nvcc builds
#else
#  include <hip/hip_runtime.h>   // native HIP for ROCm builds
#endif
#include "gpu_atomics.hpp"       // DPi32, atomicMinFP, atomicRelaxDPi32 (shared)
#include "cpu_delta_stepping.h"  // reconstruct_pi (post-hoc pi from final d[])
#include <type_traits>
#include "../core/csr_sssp.h"
#include <cstdio>
#include <climits>

#define HIP_CHECK(expr) do {                                        \
    hipError_t _e = (expr);                                         \
    if (_e != hipSuccess) {                                         \
        fprintf(stderr, "HIP error %s at %s:%d\n",                 \
                hipGetErrorString(_e), __FILE__, __LINE__);         \
        exit(1);                                                    \
    }                                                               \
} while (0)

// ── Kernels ───────────────────────────────────────────────────────────────────

template<typename W>
__global__ void kernel_bf_init(
    W* dist, vid_t* pi, vid_t n_v, vid_t source, W inv, bool emit)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dist[v] = inv;
    if (emit) pi[v] = INVALID_VID;
    if (v == source) dist[v] = W{0};
}

__global__ void kernel_bf_init_dpi32(
    DPi32* dpi, vid_t n_v, vid_t source, float inv)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dpi[v].d  = inv;
    dpi[v].pi = INVALID_VID;
    if (v == source) dpi[v].d = 0.0f;
}

// ── FP32 packed BF relaxation ────────────────────────────────────────────────
// One thread per source vertex u; relaxes all of u's out-edges into the
// packed (d, pi) array via atomicRelaxDPi32.  Skips u when d[u] == INF.
__global__ void kernel_bf_relax_dpi32(
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const float* __restrict__ weights,
    DPi32* dpi, vid_t n_v, float inv, int* any_updated)
{
    vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_v) return;

    float du = dpi[u].d;
    if (du >= inv) return;  // unreachable — no useful relaxations

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        float w  = weights[e];
        vid_t v  = cols[e];
        float nd = du + w;
        int outcome = atomicRelaxDPi32(&dpi[v], nd, u);
        if (outcome == 1 && any_updated) atomicOr(any_updated, 1);
    }
}

// ── Generic BF relaxation (FP64 always; FP32 non-EMIT) ───────────────────────
// One thread per source vertex u; relaxes all of u's out-edges via
// atomicMinFP for d, atomicExch for pi (mirrors delta_stepping.hip's
// kernel_relax body but without the LIGHT/HEAVY split — BF has no Δ).
template<typename W, bool EMIT>
__global__ void kernel_bf_relax(
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const W*     __restrict__ weights,
    W* dist, vid_t* pi, vid_t n_v, W inv, int* any_updated)
{
    vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_v) return;

    W du = dist[u];
    if (du >= inv) return;  // unreachable — no useful relaxations

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        W     w  = weights[e];
        vid_t v  = cols[e];
        W     nd = du + w;
        W     old = atomicMinFP(&dist[v], nd);

        if (nd < old) {
            // Strict improvement: overwrite predecessor unconditionally.
            // (We just set d[v] = nd via atomicMinFP; installing u is correct.)
            if (EMIT) atomicExch(&pi[v], u);
            if (any_updated) atomicOr(any_updated, 1);
        }
        // No tie-break here — pi is reconstructed post-hoc for the separate
        // path (mirrors delta_stepping_gpu_separate's policy).
    }
}

// ── Host driver: FP32 packed (in-flight emission) ────────────────────────────
// Used when W=float and emit_cert=true.  d and pi update atomically via
// 64-bit CAS, eliminating the dual-atomic race that would otherwise force
// post-hoc reconstruct_pi.
static Certificate<float> bellman_ford_gpu_packed_f32(
    const CSR<float>& g, vid_t source)
{
    const float INF   = Sentinel<float>::inf;
    const int   BLOCK = 256;
    const vid_t NV = g.n_vertices;
    const eid_t NE = g.n_edges;

    DPi32* d_dpi;     HIP_CHECK(hipMalloc(&d_dpi,     NV * sizeof(DPi32)));
    eid_t* d_row;     HIP_CHECK(hipMalloc(&d_row,    (NV+1) * sizeof(eid_t)));
    vid_t* d_col;     HIP_CHECK(hipMalloc(&d_col,     NE * sizeof(vid_t)));
    float* d_wgt;     HIP_CHECK(hipMalloc(&d_wgt,     NE * sizeof(float)));
    int*   d_updated; HIP_CHECK(hipMalloc(&d_updated, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(), NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),     NE*sizeof(float), hipMemcpyHostToDevice));

    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_bf_init_dpi32, grid_v, BLOCK, 0, 0,
                       d_dpi, NV, source, INF);
    HIP_CHECK(hipDeviceSynchronize());

    // BF outer loop: cap at NV iterations (worst case for non-negative weights);
    // early-terminate when no update happens in an iteration.
    for (vid_t iter = 0; iter < NV; ++iter) {
        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_updated, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(kernel_bf_relax_dpi32, grid_v, BLOCK, 0, 0,
            d_row, d_col, d_wgt, d_dpi, NV, INF, d_updated);
        HIP_CHECK(hipDeviceSynchronize());

        int any;
        HIP_CHECK(hipMemcpy(&any, d_updated, sizeof(int), hipMemcpyDeviceToHost));
        if (!any) break;
    }

    // Download packed dpi and unpack into separate d, pi for the host
    std::vector<DPi32> host_dpi(NV);
    HIP_CHECK(hipMemcpy(host_dpi.data(), d_dpi, NV * sizeof(DPi32), hipMemcpyDeviceToHost));
    Certificate<float> cert;
    cert.d.resize(NV);
    cert.pi.resize(NV);
    for (vid_t v = 0; v < NV; ++v) {
        cert.d[v]  = host_dpi[v].d;
        cert.pi[v] = host_dpi[v].pi;
    }

    hipFree(d_dpi);
    hipFree(d_row); hipFree(d_col); hipFree(d_wgt);
    hipFree(d_updated);
    return cert;
}

// ── Host driver: separate (existing) ─────────────────────────────────────────
// Used for FP64 always (96-bit packed atomic infeasible) and FP32 non-emit.
// For FP64 + emit_cert=true: post-hoc reconstruct_pi rebuilds pi from final d[].
template<typename W>
static Certificate<W> bellman_ford_gpu_separate(
    const CSR<W>& g, vid_t source, bool emit_cert)
{
    const W   INF   = Sentinel<W>::inf;
    const int BLOCK = 256;
    const vid_t NV  = g.n_vertices;
    const eid_t NE  = g.n_edges;

    // Device allocations
    W*     d_dist;      HIP_CHECK(hipMalloc(&d_dist,    NV * sizeof(W)));
    vid_t* d_pi = nullptr;
    if (emit_cert)     { HIP_CHECK(hipMalloc(&d_pi,     NV * sizeof(vid_t))); }
    eid_t* d_row;       HIP_CHECK(hipMalloc(&d_row,    (NV+1) * sizeof(eid_t)));
    vid_t* d_col;       HIP_CHECK(hipMalloc(&d_col,     NE * sizeof(vid_t)));
    W*     d_wgt;       HIP_CHECK(hipMalloc(&d_wgt,     NE * sizeof(W)));
    int*   d_updated;   HIP_CHECK(hipMalloc(&d_updated, sizeof(int)));

    // Upload graph
    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(),     NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),         NE*sizeof(W),     hipMemcpyHostToDevice));

    // Init
    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_bf_init<W>, grid_v, BLOCK, 0, 0,
                       d_dist, d_pi, NV, source, INF, emit_cert);
    HIP_CHECK(hipDeviceSynchronize());

    // BF outer loop: cap at NV iterations; early-terminate when no update.
    for (vid_t iter = 0; iter < NV; ++iter) {
        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_updated, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        if (emit_cert)
            hipLaunchKernelGGL((kernel_bf_relax<W, true>), grid_v, BLOCK, 0, 0,
                d_row, d_col, d_wgt, d_dist, d_pi, NV, INF, d_updated);
        else
            hipLaunchKernelGGL((kernel_bf_relax<W, false>), grid_v, BLOCK, 0, 0,
                d_row, d_col, d_wgt, d_dist, nullptr, NV, INF, d_updated);
        HIP_CHECK(hipDeviceSynchronize());

        int any;
        HIP_CHECK(hipMemcpy(&any, d_updated, sizeof(int), hipMemcpyDeviceToHost));
        if (!any) break;
    }

    // Download distances (always); reconstruct pi on CPU for consistency.
    // In-flight GPU pi (dual-atomic) can be inconsistent with final d[] due to
    // races between atomicMinFP and atomicExch. Post-hoc reconstruct_pi() is
    // O(E) and guaranteed consistent (§3.6 fallback).
    Certificate<W> cert;
    cert.d.resize(NV);
    cert.pi.assign(NV, INVALID_VID);
    HIP_CHECK(hipMemcpy(cert.d.data(), d_dist, NV*sizeof(W), hipMemcpyDeviceToHost));
    if (emit_cert)
        reconstruct_pi(g, source, cert.d, cert.pi);

    // Free
    hipFree(d_dist); if (d_pi) hipFree(d_pi);
    hipFree(d_row); hipFree(d_col); hipFree(d_wgt);
    hipFree(d_updated);

    return cert;
}

// ── Top-level dispatcher ─────────────────────────────────────────────────────
// FP32 + emit_cert=true → packed atomic (in-flight, race-free)
// All other cases     → separate arrays (FP64: post-hoc reconstruct_pi for emit)
template<typename W>
Certificate<W> bellman_ford_gpu(
    const CSR<W>& g, vid_t source, bool emit_cert)
{
    if constexpr (std::is_same_v<W, float>) {
        if (emit_cert)
            return bellman_ford_gpu_packed_f32(g, source);
    }
    return bellman_ford_gpu_separate<W>(g, source, emit_cert);
}

template Certificate<float>  bellman_ford_gpu<float> (const CSR<float>&,  vid_t, bool);
template Certificate<double> bellman_ford_gpu<double>(const CSR<double>&, vid_t, bool);
