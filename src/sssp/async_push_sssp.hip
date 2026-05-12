// async_push_sssp.hip — GPU asynchronous push-based SSSP, HIP unified.
//
// Algorithm (textbook chaotic relaxation):
//
//   active[v] = (v == source)
//   d[v] = (v == source) ? 0 : +inf
//   loop:
//     any_active = 0
//     for each u with active[u] == true (parallel):
//       active[u] = false   (one-shot consume)
//       for each (u → v, w) in u's out-edges:
//         try atomic relaxation: d[v] = min(d[v], d[u] + w)
//         if relaxation actually lowered d[v]:
//           active[v] = 1
//           any_active = 1
//     if any_active == 0: break
//
// Like Bellman-Ford: iterate-to-fixedpoint termination on a global atomic
// flag.  Like Δ-stepping: only active vertices participate per round (no
// full vertex sweep every iteration).  No bucket structure, no Δ heuristic.
//
// Used for §5 boundary characterization as the third algorithm instance
// (A1.38 / F25): Lemma 5.1 predicts race-tolerance under relaxed per-vertex
// stores because the (any_active == 0) check still re-validates the global
// edge inequality at termination, just like Bellman-Ford's any_updated.

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

// ── Init kernels ──────────────────────────────────────────────────────────────
// Double-buffered active arrays: kernel reads active_in (read-only this round)
// and writes active_out (write-only).  Host swaps them between iterations and
// zeroes the new active_out.  This avoids the race where one thread clearing
// active[u] and another thread setting active[u] (re-arm) lose each other.

template<typename W>
__global__ void kernel_apush_init(
    W* dist, vid_t* pi, uint8_t* active_in, uint8_t* active_out,
    vid_t n_v, vid_t source, W inv, bool emit)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dist[v] = inv;
    if (emit) pi[v] = INVALID_VID;
    active_in[v]  = (v == source) ? (uint8_t)1 : (uint8_t)0;
    active_out[v] = 0;
    if (v == source) dist[v] = W{0};
}

__global__ void kernel_apush_init_dpi32(
    DPi32* dpi, uint8_t* active_in, uint8_t* active_out,
    vid_t n_v, vid_t source, float inv)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dpi[v].d  = inv;
    dpi[v].pi = INVALID_VID;
    active_in[v]  = (v == source) ? (uint8_t)1 : (uint8_t)0;
    active_out[v] = 0;
    if (v == source) dpi[v].d = 0.0f;
}

// Reset kernel: clear active_out at the start of each iteration (so that
// active_out only holds the "newly armed" set for THIS round).  Run with
// grid_v threads.
__global__ void kernel_apush_clear(uint8_t* active_out, vid_t n_v) {
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n_v) active_out[v] = 0;
}

// ── FP32 packed (in-flight (d, pi) emission via 64-bit CAS) ──────────────────
//
// One thread per source vertex u; processes only those with active_in[u] == 1.
// Race-free: kernel only READS active_in and only WRITES active_out (writes
// are 1-byte stores of constant value 1 → no lost-update concern across
// concurrent writers to the same v).
__global__ void kernel_apush_relax_dpi32(
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const float* __restrict__ weights,
    DPi32* dpi,
    const uint8_t* __restrict__ active_in,
    uint8_t* active_out,
    vid_t n_v, float inv, int* any_active)
{
    vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_v) return;
    if (active_in[u] == 0) return;

    float du = dpi[u].d;
    if (du >= inv) return;

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        float w  = weights[e];
        vid_t v  = cols[e];
        float nd = du + w;
        int outcome = atomicRelaxDPi32(&dpi[v], nd, u);
        if (outcome == 1) {
            active_out[v] = 1;
            atomicOr(any_active, 1);
        }
    }
}

// ── Generic relaxation (FP64 always; FP32 non-EMIT) ──────────────────────────
template<typename W, bool EMIT>
__global__ void kernel_apush_relax(
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const W*     __restrict__ weights,
    W* dist, vid_t* pi,
    const uint8_t* __restrict__ active_in,
    uint8_t* active_out,
    vid_t n_v, W inv, int* any_active)
{
    vid_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_v) return;
    if (active_in[u] == 0) return;

    W du = dist[u];
    if (du >= inv) return;

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        W     w  = weights[e];
        vid_t v  = cols[e];
        W     nd = du + w;
        W     old = atomicMinFP(&dist[v], nd);

        if (nd < old) {
            if (EMIT) atomicExch(&pi[v], u);
            active_out[v] = 1;
            atomicOr(any_active, 1);
        }
    }
}

// ── Host driver: FP32 packed (in-flight emission, race-free pi) ──────────────
static Certificate<float> async_push_sssp_gpu_packed_f32(
    const CSR<float>& g, vid_t source)
{
    const float INF   = Sentinel<float>::inf;
    const int   BLOCK = 256;
    const vid_t NV = g.n_vertices;
    const eid_t NE = g.n_edges;

    DPi32*   d_dpi;        HIP_CHECK(hipMalloc(&d_dpi,        NV * sizeof(DPi32)));
    uint8_t* d_active_in;  HIP_CHECK(hipMalloc(&d_active_in,  NV * sizeof(uint8_t)));
    uint8_t* d_active_out; HIP_CHECK(hipMalloc(&d_active_out, NV * sizeof(uint8_t)));
    eid_t*   d_row;        HIP_CHECK(hipMalloc(&d_row,       (NV+1) * sizeof(eid_t)));
    vid_t*   d_col;        HIP_CHECK(hipMalloc(&d_col,        NE * sizeof(vid_t)));
    float*   d_wgt;        HIP_CHECK(hipMalloc(&d_wgt,        NE * sizeof(float)));
    int*     d_any;        HIP_CHECK(hipMalloc(&d_any,        sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(), NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),     NE*sizeof(float), hipMemcpyHostToDevice));

    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_apush_init_dpi32, grid_v, BLOCK, 0, 0,
                       d_dpi, d_active_in, d_active_out, NV, source, INF);
    HIP_CHECK(hipDeviceSynchronize());

    // Outer loop: cap at NV iterations as a runaway-protection guard
    // (Bellman-style worst case for non-negative weights).  Real termination
    // happens when any_active observes 0 after the post-kernel barrier.
    for (vid_t iter = 0; iter < NV; ++iter) {
        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_any, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        // Clear active_out (= "newly armed" buffer for this iteration)
        hipLaunchKernelGGL(kernel_apush_clear, grid_v, BLOCK, 0, 0,
                           d_active_out, NV);
        // (no sync — both kernels read d_dpi/d_active_in or write d_active_out;
        //  the kernel below only writes d_active_out, no read-after-write hazard
        //  on the freshly-cleared buffer because the clear happens-before the
        //  relax in stream order.)

        hipLaunchKernelGGL(kernel_apush_relax_dpi32, grid_v, BLOCK, 0, 0,
            d_row, d_col, d_wgt, d_dpi,
            d_active_in, d_active_out, NV, INF, d_any);
        HIP_CHECK(hipDeviceSynchronize());

        int any;
        HIP_CHECK(hipMemcpy(&any, d_any, sizeof(int), hipMemcpyDeviceToHost));
        if (!any) break;

        // Swap: next iteration reads what we just wrote
        std::swap(d_active_in, d_active_out);
    }

    std::vector<DPi32> host_dpi(NV);
    HIP_CHECK(hipMemcpy(host_dpi.data(), d_dpi, NV * sizeof(DPi32), hipMemcpyDeviceToHost));
    Certificate<float> cert;
    cert.d.resize(NV);
    cert.pi.resize(NV);
    for (vid_t v = 0; v < NV; ++v) {
        cert.d[v]  = host_dpi[v].d;
        cert.pi[v] = host_dpi[v].pi;
    }

    hipFree(d_dpi); hipFree(d_active_in); hipFree(d_active_out);
    hipFree(d_row); hipFree(d_col); hipFree(d_wgt);
    hipFree(d_any);
    return cert;
}

// ── Host driver: separate arrays (FP64 always; FP32 non-emit) ────────────────
template<typename W>
static Certificate<W> async_push_sssp_gpu_separate(
    const CSR<W>& g, vid_t source, bool emit_cert)
{
    const W   INF   = Sentinel<W>::inf;
    const int BLOCK = 256;
    const vid_t NV  = g.n_vertices;
    const eid_t NE  = g.n_edges;

    W*       d_dist;       HIP_CHECK(hipMalloc(&d_dist,       NV * sizeof(W)));
    vid_t*   d_pi = nullptr;
    if (emit_cert) {       HIP_CHECK(hipMalloc(&d_pi,         NV * sizeof(vid_t))); }
    uint8_t* d_active_in;  HIP_CHECK(hipMalloc(&d_active_in,  NV * sizeof(uint8_t)));
    uint8_t* d_active_out; HIP_CHECK(hipMalloc(&d_active_out, NV * sizeof(uint8_t)));
    eid_t*   d_row;        HIP_CHECK(hipMalloc(&d_row,       (NV+1) * sizeof(eid_t)));
    vid_t*   d_col;        HIP_CHECK(hipMalloc(&d_col,        NE * sizeof(vid_t)));
    W*       d_wgt;        HIP_CHECK(hipMalloc(&d_wgt,        NE * sizeof(W)));
    int*     d_any;        HIP_CHECK(hipMalloc(&d_any,        sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(),     NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),         NE*sizeof(W),     hipMemcpyHostToDevice));

    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_apush_init<W>, grid_v, BLOCK, 0, 0,
                       d_dist, d_pi, d_active_in, d_active_out, NV, source, INF, emit_cert);
    HIP_CHECK(hipDeviceSynchronize());

    for (vid_t iter = 0; iter < NV; ++iter) {
        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_any, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(kernel_apush_clear, grid_v, BLOCK, 0, 0,
                           d_active_out, NV);

        if (emit_cert)
            hipLaunchKernelGGL((kernel_apush_relax<W, true>), grid_v, BLOCK, 0, 0,
                d_row, d_col, d_wgt, d_dist, d_pi,
                d_active_in, d_active_out, NV, INF, d_any);
        else
            hipLaunchKernelGGL((kernel_apush_relax<W, false>), grid_v, BLOCK, 0, 0,
                d_row, d_col, d_wgt, d_dist, nullptr,
                d_active_in, d_active_out, NV, INF, d_any);
        HIP_CHECK(hipDeviceSynchronize());

        int any;
        HIP_CHECK(hipMemcpy(&any, d_any, sizeof(int), hipMemcpyDeviceToHost));
        if (!any) break;

        std::swap(d_active_in, d_active_out);
    }

    Certificate<W> cert;
    cert.d.resize(NV);
    cert.pi.assign(NV, INVALID_VID);
    HIP_CHECK(hipMemcpy(cert.d.data(), d_dist, NV*sizeof(W), hipMemcpyDeviceToHost));
    if (emit_cert)
        reconstruct_pi(g, source, cert.d, cert.pi);

    hipFree(d_dist); if (d_pi) hipFree(d_pi);
    hipFree(d_active_in); hipFree(d_active_out);
    hipFree(d_row); hipFree(d_col); hipFree(d_wgt);
    hipFree(d_any);

    return cert;
}

// ── Top-level dispatcher ─────────────────────────────────────────────────────
template<typename W>
Certificate<W> async_push_sssp_gpu(
    const CSR<W>& g, vid_t source, bool emit_cert)
{
    if constexpr (std::is_same_v<W, float>) {
        if (emit_cert)
            return async_push_sssp_gpu_packed_f32(g, source);
    }
    return async_push_sssp_gpu_separate<W>(g, source, emit_cert);
}

template Certificate<float>  async_push_sssp_gpu<float> (const CSR<float>&,  vid_t, bool);
template Certificate<double> async_push_sssp_gpu<double>(const CSR<double>&, vid_t, bool);
