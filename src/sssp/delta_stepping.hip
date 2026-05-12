// delta_stepping.hip — GPU Δ-stepping, HIP unified (NVIDIA + AMD)
//
// Algorithm: Meyer & Sanders (2003) Δ-stepping, GPU-parallel variant.
//
// Bucket structure:
//   bucket_id[v] = floor(d[v] / Δ)  when v is active, UINT32_MAX otherwise
//   We track the global minimum active bucket index i_min.
//   Each outer iteration processes all vertices in bucket i_min:
//     Phase A (repeat until stable): relax light edges (w ≤ Δ) from bucket i_min
//     Phase B: relax heavy edges (w > Δ) from the set of vertices removed in Phase A
//
// Build:
//   NVIDIA: hipcc --offload-arch=sm_80 -O2 delta_stepping.hip ...
//   AMD:    hipcc --offload-arch=gfx90a -O2 delta_stepping.hip ...

#if defined(__NVCC__) || defined(__CUDACC__)
#  include "../core/hip_cuda_compat.h"   // HIP → CUDA shim for nvcc builds
#else
#  include <hip/hip_runtime.h>   // native HIP for ROCm builds
#endif
#include "gpu_atomics.hpp"       // DPi32, atomicMinFP, atomicRelaxDPi32 (shared)
#include "cpu_delta_stepping.h"  // reconstruct_pi (post-hoc pi from final d[])
#include <type_traits>
#include "../core/csr_sssp.h"
#include "tiebreak.h"
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

// ── FP atomicMin via CAS ──────────────────────────────────────────────────────
// atomicMinFP(float*, float) and atomicMinFP(double*, double) live in
// gpu_atomics.hpp (shared with bellman_ford.hip).  See that header for the
// RELAX_ATOMICS experiment branches.

// ── Kernels ───────────────────────────────────────────────────────────────────

template<typename W>
__global__ void kernel_init(
    W* dist, vid_t* pi, int* bucket_id,
    vid_t n_v, vid_t source, W inv, bool emit)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dist[v]      = inv;
    bucket_id[v] = INT_MAX;
    if (emit) pi[v] = INVALID_VID;
    if (v == source) {
        dist[v]      = W{0};
        bucket_id[v] = 0;
    }
}

// Collect vertices currently in bucket i_min → frontier[], frontier_sz
__global__ void kernel_collect_frontier(
    const int* bucket_id, vid_t n_v, int i_min,
    vid_t* frontier, int* frontier_sz)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    if (bucket_id[v] == i_min) {
        int pos = atomicAdd(frontier_sz, 1);
        frontier[pos] = v;
    }
}

// Remove vertices from bucket (mark inactive)
__global__ void kernel_remove_from_bucket(
    int* bucket_id, const vid_t* frontier, int sz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sz) return;
    bucket_id[frontier[tid]] = INT_MAX;
}

// ── Tiebreak CAS helper ───────────────────────────────────────────────────────
// Install predecessor u into pi[v] only if u < current pi[v].
// Uses 32-bit CAS (vid_t = uint32_t).  Called when nd == d[v] (tie).
// FIX #4: atomicExch alone is not safe for ties — it does not honour the
// smallest-vid rule because any racing thread can overwrite.  This CAS loop
// ensures only the globally smallest u wins the tie, regardless of thread order.
__device__ inline void atomicMinPi(vid_t* pi_ptr, vid_t u) {
    unsigned* p = reinterpret_cast<unsigned*>(pi_ptr);
    unsigned  cur = *p, assumed;
    do {
        assumed = cur;
        if (u >= assumed) return;  // current pi is already smaller
        cur = atomicCAS(p, assumed, (unsigned)u);
    } while (assumed != cur);
}

// ── FP32 packed (d, pi) — in-flight emission via 64-bit CAS ──────────────────
// DPi32, pack_dpi32, unpack_dpi32, and atomicRelaxDPi32 live in gpu_atomics.hpp
// (shared with bellman_ford.hip).  See that header for the no-tie-break
// rationale and the RELAX_ATOMICS experiment branch.

__global__ void kernel_init_dpi32(
    DPi32* dpi, int* bucket_id,
    vid_t n_v, vid_t source, float inv)
{
    vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_v) return;
    dpi[v].d  = inv;
    dpi[v].pi = INVALID_VID;
    bucket_id[v] = INT_MAX;
    if (v == source) {
        dpi[v].d  = 0.0f;
        bucket_id[v] = 0;
    }
}

template<bool LIGHT>
__global__ void kernel_relax_dpi32(
    const vid_t* __restrict__ frontier, int sz,
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const float* __restrict__ weights,
    DPi32* dpi, int* bucket_id,
    float delta, int* any_updated)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sz) return;

    vid_t u  = frontier[tid];
    float du = dpi[u].d;

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        float w = weights[e];
        if ( LIGHT && w > delta) continue;
        if (!LIGHT && w <= delta) continue;

        vid_t v  = cols[e];
        float nd = du + w;
        int outcome = atomicRelaxDPi32(&dpi[v], nd, u);
        if (outcome == 1) {
            int new_bid = (int)(nd / delta);
            atomicMin(&bucket_id[v], new_bid);
            if (any_updated) atomicOr(any_updated, 1);
        }
        // outcome == 2: tiebreak only, d unchanged, no bucket update
    }
}

// ── Original kernel_relax (used for FP64 always; FP32 non-EMIT) ──────────────
// Relax edges from frontier. LIGHT=true: only w<=delta. LIGHT=false: only w>delta.
template<typename W, bool EMIT, bool LIGHT>
__global__ void kernel_relax(
    const vid_t* __restrict__ frontier, int sz,
    const eid_t* __restrict__ row_off,
    const vid_t* __restrict__ cols,
    const W*     __restrict__ weights,
    W* dist, vid_t* pi, int* bucket_id,
    W delta, W inv, int* any_updated)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sz) return;

    vid_t u  = frontier[tid];
    W    du  = dist[u];

    for (eid_t e = row_off[u]; e < row_off[u + 1]; ++e) {
        W w = weights[e];
        if ( LIGHT && w > delta) continue;
        if (!LIGHT && w <= delta) continue;

        vid_t v  = cols[e];
        W    nd  = du + w;
        W    old = atomicMinFP(&dist[v], nd);

        if (nd < old) {
            // Strict improvement: overwrite predecessor unconditionally.
            // (We just set d[v] = nd via atomicMinFP; installing u is correct.)
            if (EMIT) atomicExch(&pi[v], u);
            int new_bid = (int)(nd / delta);
            atomicMin(&bucket_id[v], new_bid);
            if (any_updated) atomicOr(any_updated, 1);
        } else if (EMIT && nd == old) {
            // Tie (nd == d[v]): apply tiebreak rule — smallest vid wins.
            // FIX #4: must use CAS loop, not atomicExch, to handle races.
            // Works for both FP32 and FP64 (pi array is always vid_t/uint32_t).
            atomicMinPi(&pi[v], u);
        }
    }
}

// Find minimum active bucket index; returns INT_MAX if all inactive.
__global__ void kernel_find_min_bucket(
    const int* bucket_id, vid_t n_v, int* result)
{
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int local_min = INT_MAX;
    for (vid_t v = gid; v < n_v; v += gridDim.x * blockDim.x)
        local_min = min(local_min, bucket_id[v]);

    smem[tid] = local_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = min(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMin(result, smem[0]);
}

// ── Host driver: FP32 packed (in-flight emission) ────────────────────────────
// Used when W=float and emit_cert=true. d and pi update atomically via 64-bit
// CAS, eliminating the dual-atomic race that requires post-hoc reconstruct_pi.
static Certificate<float> delta_stepping_gpu_packed_f32(
    const CSR<float>& g, vid_t source, float delta)
{
    const float INF   = Sentinel<float>::inf;
    const int   BLOCK = 256;
    const vid_t NV = g.n_vertices;
    const eid_t NE = g.n_edges;

    DPi32* d_dpi;     HIP_CHECK(hipMalloc(&d_dpi,     NV * sizeof(DPi32)));
    int*   d_bid;     HIP_CHECK(hipMalloc(&d_bid,     NV * sizeof(int)));
    eid_t* d_row;     HIP_CHECK(hipMalloc(&d_row,    (NV+1) * sizeof(eid_t)));
    vid_t* d_col;     HIP_CHECK(hipMalloc(&d_col,     NE * sizeof(vid_t)));
    float* d_wgt;     HIP_CHECK(hipMalloc(&d_wgt,     NE * sizeof(float)));
    vid_t* d_front;   HIP_CHECK(hipMalloc(&d_front,   NV * sizeof(vid_t)));
    // d_removed accumulates frontiers across Phase A iterations within a single
    // outer i_min cycle. A vertex can be re-added to bucket i_min multiple
    // times if light relaxations lower its d back into i_min's range, so the
    // total can exceed NV. Oversize to 16*NV (empirically safe on dense graphs at any tested Delta
    // like livejournal). For correctness we only need to PROCESS each removed
    // vertex's heavy edges; duplicates are wasted work but harmless.
    vid_t* d_removed; HIP_CHECK(hipMalloc(&d_removed, 16ull * NV * sizeof(vid_t)));
    int*   d_fsz;     HIP_CHECK(hipMalloc(&d_fsz,     sizeof(int)));
    int*   d_rsz;     HIP_CHECK(hipMalloc(&d_rsz,     sizeof(int)));
    int*   d_min_bid; HIP_CHECK(hipMalloc(&d_min_bid, sizeof(int)));
    int*   d_updated; HIP_CHECK(hipMalloc(&d_updated, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(), NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),     NE*sizeof(float), hipMemcpyHostToDevice));

    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_init_dpi32, grid_v, BLOCK, 0, 0,
                       d_dpi, d_bid, NV, source, INF);
    HIP_CHECK(hipDeviceSynchronize());

    while (true) {
        int h_max = INT_MAX;
        HIP_CHECK(hipMemcpy(d_min_bid, &h_max, sizeof(int), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(kernel_find_min_bucket,
            (grid_v + 1) / 2, BLOCK, BLOCK * sizeof(int), 0,
            d_bid, NV, d_min_bid);
        HIP_CHECK(hipDeviceSynchronize());

        int i_min;
        HIP_CHECK(hipMemcpy(&i_min, d_min_bid, sizeof(int), hipMemcpyDeviceToHost));
        if (i_min == INT_MAX) break;

        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_rsz, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        // Phase A: light edges
        while (true) {
            HIP_CHECK(hipMemcpy(d_fsz, &h_zero, sizeof(int), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(kernel_collect_frontier, grid_v, BLOCK, 0, 0,
                               d_bid, NV, i_min, d_front, d_fsz);
            HIP_CHECK(hipDeviceSynchronize());

            int fsz;
            HIP_CHECK(hipMemcpy(&fsz, d_fsz, sizeof(int), hipMemcpyDeviceToHost));
            if (fsz == 0) break;

            int rsz;
            HIP_CHECK(hipMemcpy(&rsz, d_rsz, sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(d_removed + rsz, d_front, fsz * sizeof(vid_t), hipMemcpyDeviceToDevice));
            rsz += fsz;
            HIP_CHECK(hipMemcpy(d_rsz, &rsz, sizeof(int), hipMemcpyHostToDevice));

            int grid_f = (fsz + BLOCK - 1) / BLOCK;
            hipLaunchKernelGGL(kernel_remove_from_bucket, grid_f, BLOCK, 0, 0,
                               d_bid, d_front, fsz);

            HIP_CHECK(hipMemcpy(d_updated, &h_zero, sizeof(int), hipMemcpyHostToDevice));
            hipLaunchKernelGGL((kernel_relax_dpi32<true>), grid_f, BLOCK, 0, 0,
                d_front, fsz, d_row, d_col, d_wgt,
                d_dpi, d_bid, delta, d_updated);
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Phase B: heavy edges
        int rsz;
        HIP_CHECK(hipMemcpy(&rsz, d_rsz, sizeof(int), hipMemcpyDeviceToHost));
        if (rsz > 0) {
            int grid_r = (rsz + BLOCK - 1) / BLOCK;
            hipLaunchKernelGGL((kernel_relax_dpi32<false>), grid_r, BLOCK, 0, 0,
                d_removed, rsz, d_row, d_col, d_wgt,
                d_dpi, d_bid, delta, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
        }
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

    hipFree(d_dpi);   hipFree(d_bid);
    hipFree(d_row);   hipFree(d_col);   hipFree(d_wgt);
    hipFree(d_front); hipFree(d_removed);
    hipFree(d_fsz); hipFree(d_rsz); hipFree(d_min_bid); hipFree(d_updated);
    return cert;
}

// ── Host driver: separate (existing) ─────────────────────────────────────────
// Used for FP64 always (96-bit packed atomic infeasible) and FP32 non-emit.
// For FP64 + emit_cert=true: post-hoc reconstruct_pi rebuilds pi from final d[].
template<typename W>
static Certificate<W> delta_stepping_gpu_separate(
    const CSR<W>& g, vid_t source, W delta, bool emit_cert)
{
    const W   INF   = Sentinel<W>::inf;
    const int BLOCK = 256;
    const vid_t NV  = g.n_vertices;
    const eid_t NE  = g.n_edges;

    // Device allocations
    W*     d_dist;      HIP_CHECK(hipMalloc(&d_dist,    NV * sizeof(W)));
    vid_t* d_pi = nullptr;
    if (emit_cert)     { HIP_CHECK(hipMalloc(&d_pi,     NV * sizeof(vid_t))); }
    int*   d_bid;       HIP_CHECK(hipMalloc(&d_bid,     NV * sizeof(int)));
    eid_t* d_row;       HIP_CHECK(hipMalloc(&d_row,    (NV+1) * sizeof(eid_t)));
    vid_t* d_col;       HIP_CHECK(hipMalloc(&d_col,     NE * sizeof(vid_t)));
    W*     d_wgt;       HIP_CHECK(hipMalloc(&d_wgt,     NE * sizeof(W)));
    vid_t* d_front;     HIP_CHECK(hipMalloc(&d_front,   NV * sizeof(vid_t)));
    vid_t* d_removed;   HIP_CHECK(hipMalloc(&d_removed, 16ull * NV * sizeof(vid_t)));
    int*   d_fsz;       HIP_CHECK(hipMalloc(&d_fsz,     sizeof(int)));
    int*   d_rsz;       HIP_CHECK(hipMalloc(&d_rsz,     sizeof(int)));
    int*   d_min_bid;   HIP_CHECK(hipMalloc(&d_min_bid, sizeof(int)));
    int*   d_updated;   HIP_CHECK(hipMalloc(&d_updated, sizeof(int)));

    // Upload graph
    HIP_CHECK(hipMemcpy(d_row, g.row_offsets.data(), (NV+1)*sizeof(eid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_indices.data(),     NE*sizeof(vid_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wgt, g.weights.data(),         NE*sizeof(W),     hipMemcpyHostToDevice));

    // Init
    int grid_v = (NV + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(kernel_init<W>, grid_v, BLOCK, 0, 0,
                       d_dist, d_pi, d_bid, NV, source, INF, emit_cert);
    HIP_CHECK(hipDeviceSynchronize());

    // Main loop
    while (true) {
        // Find i_min
        int h_max = INT_MAX;
        HIP_CHECK(hipMemcpy(d_min_bid, &h_max, sizeof(int), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(kernel_find_min_bucket,
            (grid_v + 1) / 2, BLOCK, BLOCK * sizeof(int), 0,
            d_bid, NV, d_min_bid);
        HIP_CHECK(hipDeviceSynchronize());

        int i_min;
        HIP_CHECK(hipMemcpy(&i_min, d_min_bid, sizeof(int), hipMemcpyDeviceToHost));
        if (i_min == INT_MAX) break;  // all done

        // Accumulate removed set across Phase A iterations
        int h_zero = 0;
        HIP_CHECK(hipMemcpy(d_rsz, &h_zero, sizeof(int), hipMemcpyHostToDevice));

        // Phase A: iterate light relaxations until bucket i_min is empty
        while (true) {
            HIP_CHECK(hipMemcpy(d_fsz, &h_zero, sizeof(int), hipMemcpyHostToDevice));
            hipLaunchKernelGGL(kernel_collect_frontier, grid_v, BLOCK, 0, 0,
                               d_bid, NV, i_min, d_front, d_fsz);
            HIP_CHECK(hipDeviceSynchronize());

            int fsz;
            HIP_CHECK(hipMemcpy(&fsz, d_fsz, sizeof(int), hipMemcpyDeviceToHost));
            if (fsz == 0) break;

            // Append frontier to removed set
            int rsz;
            HIP_CHECK(hipMemcpy(&rsz, d_rsz, sizeof(int), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(d_removed + rsz, d_front, fsz * sizeof(vid_t), hipMemcpyDeviceToDevice));
            rsz += fsz;
            HIP_CHECK(hipMemcpy(d_rsz, &rsz, sizeof(int), hipMemcpyHostToDevice));

            // Remove from bucket
            int grid_f = (fsz + BLOCK - 1) / BLOCK;
            hipLaunchKernelGGL(kernel_remove_from_bucket, grid_f, BLOCK, 0, 0,
                               d_bid, d_front, fsz);

            // Light relaxation
            HIP_CHECK(hipMemcpy(d_updated, &h_zero, sizeof(int), hipMemcpyHostToDevice));
            if (emit_cert)
                hipLaunchKernelGGL((kernel_relax<W, true,  true>), grid_f, BLOCK, 0, 0,
                    d_front, fsz, d_row, d_col, d_wgt,
                    d_dist, d_pi, d_bid, delta, INF, d_updated);
            else
                hipLaunchKernelGGL((kernel_relax<W, false, true>), grid_f, BLOCK, 0, 0,
                    d_front, fsz, d_row, d_col, d_wgt,
                    d_dist, nullptr, d_bid, delta, INF, d_updated);
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Phase B: heavy relaxations from removed set
        int rsz;
        HIP_CHECK(hipMemcpy(&rsz, d_rsz, sizeof(int), hipMemcpyDeviceToHost));
        if (rsz > 0) {
            int grid_r = (rsz + BLOCK - 1) / BLOCK;
            if (emit_cert)
                hipLaunchKernelGGL((kernel_relax<W, true,  false>), grid_r, BLOCK, 0, 0,
                    d_removed, rsz, d_row, d_col, d_wgt,
                    d_dist, d_pi, d_bid, delta, INF, nullptr);
            else
                hipLaunchKernelGGL((kernel_relax<W, false, false>), grid_r, BLOCK, 0, 0,
                    d_removed, rsz, d_row, d_col, d_wgt,
                    d_dist, nullptr, d_bid, delta, INF, nullptr);
            HIP_CHECK(hipDeviceSynchronize());
        }
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
    hipFree(d_bid);  hipFree(d_row); hipFree(d_col); hipFree(d_wgt);
    hipFree(d_front); hipFree(d_removed);
    hipFree(d_fsz); hipFree(d_rsz); hipFree(d_min_bid); hipFree(d_updated);

    return cert;
}

// ── Top-level dispatcher ─────────────────────────────────────────────────────
// FP32 + emit_cert=true → packed atomic (in-flight, race-free)
// All other cases     → separate arrays (FP64: post-hoc reconstruct_pi for emit)
template<typename W>
Certificate<W> delta_stepping_gpu(
    const CSR<W>& g, vid_t source, W delta, bool emit_cert)
{
    if constexpr (std::is_same_v<W, float>) {
        if (emit_cert)
            return delta_stepping_gpu_packed_f32(g, source, delta);
    }
    return delta_stepping_gpu_separate<W>(g, source, delta, emit_cert);
}

template Certificate<float>  delta_stepping_gpu<float> (const CSR<float>&,  vid_t, float,  bool);
template Certificate<double> delta_stepping_gpu<double>(const CSR<double>&, vid_t, double, bool);
