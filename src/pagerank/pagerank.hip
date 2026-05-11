// RE0 PageRank — push-based, FP32, atomic-add scatter.
//
// Designed for cross-vendor drift detection: per-iteration scatter uses
// non-deterministic atomicAdd over float, so the reduction order at each
// destination vertex depends on warp scheduling. This is exactly what RE0
// is trying to measure — do NOT replace with a deterministic reduce.
//
// Source written in HIP API; on NVIDIA, hip_cuda_compat.h aliases the
// hip* surface to cuda* (mirrors the convention from Paper 2.1's
// src/sssp/*.hip — see asplos-27 sibling repo).

#ifdef __CUDACC__
    #include "hip_cuda_compat.h"
#else
    #include <hip/hip_runtime.h>
#endif

#include "core/csr.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(call) do {                                                  \
    hipError_t _err = (call);                                                 \
    if (_err != hipSuccess) {                                                 \
        std::fprintf(stderr, "HIP error %s:%d: %s\n",                         \
                     __FILE__, __LINE__, hipGetErrorString(_err));            \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

#if defined(__CUDACC__)
    static constexpr const char* kVendorStr = "nvidia";
#else
    static constexpr const char* kVendorStr = "amd";
#endif

// CRC32 (zlib polynomial) over the raw float32 byte buffer. Used as a
// quick "are these two outputs literally byte-identical" check before
// going to numerical comparison.
static uint32_t crc32_zlib(const void* data, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// --- Kernels ---

// Scatter contributions: for each source u, push damping * pr_in[u] / outdeg(u)
// to every neighbor v via atomicAdd. Dangling vertices (outdeg=0) contribute
// nothing here; their mass is collected separately on the host.
__global__ void pr_push_scatter(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const float*   __restrict__ pr_in,
    float*         __restrict__ pr_out,
    int N, float damping)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    int beg = row_ptr[u];
    int end = row_ptr[u + 1];
    int deg = end - beg;
    if (deg == 0) return;
    float share = damping * pr_in[u] / (float)deg;
    for (int e = beg; e < end; ++e) {
        atomicAdd(&pr_out[col_idx[e]], share);
    }
}

// Initialise pr_out to base = (1 - damping)/N + damping * dangling_mass / N.
// Run before scatter each iteration.
__global__ void pr_init_iter(float* pr_out, int N, float base) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N) pr_out[u] = base;
}

// Sum dangling-vertex pr values (outdeg == 0). Block-level reduction with
// atomicAdd to a single global accumulator. Called once per iteration so
// the cross-block atomic isn't a bottleneck.
__global__ void pr_dangling_sum(
    const int32_t* __restrict__ row_ptr,
    const float*   __restrict__ pr_in,
    int N, float* out_sum)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int u = blockIdx.x * blockDim.x + tid; u < N; u += blockDim.x * gridDim.x) {
        if (row_ptr[u + 1] - row_ptr[u] == 0) acc += pr_in[u];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_sum, sdata[0]);
}

// L1 difference between successive PR vectors, for convergence check.
__global__ void pr_l1_diff(const float* a, const float* b, int N, float* out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int u = blockIdx.x * blockDim.x + tid; u < N; u += blockDim.x * gridDim.x) {
        acc += std::fabs(a[u] - b[u]);
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// --- Driver ---

struct Args {
    std::string graph_path;
    std::string out_prefix;
    std::string dataset_label;
    float damping = 0.85f;
    int   max_iter = 100;
    float tol = 1e-6f;
    int   block = 256;
};

static void usage() {
    std::fprintf(stderr,
        "pagerank --graph <csr.bin> --out <prefix> --dataset <label>\n"
        "         [--damping 0.85] [--max-iter 100] [--tol 1e-6] [--block 256]\n");
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--graph")    a.graph_path = next();
        else if (k == "--out")      a.out_prefix = next();
        else if (k == "--dataset")  a.dataset_label = next();
        else if (k == "--damping")  a.damping = std::stof(next());
        else if (k == "--max-iter") a.max_iter = std::stoi(next());
        else if (k == "--tol")      a.tol = std::stof(next());
        else if (k == "--block")    a.block = std::stoi(next());
        else { usage(); std::exit(2); }
    }
    if (a.graph_path.empty() || a.out_prefix.empty() || a.dataset_label.empty()) {
        usage(); std::exit(2);
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    CsrGraph g = load_csr(args.graph_path);
    int N = (int)g.N;
    int E = (int)g.E;

    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    int32_t *d_row, *d_col;
    float   *d_pr_a, *d_pr_b, *d_dangling, *d_l1;
    HIP_CHECK(hipMalloc(&d_row, sizeof(int32_t) * (N + 1)));
    HIP_CHECK(hipMalloc(&d_col, sizeof(int32_t) * E));
    HIP_CHECK(hipMalloc(&d_pr_a, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_pr_b, sizeof(float) * N));
    HIP_CHECK(hipMalloc(&d_dangling, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_l1, sizeof(float)));

    HIP_CHECK(hipMemcpy(d_row, g.row_ptr.data(), sizeof(int32_t) * (N + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_idx.data(), sizeof(int32_t) * E,       hipMemcpyHostToDevice));

    std::vector<float> init(N, 1.0f / (float)N);
    HIP_CHECK(hipMemcpy(d_pr_a, init.data(), sizeof(float) * N, hipMemcpyHostToDevice));

    int grid = (N + args.block - 1) / args.block;
    int reduce_grid = std::min(grid, 1024);
    int reduce_shmem = args.block * sizeof(float);

    auto t0 = std::chrono::steady_clock::now();
    int iters_run = 0;
    float final_l1 = 0.0f;
    float* d_in  = d_pr_a;
    float* d_out = d_pr_b;
    for (int it = 0; it < args.max_iter; ++it) {
        // 1. Dangling mass
        HIP_CHECK(hipMemset(d_dangling, 0, sizeof(float)));
        hipLaunchKernelGGL(pr_dangling_sum, dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_row, d_in, N, d_dangling);
        HIP_CHECK(hipGetLastError());
        float h_dangling = 0.0f;
        HIP_CHECK(hipMemcpy(&h_dangling, d_dangling, sizeof(float), hipMemcpyDeviceToHost));

        // 2. Init pr_out to base = (1 - d)/N + d * dangling_mass / N
        float base = (1.0f - args.damping) / (float)N + args.damping * h_dangling / (float)N;
        hipLaunchKernelGGL(pr_init_iter, dim3(grid), dim3(args.block), 0, 0,
                           d_out, N, base);
        HIP_CHECK(hipGetLastError());

        // 3. Scatter contributions from non-dangling vertices
        hipLaunchKernelGGL(pr_push_scatter, dim3(grid), dim3(args.block), 0, 0,
                           d_row, d_col, d_in, d_out, N, args.damping);
        HIP_CHECK(hipGetLastError());

        // 4. Convergence check (L1 diff)
        HIP_CHECK(hipMemset(d_l1, 0, sizeof(float)));
        hipLaunchKernelGGL(pr_l1_diff, dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_in, d_out, N, d_l1);
        HIP_CHECK(hipGetLastError());
        float h_l1 = 0.0f;
        HIP_CHECK(hipMemcpy(&h_l1, d_l1, sizeof(float), hipMemcpyDeviceToHost));
        final_l1 = h_l1;
        iters_run = it + 1;

        std::swap(d_in, d_out);
        if (h_l1 < args.tol) break;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // d_in now holds the latest values (post-swap).
    std::vector<float> pr(N);
    HIP_CHECK(hipMemcpy(pr.data(), d_in, sizeof(float) * N, hipMemcpyDeviceToHost));

    uint32_t crc = crc32_zlib(pr.data(), sizeof(float) * N);

    std::string bin_path = args.out_prefix + ".bin";
    std::FILE* fb = std::fopen(bin_path.c_str(), "wb");
    if (!fb) { std::fprintf(stderr, "cannot write %s\n", bin_path.c_str()); return 1; }
    std::fwrite(pr.data(), sizeof(float), N, fb);
    std::fclose(fb);

    std::string json_path = args.out_prefix + ".json";
    std::FILE* fj = std::fopen(json_path.c_str(), "w");
    if (!fj) { std::fprintf(stderr, "cannot write %s\n", json_path.c_str()); return 1; }
    std::fprintf(fj,
        "{\n"
        "  \"vendor\": \"%s\",\n"
        "  \"device\": \"%s\",\n"
        "  \"dataset\": \"%s\",\n"
        "  \"graph_path\": \"%s\",\n"
        "  \"N\": %d,\n"
        "  \"E\": %d,\n"
        "  \"damping\": %.8f,\n"
        "  \"max_iter\": %d,\n"
        "  \"tol\": %.8e,\n"
        "  \"iters_run\": %d,\n"
        "  \"final_l1\": %.8e,\n"
        "  \"wall_ms\": %.3f,\n"
        "  \"output_bin\": \"%s\",\n"
        "  \"output_crc32\": \"%08x\",\n"
        "  \"precision\": \"fp32\"\n"
        "}\n",
        kVendorStr, prop.name, args.dataset_label.c_str(), args.graph_path.c_str(),
        N, E, args.damping, args.max_iter, args.tol,
        iters_run, final_l1, wall_ms, bin_path.c_str(), crc);
    std::fclose(fj);

    std::fprintf(stderr,
        "[pagerank] vendor=%s device=%s N=%d E=%d iters=%d l1=%.3e wall=%.1fms crc=%08x\n",
        kVendorStr, prop.name, N, E, iters_run, final_l1, wall_ms, crc);

    HIP_CHECK(hipFree(d_row));
    HIP_CHECK(hipFree(d_col));
    HIP_CHECK(hipFree(d_pr_a));
    HIP_CHECK(hipFree(d_pr_b));
    HIP_CHECK(hipFree(d_dangling));
    HIP_CHECK(hipFree(d_l1));
    return 0;
}
