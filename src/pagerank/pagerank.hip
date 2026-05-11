// RE0 / RE1 PageRank — push-based, atomic-add scatter, FP32 or FP64.
//
// Designed for cross-vendor drift detection: per-iteration scatter uses
// non-deterministic atomicAdd over the chosen float type, so the reduction
// order at each destination vertex depends on warp/wavefront scheduling.
// This is exactly what RE0/RE1 measure — do NOT replace with a deterministic
// reduce.
//
// Source written in HIP API; on NVIDIA, hip_cuda_compat.h aliases the
// hip* surface to cuda* (mirrors the convention from Paper 2.1's
// src/sssp/*.hip — see asplos-27 sibling repo).
//
// Precision selected at runtime via --precision fp32 | fp64. Same kernel
// source compiles once; main() dispatches to run<float>() or run<double>().
// Both A10 (sm_86) and MI300X (gfx942) natively support atomicAdd<double>.

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

// CRC32 (zlib polynomial) over the raw byte buffer. Quick check for
// "are these two outputs literally byte-identical" before numerical compare.
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

// --- Kernels (templated on T = float or double) ---

// Scatter contributions: for each source u, push damping * pr_in[u] / outdeg(u)
// to every neighbor v via atomicAdd. Dangling vertices (outdeg=0) contribute
// nothing here; their mass is collected separately on the host.
template<typename T>
__global__ void pr_push_scatter(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const T*       __restrict__ pr_in,
    T*             __restrict__ pr_out,
    int N, T damping)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    int beg = row_ptr[u];
    int end = row_ptr[u + 1];
    int deg = end - beg;
    if (deg == 0) return;
    T share = damping * pr_in[u] / (T)deg;
    for (int e = beg; e < end; ++e) {
        atomicAdd(&pr_out[col_idx[e]], share);
    }
}

// Initialise pr_out to base = (1 - damping)/N + damping * dangling_mass / N.
template<typename T>
__global__ void pr_init_iter(T* pr_out, int N, T base) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N) pr_out[u] = base;
}

// Sum dangling-vertex pr values (outdeg == 0). Block-level reduction with
// atomicAdd to a single global accumulator. Shmem is allocated as a byte
// buffer + reinterpreted to T to keep extern __shared__ unambiguous across
// template instantiations (nvcc/amdclang both accept this idiom).
template<typename T>
__global__ void pr_dangling_sum(
    const int32_t* __restrict__ row_ptr,
    const T*       __restrict__ pr_in,
    int N, T* out_sum)
{
    extern __shared__ unsigned char raw_shmem_dangling[];
    T* sdata = reinterpret_cast<T*>(raw_shmem_dangling);
    int tid = threadIdx.x;
    T acc = T(0);
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
template<typename T>
__global__ void pr_l1_diff(const T* a, const T* b, int N, T* out) {
    extern __shared__ unsigned char raw_shmem_l1[];
    T* sdata = reinterpret_cast<T*>(raw_shmem_l1);
    int tid = threadIdx.x;
    T acc = T(0);
    for (int u = blockIdx.x * blockDim.x + tid; u < N; u += blockDim.x * gridDim.x) {
        T d = a[u] - b[u];
        acc += d < T(0) ? -d : d;
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
    std::string precision = "fp32";   // "fp32" | "fp64"
    double damping = 0.85;
    int    max_iter = 100;
    double tol = 1e-6;
    int    block = 256;
};

static void usage() {
    std::fprintf(stderr,
        "pagerank --graph <csr.bin> --out <prefix> --dataset <label>\n"
        "         [--precision fp32|fp64] [--damping 0.85] [--max-iter 100]\n"
        "         [--tol 1e-6] [--block 256]\n");
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if      (k == "--graph")     a.graph_path = next();
        else if (k == "--out")       a.out_prefix = next();
        else if (k == "--dataset")   a.dataset_label = next();
        else if (k == "--precision") a.precision = next();
        else if (k == "--damping")   a.damping = std::stod(next());
        else if (k == "--max-iter")  a.max_iter = std::stoi(next());
        else if (k == "--tol")       a.tol = std::stod(next());
        else if (k == "--block")     a.block = std::stoi(next());
        else { usage(); std::exit(2); }
    }
    if (a.graph_path.empty() || a.out_prefix.empty() || a.dataset_label.empty()) {
        usage(); std::exit(2);
    }
    if (a.precision != "fp32" && a.precision != "fp64") {
        std::fprintf(stderr, "--precision must be fp32 or fp64, got '%s'\n", a.precision.c_str());
        std::exit(2);
    }
    return a;
}

template<typename T>
static int run(const Args& args, const hipDeviceProp_t& prop) {
    CsrGraph g = load_csr(args.graph_path);
    int N = (int)g.N;
    int E = (int)g.E;

    int32_t *d_row, *d_col;
    T *d_pr_a, *d_pr_b, *d_dangling, *d_l1;
    HIP_CHECK(hipMalloc(&d_row, sizeof(int32_t) * (N + 1)));
    HIP_CHECK(hipMalloc(&d_col, sizeof(int32_t) * E));
    HIP_CHECK(hipMalloc(&d_pr_a, sizeof(T) * N));
    HIP_CHECK(hipMalloc(&d_pr_b, sizeof(T) * N));
    HIP_CHECK(hipMalloc(&d_dangling, sizeof(T)));
    HIP_CHECK(hipMalloc(&d_l1, sizeof(T)));

    HIP_CHECK(hipMemcpy(d_row, g.row_ptr.data(), sizeof(int32_t) * (N + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, g.col_idx.data(), sizeof(int32_t) * E,       hipMemcpyHostToDevice));

    std::vector<T> init(N, T(1) / T(N));
    HIP_CHECK(hipMemcpy(d_pr_a, init.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    int grid = (N + args.block - 1) / args.block;
    int reduce_grid = std::min(grid, 1024);
    size_t reduce_shmem = args.block * sizeof(T);

    T damping_T = T(args.damping);
    T tol_T     = T(args.tol);

    auto t0 = std::chrono::steady_clock::now();
    int iters_run = 0;
    T final_l1 = T(0);
    T* d_in  = d_pr_a;
    T* d_out = d_pr_b;
    for (int it = 0; it < args.max_iter; ++it) {
        // 1. Dangling mass
        HIP_CHECK(hipMemset(d_dangling, 0, sizeof(T)));
        hipLaunchKernelGGL((pr_dangling_sum<T>), dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_row, d_in, N, d_dangling);
        HIP_CHECK(hipGetLastError());
        T h_dangling = T(0);
        HIP_CHECK(hipMemcpy(&h_dangling, d_dangling, sizeof(T), hipMemcpyDeviceToHost));

        // 2. Init pr_out to base = (1 - d)/N + d * dangling_mass / N
        T base = (T(1) - damping_T) / T(N) + damping_T * h_dangling / T(N);
        hipLaunchKernelGGL((pr_init_iter<T>), dim3(grid), dim3(args.block), 0, 0,
                           d_out, N, base);
        HIP_CHECK(hipGetLastError());

        // 3. Scatter contributions from non-dangling vertices
        hipLaunchKernelGGL((pr_push_scatter<T>), dim3(grid), dim3(args.block), 0, 0,
                           d_row, d_col, d_in, d_out, N, damping_T);
        HIP_CHECK(hipGetLastError());

        // 4. Convergence check (L1 diff)
        HIP_CHECK(hipMemset(d_l1, 0, sizeof(T)));
        hipLaunchKernelGGL((pr_l1_diff<T>), dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_in, d_out, N, d_l1);
        HIP_CHECK(hipGetLastError());
        T h_l1 = T(0);
        HIP_CHECK(hipMemcpy(&h_l1, d_l1, sizeof(T), hipMemcpyDeviceToHost));
        final_l1 = h_l1;
        iters_run = it + 1;

        std::swap(d_in, d_out);
        if (h_l1 < tol_T) break;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // d_in holds latest values (post-swap).
    std::vector<T> pr(N);
    HIP_CHECK(hipMemcpy(pr.data(), d_in, sizeof(T) * N, hipMemcpyDeviceToHost));

    uint32_t crc = crc32_zlib(pr.data(), sizeof(T) * N);

    std::string bin_path = args.out_prefix + ".bin";
    std::FILE* fb = std::fopen(bin_path.c_str(), "wb");
    if (!fb) { std::fprintf(stderr, "cannot write %s\n", bin_path.c_str()); return 1; }
    std::fwrite(pr.data(), sizeof(T), N, fb);
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
        "  \"damping\": %.10f,\n"
        "  \"max_iter\": %d,\n"
        "  \"tol\": %.10e,\n"
        "  \"iters_run\": %d,\n"
        "  \"final_l1\": %.10e,\n"
        "  \"wall_ms\": %.3f,\n"
        "  \"output_bin\": \"%s\",\n"
        "  \"output_crc32\": \"%08x\",\n"
        "  \"precision\": \"%s\"\n"
        "}\n",
        kVendorStr, prop.name, args.dataset_label.c_str(), args.graph_path.c_str(),
        N, E, args.damping, args.max_iter, args.tol,
        iters_run, (double)final_l1, wall_ms, bin_path.c_str(), crc,
        args.precision.c_str());
    std::fclose(fj);

    std::fprintf(stderr,
        "[pagerank] vendor=%s device=%s precision=%s N=%d E=%d iters=%d l1=%.3e wall=%.1fms crc=%08x\n",
        kVendorStr, prop.name, args.precision.c_str(),
        N, E, iters_run, (double)final_l1, wall_ms, crc);

    HIP_CHECK(hipFree(d_row));
    HIP_CHECK(hipFree(d_col));
    HIP_CHECK(hipFree(d_pr_a));
    HIP_CHECK(hipFree(d_pr_b));
    HIP_CHECK(hipFree(d_dangling));
    HIP_CHECK(hipFree(d_l1));
    return 0;
}

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    if (args.precision == "fp64") return run<double>(args, prop);
    return run<float>(args, prop);
}
