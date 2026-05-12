// RE2e_v2 PageRank — pull-based, ZERO atomicAdd, fully deterministic.
//
// Identical to pagerank_pull.hip except:
//   pr_dangling_sum  → per-block partial sums to buffer, host sequential reduce
//   pr_l1_diff       → per-block partial sums to buffer, host sequential reduce
//
// If this variant is byte-identical across seeds on ALL datasets,
// atomicAdd scheduling is confirmed as the sole drift source.

#ifdef __CUDACC__
    #include "core/hip_cuda_compat.h"
#else
    #include <hip/hip_runtime.h>
#endif

#include "core/csr_pagerank.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

#include "common/crc32.h"

// --- CSR to CSC transpose (CPU, one-time at startup) ---

struct CscGraph {
    int32_t N, E;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

static CscGraph csr_to_csc(const CsrGraph& g) {
    int32_t N = (int32_t)g.N;
    int32_t E = (int32_t)g.E;
    CscGraph csc;
    csc.N = N;
    csc.E = E;
    csc.col_ptr.assign(N + 1, 0);
    csc.row_idx.resize(E);

    for (int32_t e = 0; e < E; ++e)
        csc.col_ptr[g.col_idx[e] + 1]++;
    for (int32_t j = 1; j <= N; ++j)
        csc.col_ptr[j] += csc.col_ptr[j - 1];

    std::vector<int32_t> pos(csc.col_ptr.begin(), csc.col_ptr.end());
    for (int32_t u = 0; u < N; ++u) {
        for (int32_t e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
            int32_t v = g.col_idx[e];
            csc.row_idx[pos[v]++] = u;
        }
    }
    return csc;
}

// --- Kernels (pull-based, ZERO atomicAdd) ---

// Pull gather: identical to pagerank_pull.hip — already atomic-free.
template<typename T>
__global__ void pr_pull_gather(
    const int32_t* __restrict__ csc_col_ptr,
    const int32_t* __restrict__ csc_row_idx,
    const int32_t* __restrict__ outdeg,
    const T*       __restrict__ pr_in,
    T*             __restrict__ pr_out,
    int N, T damping, T base)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    int beg = csc_col_ptr[u];
    int end = csc_col_ptr[u + 1];
    T acc = base;
    for (int e = beg; e < end; ++e) {
        int v = csc_row_idx[e];
        int deg_v = outdeg[v];
        if (deg_v > 0)
            acc += damping * pr_in[v] / (T)deg_v;
    }
    pr_out[u] = acc;
}

// Dangling sum: per-block partial sums → block_out[blockIdx.x]. NO atomicAdd.
template<typename T>
__global__ void pr_dangling_sum_v2(
    const int32_t* __restrict__ outdeg,
    const T*       __restrict__ pr_in,
    int N, T* block_out)
{
    extern __shared__ unsigned char raw_shmem[];
    T* sdata = reinterpret_cast<T*>(raw_shmem);
    int tid = threadIdx.x;
    T acc = T(0);
    for (int u = blockIdx.x * blockDim.x + tid; u < N; u += blockDim.x * gridDim.x) {
        if (outdeg[u] == 0) acc += pr_in[u];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_out[blockIdx.x] = sdata[0];
}

// L1 diff: per-block partial sums → block_out[blockIdx.x]. NO atomicAdd.
template<typename T>
__global__ void pr_l1_diff_v2(const T* a, const T* b, int N, T* block_out) {
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
    if (tid == 0) block_out[blockIdx.x] = sdata[0];
}

// Host-side sequential sum of per-block partial sums (deterministic).
template<typename T>
static T host_reduce(const T* buf, int n) {
    T sum = T(0);
    for (int i = 0; i < n; ++i) sum += buf[i];
    return sum;
}

// --- Driver ---

struct Args {
    std::string graph_path;
    std::string out_prefix;
    std::string dataset_label;
    std::string precision = "fp32";
    double damping = 0.85;
    int    max_iter = 100;
    double tol = 1e-6;
    int    block = 256;
};

static void usage() {
    std::fprintf(stderr,
        "pagerank_pull_v2 --graph <csr.bin> --out <prefix> --dataset <label>\n"
        "                 [--precision fp32|fp64] [--damping 0.85] [--max-iter 100]\n"
        "                 [--tol 1e-6] [--block 256]\n");
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

    std::fprintf(stderr, "[pagerank_pull_v2] transposing CSR -> CSC (N=%d E=%d)...\n", N, E);
    CscGraph csc = csr_to_csc(g);

    std::vector<int32_t> outdeg(N);
    for (int u = 0; u < N; ++u)
        outdeg[u] = g.row_ptr[u + 1] - g.row_ptr[u];

    int grid = (N + args.block - 1) / args.block;
    int reduce_grid = std::min(grid, 1024);
    size_t reduce_shmem = args.block * sizeof(T);

    // Device allocations
    int32_t *d_csc_col, *d_csc_row, *d_outdeg;
    T *d_pr_a, *d_pr_b, *d_block_buf;
    HIP_CHECK(hipMalloc(&d_csc_col, sizeof(int32_t) * (N + 1)));
    HIP_CHECK(hipMalloc(&d_csc_row, sizeof(int32_t) * E));
    HIP_CHECK(hipMalloc(&d_outdeg,  sizeof(int32_t) * N));
    HIP_CHECK(hipMalloc(&d_pr_a, sizeof(T) * N));
    HIP_CHECK(hipMalloc(&d_pr_b, sizeof(T) * N));
    HIP_CHECK(hipMalloc(&d_block_buf, sizeof(T) * reduce_grid));

    HIP_CHECK(hipMemcpy(d_csc_col, csc.col_ptr.data(), sizeof(int32_t) * (N + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_csc_row, csc.row_idx.data(), sizeof(int32_t) * E,       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_outdeg,  outdeg.data(),       sizeof(int32_t) * N,       hipMemcpyHostToDevice));

    std::vector<T> init(N, T(1) / T(N));
    HIP_CHECK(hipMemcpy(d_pr_a, init.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    // Host buffer for per-block partial sums
    std::vector<T> h_block_buf(reduce_grid);

    T damping_T = T(args.damping);
    T tol_T     = T(args.tol);

    auto t0 = std::chrono::steady_clock::now();
    int iters_run = 0;
    T final_l1 = T(0);
    T* d_in  = d_pr_a;
    T* d_out = d_pr_b;
    for (int it = 0; it < args.max_iter; ++it) {
        // 1. Dangling mass — per-block partial sums, then host reduce
        hipLaunchKernelGGL((pr_dangling_sum_v2<T>), dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_outdeg, d_in, N, d_block_buf);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(h_block_buf.data(), d_block_buf, sizeof(T) * reduce_grid, hipMemcpyDeviceToHost));
        T h_dangling = host_reduce(h_block_buf.data(), reduce_grid);

        // 2. Base = (1 - d)/N + d * dangling_mass / N
        T base = (T(1) - damping_T) / T(N) + damping_T * h_dangling / T(N);

        // 3. Pull-gather
        hipLaunchKernelGGL((pr_pull_gather<T>), dim3(grid), dim3(args.block), 0, 0,
                           d_csc_col, d_csc_row, d_outdeg, d_in, d_out, N, damping_T, base);
        HIP_CHECK(hipGetLastError());

        // 4. L1 diff — per-block partial sums, then host reduce
        hipLaunchKernelGGL((pr_l1_diff_v2<T>), dim3(reduce_grid), dim3(args.block),
                           reduce_shmem, 0, d_in, d_out, N, d_block_buf);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(h_block_buf.data(), d_block_buf, sizeof(T) * reduce_grid, hipMemcpyDeviceToHost));
        T h_l1 = host_reduce(h_block_buf.data(), reduce_grid);

        final_l1 = h_l1;
        iters_run = it + 1;

        std::swap(d_in, d_out);
        if (h_l1 < tol_T) break;
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
    bool converged = (double)final_l1 < args.tol;
    std::fprintf(fj,
        "{\n"
        "  \"algorithm\": \"pagerank_pull_v2\",\n"
        "  \"dataset\": \"%s\",\n"
        "  \"gpu\": \"%s\",\n"
        "  \"precision\": \"%s\",\n"
        "  \"wall_ms\": %.3f,\n"
        "  \"output_crc32\": \"%08x\",\n"
        "  \"converged\": %s,\n"
        "  \"iterations\": %d,\n"
        "  \"N\": %d,\n"
        "  \"E\": %d,\n"
        "  \"extra\": {\n"
        "    \"vendor\": \"%s\",\n"
        "    \"damping\": %.10f,\n"
        "    \"final_l1\": %.10e,\n"
        "    \"tol\": %.10e,\n"
        "    \"max_iter\": %d,\n"
        "    \"output_bin\": \"%s\"\n"
        "  }\n"
        "}\n",
        args.dataset_label.c_str(), prop.name, args.precision.c_str(),
        wall_ms, crc, converged ? "true" : "false", iters_run,
        N, E, kVendorStr, args.damping, (double)final_l1, args.tol,
        args.max_iter, bin_path.c_str());
    std::fclose(fj);

    std::fprintf(stderr,
        "[pagerank_pull_v2] vendor=%s device=%s precision=%s N=%d E=%d iters=%d l1=%.3e wall=%.1fms crc=%08x\n",
        kVendorStr, prop.name, args.precision.c_str(),
        N, E, iters_run, (double)final_l1, wall_ms, crc);

    HIP_CHECK(hipFree(d_csc_col));
    HIP_CHECK(hipFree(d_csc_row));
    HIP_CHECK(hipFree(d_outdeg));
    HIP_CHECK(hipFree(d_pr_a));
    HIP_CHECK(hipFree(d_pr_b));
    HIP_CHECK(hipFree(d_block_buf));
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
