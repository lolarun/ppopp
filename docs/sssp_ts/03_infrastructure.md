# 03 — Infrastructure (Build / Datasets / Harness / 实验工具)

**Document purpose:** 算法和 verifier 之外的所有支持代码 — CMake、dataset loader、多精度 wiring、run harness、error injection、drift analysis、profiling、数据管道。W1 + W8 集中开发,服务整个实验阶段。

**对应论文章节:** §IV.D (heterogeneous deployment)、§V (methodology)、§VI-VIII (实验)

---

## 1. Build system (W1, W8 收尾)

### 1.1 顶层 CMake

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.21)
project(gpu_sssp_certifying LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU backend selection: -DGPU_BACKEND=CUDA or =ROCM
set(GPU_BACKEND "AUTO" CACHE STRING "CUDA / ROCM / AUTO")

include(cmake/Dependencies.cmake)
include(cmake/DetectGPU.cmake)        # 决定 enable_language(HIP) 还是 (CUDA)

add_subdirectory(src/core)
add_subdirectory(src/io)
add_subdirectory(src/sssp)
add_subdirectory(src/verifier)
add_subdirectory(src/harness)

enable_testing()
add_subdirectory(tests)
```

### 1.2 GPU 检测 + HIP unified

`cmake/DetectGPU.cmake` 逻辑:
- 如果 `$ENV{ROCM_PATH}` 存在 → AMD 路径,`enable_language(HIP)`,target 是 ROCm
- 否则 nvidia-smi 探测成功 → NVIDIA 路径,`enable_language(HIP)` + `set(HIP_PLATFORM nvidia)` (HIP 转 CUDA backend)
- 否则 build CPU-only (verifier + reference 仍可 build)

**关键: 不论 NVIDIA 还是 AMD,都用 `*.hip` 文件 + `enable_language(HIP)`.** 不写两份 toolchain. 由 hipcc 或 hipcc-cuda backend 选择性编译。

### 1.3 依赖

`cmake/Dependencies.cmake`:

```cmake
find_package(OpenMP 4.5 REQUIRED)
find_package(Boost 1.83 EXACT REQUIRED COMPONENTS graph)
find_package(nlohmann_json 3.11 EXACT REQUIRED)
find_package(fmt 10 REQUIRED)
find_package(Catch2 3 REQUIRED)
```

EXACT 强制版本一致. 升级触发 build 失败 (W2 lock 后不升级)。

### 1.4 Build 命令

```bash
# Configure (NVIDIA)
cmake -S . -B build_cuda -DGPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release

# Configure (AMD)
cmake -S . -B build_rocm -DGPU_BACKEND=ROCM -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build_cuda -j
cmake --build build_rocm -j

# 同一份代码,两次 build,两个 binary
ls build_cuda/src/harness/run_sssp
ls build_rocm/src/harness/run_sssp
```

CI: GitHub Actions 跑 CPU-only build + tests; GPU 测试每周一次手工 run.

---

## 2. Dataset infrastructure (W2, W8)

### 2.1 支持的格式

| 格式 | 用途 | Loader |
|---|---|---|
| 内部 binary CSR (`.csr`) | 主格式,所有实验用 | `csr_io.h` |
| DIMACS (`.gr`) | road network challenge | `dimacs_loader.cpp` |
| GAP / Graph500 (`.mtx` / Galois 格式) | 大型 benchmark | `gap_loader.cpp` |
| SNAP (edge list `.txt`) | 社交网络 | `snap_loader.cpp` |
| RMAT (synthetic) | scaling study | `rmat_generator.cpp` |

**约定:** loader 统一返回 `CSR<float>`. 多精度只在算法内部 cast,数据集 always FP32 source. (FP16 / FP64 实验通过 CLI flag 转换。)

### 2.2 Dataset 选择 (lock W2)

最低 8 张图 (Tier B 实验需要):

| Name | Source | n_v | n_e | Type |
|---|---|---|---|---|
| usa_road | DIMACS | 24M | 58M | road |
| ny_road | DIMACS | 264K | 733K | road |
| livejournal | SNAP | 4.8M | 69M | social |
| twitter | GAP | 41M | 1.4B | social |
| friendster | SNAP | 65M | 1.8B | social |
| kron-25 | GAP | 33M | 1.05B | RMAT |
| road-USA | GAP | 23M | 58M | road |
| web-google | SNAP | 875K | 5.1M | web |

如 Tier C 启用 (E8 scaling):再加 RMAT-22/23/24/26 4 张.

**预先下载 + 预处理:** `scripts/fetch_datasets.sh` 下载源数据,`scripts/preprocess.sh` 转换成 `.csr` 二进制并 cache 在 `data/cache/`. 实验时直接读 cache,避免 loader 时间污染 measurement。

### 2.3 边权处理

DIMACS road network 自带 weight (距离). 其他数据集多为无权,需要赋 weight:

| 数据集类 | Weight 策略 |
|---|---|
| Road (DIMACS / GAP road-USA) | 用原 weight (已为 FP) |
| Social / web (LiveJournal, Twitter, Friendster, web-google) | **uniform random FP `[0.001, 1.0]`,seed=42** |
| Synthetic RMAT | **uniform random FP `[0.001, 1.0]`,seed=42** |

**FIX #8 — 必须是 FP 权重,不能是整数:** uniform integer `[1, 1024]` 在 FP32/FP64 下可以精确表示,跨平台路径加法产生相同的 bit-exact 结果 → drift 为零 → 论文的实验核心 (cross-platform distance drift 存在且可量化) 彻底失效。

使用 FP `[0.001, 1.0]` 则:
- 加法顺序变化 → FP non-associativity → 两个平台产生不同的 d[] bit pattern
- 差异量级 ~1e-6 (机器 eps × 路径长度),足以驱动 E1 drift 实验

**实现:** `scripts/snap_to_dimacs.py` 和 `src/io/{snap_loader,rmat_generator,gap_loader}.cpp` 在赋合成权重时均使用 `uniform_real_distribution<double>(0.001, 1.0)`。road network 直接用 DIMACS 原始浮点距离,无需处理。

E9 stress test 改变 weight distribution (Gaussian / power-law / adversarial) 时,baseline 是 uniform FP `[0.001, 1.0]`。

---

## 3. Run harness (W8)

### 3.1 主 entry

```cpp
// src/harness/main.cpp
int main(int argc, char** argv) {
    auto config = parse_cli(argc, argv);
    auto graph  = load_dataset(config.dataset);
    set_random_seed(config.seed);

    LogWriter log(config.output_jsonl);

    for (int rep = 0; rep < config.reps; ++rep) {
        auto start = now();
        Certificate<float> cert;
        if (config.algo == "delta_stepping_gpu") {
            cert = delta_stepping_gpu(graph, config.source,
                                      config.delta, config.emit_cert);
        }
        // ... other algos
        auto sssp_time = since(start);

        VerifyResult vr;
        if (config.verify) {
            vr = verify(graph, config.source, cert.d, cert.pi);
        }

        log.write_run(config, rep, sssp_time, vr, cert);
    }
}
```

### 3.2 CLI

```
run_sssp \
  --dataset=<path>             # .csr 二进制 path
  --source=<vid>               # default 0
  --algo=<dijkstra_cpu|delta_stepping_cpu|delta_stepping_gpu>
  --gpu=<auto|0|1|...>
  --precision=<fp16|fp32|fp64> # default fp32
  --emit_cert=<true|false>     # 决定 baseline vs augmented
  --delta=<float>              # default heuristic
  --verify=<true|false>        # default true
  --seed=<int>                 # default 42
  --reps=<int>                 # default 3
  --output=<path.jsonl>
```

### 3.3 Run log 格式

每行一个 JSON object,见 [`00_overview.md`](00_overview.md) §5.3. 关键字段:

```json
{
  "run_id": "...",
  "git_commit": "...",
  "timestamp": "...",
  "cmd": "<full command line>",
  "config": {<full parsed config>},
  "hardware": {"gpu": "A100-40GB", "cpu": "...", "driver": "...", "cuda_or_rocm": "..."},
  "dataset": {"name": "twitter", "n_v": ..., "n_e": ..., "csr_hash": "..."},
  "rep": 0,
  "sssp_ms": 1234.5,
  "verifier_ms": 234.1,
  "verifier_verdict": "SAT",
  "teps": 1.19e9,
  "cert_summary": {"d_hash": "...", "pi_hash": "...", "n_unreachable": 12}
}
```

`d_hash` / `pi_hash`: CRC64 of d / pi arrays. **Cross-platform 比较直接 hash 比对 → 快速 drift filter;hash 不同时再 dump 完整 cert 详细分析。**

---

## 4. Error injection tooling (W11, for E4)

```cpp
// src/harness/error_injector.cpp
enum class ErrorKind {
    DISTANCE_PERTURB,         // d[v] += δ
    PREDECESSOR_RANDOM,       // π[v] = random other
    INCONSISTENT,             // d/π 解耦
    MISSED_UNREACHABLE,       // 设 d[v] = INF 实际可达
    CYCLE,                    // 制造 π cycle
};

template<typename W>
Certificate<W> inject_error(
    const Certificate<W>&  original,
    ErrorKind              kind,
    int                    seed,
    int                    n_errors = 1
);
```

CLI 集成:

```bash
run_sssp ... --emit_cert=true --output=correct.jsonl
inject_errors --input=correct.cert --kind=DISTANCE_PERTURB --n=10 \
              --output=corrupted.cert
verify_cert --certificate=corrupted.cert --output=verdict.json
```

E4 实验:`scripts/run_e4.sh` 自动跑 4 类 × 10 magnitude × 100 random seed = 4000 个 inject + verify,聚合 detection rate.

---

## 5. Drift analysis (W9-W10, for E1/E2)

### 5.1 Drift 计算

`src/analysis/drift_compare.py`:

```python
def compare_certificates(cert_a: Path, cert_b: Path) -> DriftReport:
    """
    Returns:
      n_d_diff: # vertices with byte-different d
      n_pi_diff: # vertices with byte-different π
      d_diff_magnitudes: list of |d_a[v] - d_b[v]| for differing v
      both_verifier_sat: bool
    """
```

输出 `drift_<datasetA-vs-B>.csv`:

```csv
dataset,gpu_a,gpu_b,precision,seed,
n_d_diff,n_pi_diff,both_sat,
d_diff_p50,d_diff_p99,d_diff_max
```

### 5.2 Drift mechanism attribution (E2)

W11 用 Nsight Systems / ROCprof 抓 kernel-level trace,分析:

- Reduction order (atomic 顺序记录,不在 prod path,只 debug build 启)
- Kernel launch sequence
- Numerical precision drift (FP32 mid-computation vs FP64 control)

工具不写新的,直接用 vendor profiler + Python 解析 trace JSON. E2 分析在 W11,~3 天 wall-clock.

---

## 6. 实验脚本 (W8 准备 + W9-W11 执行)

每个实验一个 shell + Python pair:

| Script | 作用 |
|---|---|
| `scripts/run_e1.sh` | Drift baseline: 5 GPU × 8 dataset × 3 precision × 3 rep |
| `scripts/run_e3.sh` | Verifier soundness: E1 outputs → verify against Boost.Graph |
| `scripts/run_e4.sh` | Error injection × 4 types × magnitudes × seeds |
| `scripts/run_e6.sh` | Emission overhead: baseline vs augmented TEPS |
| `scripts/run_e7.sh` | Verifier cost vs SSSP recompute |
| `scripts/run_e2.sh` | Mechanism attribution profiling |
| `scripts/run_e5.sh` | Apply verifier on Gunrock + cuGraph outputs |
| `scripts/run_e8.sh` | Scaling: RMAT-22/23/24/25/26 |
| `scripts/run_e9.sh` | Stress: weight × precision × structure cross |

每个 script 写 JSONL,Python 聚合到 CSV → matplotlib 出图 → LaTeX table。

`scripts/dispatch_all.sh`:

```bash
#!/bin/bash
set -e
./scripts/run_e1.sh
./scripts/run_e3.sh
./scripts/run_e6.sh
./scripts/run_e7.sh
./scripts/run_e4.sh
./scripts/run_e2.sh
./scripts/run_e5.sh
./scripts/run_e8.sh
./scripts/run_e9.sh
python -m analysis.make_figures   # 出全部论文 figures
```

---

## 7. Reproducibility infrastructure

### 7.1 Environment lock

`scripts/env_lock.sh`:

```bash
#!/bin/bash
# 校验当前 host 满足 lock 文件
expected_cuda="12.4"
expected_rocm="6.0"

actual_cuda=$(nvcc --version | grep -oP 'release \K[\d.]+')
[[ "$actual_cuda" != "$expected_cuda" ]] && {
    echo "CUDA version mismatch: expected $expected_cuda, got $actual_cuda"
    exit 1
}
# ... rocm, cmake, gcc, etc.
```

每次 run 前 harness 调用 `env_lock.sh`,失败则拒绝跑 (写 log 标记 `env_locked=false`,论文用数据必须 `env_locked=true`)。

### 7.2 随机性

- C++ 端: 一个 global `std::mt19937` seed from CLI
- Python 端: `numpy.random.seed(SEED)` + `random.seed(SEED)`
- GPU kernel 不引入新随机源 (no curand etc.)
- Δ-stepping 内部 race 不可避免 (parallel),通过 tiebreak rule 保证 final state 确定

### 7.3 Run log 不可变

每个 run 写完 log 后,filename 包含 git commit + 时间戳,**不覆写**. Re-run 写新 file. 分析阶段 glob 全部 file.

```
results/
├── 2026-06-15_w9_e1/
│   ├── run_a100_twitter_fp32_seed42_<hash>.jsonl
│   ├── run_mi250_twitter_fp32_seed42_<hash>.jsonl
│   └── ...
└── 2026-06-22_w10_e6/
    └── ...
```

---

## 8. 性能 profiling (W4 + W11)

### 8.1 NVIDIA: Nsight Systems / Nsight Compute

```bash
nsys profile -o profile_a100_twitter \
    ./build_cuda/src/harness/run_sssp \
        --dataset=twitter --algo=delta_stepping_gpu --emit_cert=true
```

W4 用于 SSSP baseline tuning. W11 用于 E2 mechanism attribution。

### 8.2 AMD: ROCprof / Omnitrace

```bash
rocprof --stats --hip-trace --hsa-trace \
    ./build_rocm/src/harness/run_sssp ...
```

输出 trace JSON,Python 解析 atomic ordering / kernel sequence.

---

## 9. 论文 figures pipeline (W12+)

`src/analysis/make_figures.py` 出 §VI、§VII、§VIII 全部 figures:

```python
# 论文 figure ↔ 实验 ID 映射
FIGURES = {
    "fig1_drift_heatmap":          ["e1"],
    "fig2_drift_mechanism_pie":    ["e2"],
    "fig3_correctness_drift_split":["e1", "e3"],
    "fig4_coverage_compare":       ["e4"],
    "fig5_emission_overhead_bar":  ["e6"],
    "fig6_verifier_vs_recompute":  ["e7"],
    "fig7_scaling":                ["e8"],
    "fig8_stress_heatmap":         ["e9"],
}
```

每个 figure 生成函数读对应 JSONL → 聚合 → matplotlib → save `.pdf`. 所有 figure 单一命令再生:

```bash
python -m analysis.make_figures --output=paper/figures/
```

LaTeX paper 直接 `\includegraphics{figures/fig1_drift_heatmap.pdf}`.

---

## 10. 不写的东西 (反 over-engineering)

- 不写 web dashboard / TUI 监控 — bash + tail 够
- 不写 distributed scheduler / Kubernetes — 手工 ssh + 脚本
- 不写 database — JSONL + pandas 够
- 不写 typed config schema validator — argparse + assert 够
- 不写 telemetry / metrics export — run log JSONL 已经够细
- 不写 custom CI pipeline — GitHub Actions stock template 够
- 不写 artifact versioning — git + filename timestamp 够
- 不写 access control / auth — 单人项目
