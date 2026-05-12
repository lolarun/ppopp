# 00 — Engineering Overview

**Document purpose:** 工程蓝图,不是 normative spec. Module 边界 + 关键技术决策 + 依赖锁定 + 仓库布局.

**Style note:** 不写 V2.4.2 governance 风格 (见 plans/00 anti-pattern #1). 没有 MUST/SHALL,没有稳定标识符,没有 verdict semantics. 只写 "我打算这样写代码"。

---

## 1. 系统组成

5 个模块,组合关系:

```
                    ┌─────────────────┐
                    │  Dataset loader │  (DIMACS / GAP / SNAP / RMAT → CSR)
                    └────────┬────────┘
                             │ CSR<W>
                             ▼
              ┌──────────────────────────────┐
              │   Δ-stepping (HIP unified)   │  → distance d
              │   + certificate emission     │  → predecessor π
              └──────────┬───────────────────┘
                         │ (d, π)
                         ▼
              ┌──────────────────────────────┐
              │   Verifier (CPU, OpenMP)     │  → SAT / UNSAT_*
              └──────────┬───────────────────┘
                         │ verdict + metrics
                         ▼
              ┌──────────────────────────────┐
              │   Run harness + log writer   │  → JSONL run log
              └──────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │   Analysis pipeline (Python) │  → drift tables / figures
              └──────────────────────────────┘
```

模块边界故意松,不强制 ABI 稳定,不做版本号管理。Refactor 自由。

---

## 2. 关键技术决策

### 2.1 GPU portability: HIP unified single-source

**决定:** 用 HIP 写一份代码,用 `hipcc` 编译,后端 CUDA (NVIDIA) 或 ROCm (AMD)。**不**维护 CUDA + HIP 两份分叉代码。

**理由:**
- HIP 与 CUDA API 几乎一一对应,`hipify-perl` 可以自动转换 90%+ CUDA 代码
- 单源代码减少维护成本,跨平台 drift 实验意义更纯
- AMD 上用原生 HIP runtime,NVIDIA 上 HIP 透明转 CUDA runtime,性能差异可忽略

**风险接受:** HIP-on-NVIDIA 在 atomic-heavy workload (如 Δ-stepping) 实测开销 **10-20%**,不是 5-10%。可接受 (论文焦点是正确性,不是 SOTA 性能)。W3 micro-benchmark 确认具体数字。

**Fallback 触发条件:** 如果 W4 NVIDIA 上 HIP 性能 < Gunrock 的 30% (即慢 3 倍以上),切到 CUDA + HIP 双源代码,W5 多花 3-4 天 port。

### 2.2 Δ-stepping baseline 来源

**决定:** **从头写 lean HIP Δ-stepping** (~1500-2500 行 GPU 代码),不 fork Gunrock。

**理由:**
- Gunrock 是 CUDA-only 重型框架, hipify 它代价高且不支持 ROCm production-ready
- 单算法实现可控,容易加 certificate emission
- 性能目标: NVIDIA 上达到 Gunrock 的 50-70% TEPS 即够 (论文不靠性能,只要 reviewer 不质疑 baseline 离谱弱即可)
- Gunrock 仍作为**性能参考**用,benchmark 时一起跑

**Fallback:** 如 W3-W4 baseline 性能 < Gunrock 30%,改用 Gunrock fork + 仅 NVIDIA 路径,AMD 路径用 rocGraph 或自己写简化版,放弃 single-source。

### 2.3 Verifier: CPU only,OpenMP parallel

**决定:** CPU verifier,C++17 + OpenMP。不写 GPU verifier。

**理由:**
- O(V+E) 单次扫描在 100M 边图上 CPU 跑 < 30 秒 (实测 LiveJournal 量级 ~10s)
- GPU verifier 工程开销高 (~1-2 周),收益边际 (verifier 不在 paper performance 主线)
- E7 (verifier vs recompute) 用 CPU verifier vs GPU recompute,差距更明显,故事更干净

**Fallback:** 如果 W10 测出 CPU verifier 在最大图 (>1B 边) 跑 > 5 分钟,加一个 GPU verifier (用 HIP unified 思路,~3 天)。

### 2.4 Multi-precision

**决定:** 模板化 CSR + 算法,实例化 `float`、`double`、`__half`。运行时通过 CLI flag 选。

**实现:** `template<typename W> class CSR { ... };` + `extern template` 显式实例化避免编译爆炸。

### 2.5 Reproducibility

**决定:**
- 所有随机源固定 seed (从 CLI 传入)
- 依赖版本 lockfile (CUDA / ROCm / cmake / boost 具体版本)
- 每次 run 写一个 JSONL log,含 git commit hash + 完整命令行 + 全部参数 + 硬件标识

**不做:**
- 不做 reproducible BLAS (Demmel et al.) — 这恰恰是论文论点的反面
- 不强制 bit-exact across runs (不可能且不必要)

### 2.6 Build: CMake 3.20+

**决定:** CMake 单一 build system,不用 Make / Bazel / Meson。

**理由:** ROCm 和 CUDA 都对 CMake first-class 支持. CMake 3.20+ 原生 HIP language support.

---

## 3. 依赖锁定

| 依赖 | 版本 | 用途 |
|---|---|---|
| C++ standard | C++17 | 主语言 |
| CMake | ≥ 3.21 | build |
| CUDA Toolkit | 12.4 (lock) | NVIDIA backend |
| ROCm | 6.1 (lock) | AMD backend |
| HIP | 随 ROCm 6.0 / 也用 hipcc-cuda 路径 | unified GPU code |
| OpenMP | 4.5+ | CPU verifier 并行 |
| Boost.Graph | 1.83 | CPU SSSP reference (E3 ground truth) |
| nlohmann/json | 3.11 | run log JSON 写入 |
| fmt | 10.x | 日志格式 |
| Catch2 | 3.x | unit test |
| Python | 3.11 | analysis pipeline |
| pandas + matplotlib | 2.x / 3.8 | 表格 / 图 |

**Lockfile 形式:** `cmake/Dependencies.cmake` 显式 `find_package(... EXACT VERSION ...)`,版本不匹配直接 build 失败。

**升级策略:** 4 月窗口内**不升级任何依赖**。Lock once at W2,frozen 直到 submission。

---

## 4. 仓库布局

```
gpu-sssp-certifying/
├── CMakeLists.txt
├── cmake/
│   ├── Dependencies.cmake          # 依赖版本锁
│   ├── ToolchainCUDA.cmake
│   └── ToolchainROCm.cmake
├── docs/
│   ├── plans/                      # 论文规划 (existing)
│   └── ts/                         # 工程蓝图 (本目录)
├── external/                       # git submodule
│   └── gunrock/                    # 仅作为性能参考 baseline
├── src/
│   ├── core/
│   │   ├── csr.h                   # CSR<W> template
│   │   ├── graph_types.h
│   │   └── precision.h             # FP32/FP64/FP16 wrappers
│   ├── sssp/
│   │   ├── cpu_dijkstra.cpp        # 纯 CPU reference (Boost.Graph wrap)
│   │   ├── cpu_delta_stepping.cpp  # CPU Δ-stepping (sanity check)
│   │   ├── delta_stepping.hip      # GPU Δ-stepping HIP unified
│   │   ├── certificate.hip         # π emission augmentation
│   │   └── tiebreak.h              # 一致性 tiebreak rule
│   ├── verifier/
│   │   ├── cpu_verifier.cpp
│   │   ├── invariants.h
│   │   └── tree_check.cpp
│   ├── io/
│   │   ├── csr_io.cpp              # 内部二进制 CSR 序列化
│   │   ├── dimacs_loader.cpp
│   │   ├── gap_loader.cpp
│   │   ├── snap_loader.cpp
│   │   └── rmat_generator.cpp
│   ├── harness/
│   │   ├── main.cpp                # CLI entry: run_sssp
│   │   ├── run_config.h
│   │   ├── error_injector.cpp      # E4 用
│   │   └── log_writer.cpp          # JSONL 输出
│   └── analysis/                   # Python
│       ├── drift_compare.py        # E1
│       ├── coverage_compare.py     # E4
│       ├── overhead_summary.py     # E6
│       └── make_figures.py         # 论文 figures
├── tests/
│   ├── test_csr.cpp
│   ├── test_delta_stepping_small.cpp
│   ├── test_verifier_soundness.cpp
│   └── test_emission_correctness.cpp
├── data/                           # gitignored, 由 scripts/fetch_datasets.sh 填充
├── scripts/
│   ├── fetch_datasets.sh
│   ├── run_e1.sh                   # 跑 cross-platform drift
│   ├── run_e3.sh                   # 跑 verifier soundness
│   ├── run_e4.sh                   # error injection
│   ├── run_e6.sh                   # emission overhead
│   ├── run_e7.sh                   # verifier cost
│   └── env_lock.sh                 # 打印 + 校验环境
├── results/                        # gitignored, 全部 run log
└── README.md
```

---

## 5. 模块间数据格式 (轻量 contract)

不写 normative spec,只把关键格式定下来,改动需要全员同步。

### 5.1 CSR (内存内 + 内部二进制序列化)

```cpp
template<typename Weight>  // float / double / __half
struct CSR {
    using vid_t = uint32_t;
    using eid_t = uint64_t;

    vid_t n_vertices;   // uint32_t  (NOT eid_t — this was a typo)
    eid_t n_edges;
    std::vector<eid_t>     row_offsets;   // size n_vertices + 1
    std::vector<vid_t>     col_indices;   // size n_edges
    std::vector<Weight>    weights;       // size n_edges
};
```

二进制序列化: little-endian, 简单 header (magic + version + n_v + n_e + weight_dtype) + raw arrays. 无 schema validation,只读自己写的文件。

### 5.2 Certificate output (run-time + 可选磁盘)

```cpp
template<typename Weight>
struct Certificate {
    std::vector<Weight>   d;    // size n_vertices, ∞ 用 sentinel
    std::vector<vid_t>    pi;   // size n_vertices, root/unreachable 用 INVALID_VID
};
```

磁盘格式: 同 CSR,简单 binary,不做版本兼容。

### 5.3 Run log (JSONL,一行一条 run)

```json
{
  "run_id": "uuid-string",
  "timestamp": "2026-06-15T14:23:01Z",
  "git_commit": "abcdef1",
  "cmd": "run_sssp --dataset=twitter --gpu=A100 --precision=fp32 --algo=delta_stepping --emit_cert=true --seed=42",
  "hardware": {"gpu": "A100-40GB", "driver": "550.90.07", "cuda": "12.4", "rocm": null},
  "dataset": {"name": "twitter", "n_vertices": 41652230, "n_edges": 1468365182},
  "result": {
    "wall_time_ms": 1234.5,
    "teps": 1.19e9,
    "verifier_verdict": "SAT",
    "verifier_time_ms": 234.1,
    "emission_overhead_pct": 8.3
  }
}
```

每个 experiment script 解析 JSONL 聚合表格。

### 5.4 CLI 接口 (run_sssp)

```
run_sssp \
  --dataset=<path-or-name> \
  --gpu=<gpu-id>             # auto / 0 / "A100" / "MI250"
  --precision=<fp16|fp32|fp64> \
  --algo=<dijkstra_cpu|delta_stepping_cpu|delta_stepping_gpu> \
  --emit_cert=<bool> \
  --verify=<bool> \
  --seed=<int> \
  --reps=<int> \
  --log=<output.jsonl>
```

CLI flags 是事实上的契约;新增 flag 不破坏旧 script。

---

## 6. 测试策略

不追求高 coverage,目标是**捕获论文论点关键的正确性 bug**:

| 测试类 | 内容 | 跑频率 |
|---|---|---|
| Unit (Catch2) | CSR 序列化、tiebreak 规则、verifier invariant 单条 | 每次 commit |
| Integration small | 10 个 hand-crafted 小图 (5-50 顶点),已知 SSSP 答案 | 每次 commit |
| Integration medium | DIMACS 小图 vs Boost.Graph reference | nightly (W3 起) |
| Cross-platform parity | 同图 NVIDIA + AMD 输出,verifier 必须都 SAT | W5 起每次 push |
| Soundness fuzzer | 随机生成图 + 故意注入错误 (E4 同代码),verifier 必须 reject | W7 起每次 push |

CI: GitHub Actions 跑 CPU 测试,GPU 测试在 W3 起每周一次手工跑 (云 GPU 没 CI).

---

## 7. 不做的事 (反 scope creep)

显式列出来,自己知道也告诉协作者:

- **不写 GPU verifier** (CPU 够用,见 2.3)
- **不写 dynamic graph** (Layer 2.3 future paper)
- **不写 reduction kernel** (Layer 2.2)
- **不维护 CUDA + HIP 两份分叉代码** (HIP unified)
- **不做 reproducible BLAS 风格 bit-exact 复现** (论点反面)
- **不写 formal verification** (Layer 3,Coq/Lean 不碰)
- **不写 distributed multi-GPU** (单 GPU 即可)
- **不写 incremental SSSP** (静态批处理)
- **不构建 governance / qualification / audit 层** (anti-pattern #1)

---

## 8. 阅读顺序

按你的角色读不同文档:

- 写 GPU 算法 → `01_sssp_and_emission.md`
- 写 verifier → `02_verifier.md`
- 写 build / dataset / harness / 实验脚本 → `03_infrastructure.md`
- 整体决策 / 依赖问题 → 本文档

每份文档自洽,跨文档引用最少。
