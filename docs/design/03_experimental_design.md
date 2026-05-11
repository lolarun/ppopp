# Paper 2.2 — Experimental Design (v2)

**Document purpose**: 4 个月实验规划（2026-05 to 2026-09）
**Resource constraint**: NVIDIA + AMD GPU access (复用 Paper 2.1 setup), Google Colab (T4/A100 supplementary)
**合并决策点**: 2026-08-01

---

## 实验分层

实验分为三层，对应不同目标:

| 层 | 目标 | 实验 | 合并版是否需要 |
|---|---|---|---|
| **Layer A: GO/NO-GO** | 确认 drift 存在 | RE0 | 是 |
| **Layer B: 合并版必须** | 漂移量化，构成正反对照 | RE1, RE2 | 是 |
| **Layer C: 独立版必须** | Certificate + verifier 证据 | RE3-RE7 | 否（合并版砍掉） |

**执行顺序**: A → B → (8月判断) → C (如果不合并)

---

## RE0: Pre-launch drift verification (CRITICAL)

**Goal**: 确认 PageRank 跨 vendor 真的漂移。Paper 2.1 的教训——SSSP 预期漂但没漂。

**Deadline: 2026-05 第三周（启动后 10 天内）**

**Setup**:
- 2 GPUs: 1 NVIDIA (A10 或 Colab T4) + 1 AMD (MI300X)
- 3 datasets: web-google, livejournal, RMAT-22
- FP32, damping=0.85, max_iter=100, tolerance=1e-6
- 最简实现: 50 行 HIP push-based PageRank，或直接用 cuGraph/hipGraph

**Measurements**:
- byte-different vertex fraction
- max |y_NVIDIA - y_AMD|
- mean element-wise difference
- L2 norm of difference vector

**Decision logic**:

| 结果 | 行动 |
|---|---|
| > 1% vertices byte-different | **GO** — 按计划推进 |
| 0.1-1% byte-different | GO but framing 需谨慎量化 |
| < 0.1% byte-different | **STOP** — 假设被 falsify |

**如果 STOP:**
- Option 1: PageRank 也不漂 → 这本身是 surprising finding，可以 reframe 2.1+2.2 为 "更多算法也不漂" 的泛化论文
- Option 2: 切换到 betweenness centrality（更重的 FP reduction）
- Option 3: 2.1 单独投

**Resource**: 2-4 GPU-hours, 1 天 wall-clock

---

## RE1: Cross-vendor PageRank drift baseline

**Goal**: 在完整硬件 × dataset matrix 上量化漂移。

**Setup**:
- GPUs: NVIDIA A10 + AMD MI300X + Colab T4 (Turing) + Colab A100 (如果拿到)
- Datasets: web-google, livejournal, RMAT-20, RMAT-22, road networks (2-3 个), social networks (1-2 个)
- Precisions: FP32, FP64
- 每配置 5 次 run

**Measurements (per configuration)**:

| Metric | 含义 |
|---|---|
| byte_diff_fraction | 跨 vendor byte-different 的 vertex 比例 |
| max_Linf | max |y_A - y_B| across all vertices |
| mean_diff | 平均 element-wise 差异 |
| L2_norm | difference vector 的 L2 范数 |
| rank_order_diff | PageRank 排序（top-100）是否一致 |
| cross_run_variance | 同卡多次 run 的 variance |

**输出**: 一张 Table，类似 Paper 2.1 的 17-cell matrix

**Resource**: 30-60 GPU-hours

---

## RE2: Drift mechanism attribution

**Goal**: 把漂移归因到具体原因。

**Setup**: 受控实验，每次只变一个变量:

| 实验 | 固定 | 变量 | 隔离的因素 |
|---|---|---|---|
| RE2a | 同 GPU, 同 library | 不同 run | Warp scheduling |
| RE2b | 同 GPU | 不同 CUDA/ROCm 版本 | Library 实现 |
| RE2c | 同算法, 同 library | 不同 GPU vendor | 硬件 + scheduling |
| RE2d | 同 GPU | FP32 vs FP64 | 精度 |
| RE2e | 同配置 | Atomic add vs non-atomic | Atomic discipline |

**特别关注 RE2e**: 这和 Paper 2.1 的 strict vs relaxed 实验对称。如果 PageRank 用 atomic add 和不用 atomic add 结果都漂（预期如此——因为漂移来源是 FP non-associativity 不是 atomics），那进一步确认代数结构是 primary boundary。

**Resource**: 20-40 GPU-hours

---

## RE3: Fixed-tree reduction 对照 (合并版 optional, 独立版 recommended)

**Goal**: 用 deterministic reduction 跑 PageRank，验证它变成 byte-exact，量化性能代价。

**Setup**:
- 标准 PageRank（non-deterministic parallel reduce）
- Fixed-tree reduction PageRank（deterministic reduce order）
- 同硬件，同 dataset

**Measurements**:
- Fixed-tree 版本跨 vendor 是否 byte-exact（预期: 是）
- 性能代价: fixed-tree vs standard 的 TEPS ratio

**价值**:
- 独立版: 证明 "确定性可以买到，但有代价；certificate 是更便宜的替代"
- 合并版: 一个数据点说明 "sum-reduction 也可以 byte-exact，但代价是 X%"，与 SSSP 的 "零代价 byte-exact" 对比

**Resource**: 10-20 GPU-hours

---

## RE4: Verifier soundness validation (独立版 only)

**Goal**: 验证 verifier-accepted 的输出确实正确。

**Setup**:
- 对每个 verifier-accepted 输出，与高精度 reference (CPU FP128 或 FP64 串行) 比较
- 100+ instances

**Measurement**: accepted 中有多少与 reference 匹配
**Expected**: 100%

**Resource**: 10-20 GPU-hours (主要是 CPU 验证)

---

## RE5: Verifier vs tolerance comparison (独立版 core evidence)

**Goal**: 证明 verifier 优于 hand-tuned tolerance。

**Setup**:
- 在 PageRank 输出中注入 4 类错误:
  - Type 1: Random vertex value corruption (单点大错误)
  - Type 2: Systematic bias (所有值偏移一个小常数)
  - Type 3: Convergence short-circuit (提前终止迭代)
  - Type 4: Reduction order corruption (改变部分 reduction tree)
- 对每类错误 × 多个 magnitude:
  - Verifier detection rate
  - Hand-tuned tolerance detection rate (ε = 1e-3, 1e-4, 1e-5, 1e-6)

**Measurements**:
- True positive rate (sensitivity)
- True negative rate (specificity)
- False positive/negative comparison

**Resource**: 20-40 GPU-hours

---

## RE6: Certificate emission overhead (独立版 only)

**Goal**: 量化 certificate 记录的性能开销。

**Setup**: Baseline PageRank vs augmented PageRank (with reduction tree recording)

**Target**: < 20% overhead
**Resource**: 20-40 GPU-hours

---

## RE7: Verifier cost (独立版 only)

**Goal**: Verifier wall-clock vs PageRank recompute time。

**Target**: Verifier 5-10x faster than recompute
**Resource**: 10-20 GPU-hours

---

## 实验与目标的映射

| 实验 | 合并版需要 | 独立版需要 | 证明什么 |
|---|---|---|---|
| RE0 | ✓ | ✓ | Drift 存在 (GO/NO-GO) |
| RE1 | ✓ | ✓ | Drift 量化 |
| RE2 | ✓ (RE2a, RE2c, RE2e) | ✓ (全部) | Drift 归因 |
| RE3 | Optional | ✓ | Deterministic reduction 代价 |
| RE4 | ✗ | ✓ | Verifier soundness |
| RE5 | ✗ | ✓ | Verifier vs tolerance |
| RE6 | ✗ | ✓ | Certificate overhead |
| RE7 | ✗ | ✓ | Verifier cost |

---

## Resource budget

| 场景 | 实验 | GPU-hours | 估计成本 |
|---|---|---|---|
| 合并版 minimum | RE0 + RE1 + RE2 (部分) | 50-100 | $100-200 |
| 合并版 + RE3 | + RE3 | 60-120 | $120-250 |
| 独立版 full | RE0-RE7 全部 | 150-300 | $300-600 |

远低于 v1 设计的 700-1500 GPU-hours——因为时间压缩 + 合并场景不需要 certificate 实验。

---

## 实现依赖

### 从 Paper 2.1 复用
- HIP unified build system (CMake + HIP)
- Cross-vendor evaluation infrastructure
- JSONL run logging
- Dataset download / preprocessing scripts
- CRC32 hash comparison tooling

### 新实现
- GPU PageRank kernel (HIP, push-based or push-pull hybrid)
- Drift 量化脚本 (Python: L∞, L2, byte-diff fraction)
- Fixed-tree reduction variant (RE3)
- Certificate emission augmentation (独立版 only)
- Verifier algorithm (独立版 only)

### 实现优先级

| 优先级 | 组件 | Deadline |
|---|---|---|
| P0 | PageRank HIP kernel + drift comparison script | 2026-05 第三周 (RE0) |
| P1 | Full dataset matrix runner | 2026-06 中 (RE1) |
| P2 | Mechanism attribution experiments | 2026-07 中 (RE2) |
| P3 | Fixed-tree reduction variant | 2026-07 底 (RE3) |
| P4 | Certificate emission + verifier | 2026-08 (独立版 only, if needed) |
