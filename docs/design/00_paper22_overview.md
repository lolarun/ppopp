# Paper 2.2 设计文档 — Overview (v2)

**Paper title (working):** Auditable GPU Graph Reduction under Heterogeneity
**Layer:** Layer 2.2 (Execution / Certificates — Reduction class)
**Target venue (独立版):** PPoPP 2027 (primary), SC 2027 (backup)
**Target venue (合并版):** PPoPP 2027 — 与 Paper 2.1 合并为分类框架论文
**Predecessor:** Paper 2.1 (Certifying GPU SSSP under Heterogeneity)
**Timeline:** 2026-05-12 启动，2026-09 初 submit
**合并决策点:** 2026-08-01
**Author structure:** 独立作者
**Capability profile:** Type B (algorithm-first, atomic-disciplined, systems-capable)

---

## v1 → v2 主要变更

| 维度 | v1 (2025 设计) | v2 (2026-05 更新) |
|---|---|---|
| 启动时间 | 2027-09（Paper 2.1 ship 后一年） | 2026-05-12（与 Paper 2.1 并行） |
| 执行周期 | 8 个月 | 4 个月 |
| Venue | ASPLOS 2028 (12 页) | PPoPP 2027 (10 页) |
| 与 2.1 的关系 | 独立后继论文 | 独立设计，8 月考虑合并 |
| 实验规模 | 700-1500 GPU-hours | 150-400 GPU-hours（独立版）/ 50-100（合并版） |

**变更原因:** Paper 2.1 和 2.2 的 deadline 同期（PPoPP 2027 ~2026-09 初）。独立设计、并行推进、8 月决定是否合并是最优策略——不管最终合不合并，两条线的工作都不浪费。

---

## 文档结构

| 文档 | 内容 |
|---|---|
| `00_paper22_overview.md` | 本文档 — 总览 + 决策 summary |
| `01_problem_thesis.md` | Problem statement + framing variants（含合并版 framing） |
| `02_paper_outline.md` | Paper structure — 独立版 10 页 + 合并版 3 页压缩方案 |
| `03_experimental_design.md` | 实验设计 — 4 个月压缩版 |
| `04_timeline_milestones.md` | Timeline — 2026-05 到 2026-09 |
| `05_risk_register.md` | Risk register — 含合并相关风险 |
| `06_re0_runbook.md` | RE0 GO/NO-GO 执行手册（hardware / build / decision logic） |

---

## 与 Paper 2.1 的关系

### 继承
- 整体 program (Auditable Graph-Structured Computation under Heterogeneity)
- Atomic discipline + certificate methodology
- Cross-vendor evaluation infrastructure (NVIDIA + AMD)
- HIP unified codebase
- Verifier-vs-golden-output thesis

### 区别
- **Kernel class**: 2.1 = path-functional (SSSP, min-plus semiring); 2.2 = reduction-class (PageRank, sum semiring)
- **Drift behavior**: 2.1 发现 byte-exact (counterintuitive); 2.2 预期 drift (expected — 但必须 RE0 验证)
- **Certificate structure**: 2.1 = (d, π) path certificate; 2.2 = output value + reduction tree witness + tolerance bound
- **Verifier**: 2.1 = O(V+E) edge-inequality check; 2.2 = O(E log V) reduction-tree walk

### 正反对照价值
2.1 + 2.2 构成 GPU 图计算确定性的完整光谱：
- SSSP (min-plus): **不漂** — 因为 min 是 exact 选择操作
- PageRank (sum): **漂** — 因为 FP sum 是 non-associative

这个正反对照是合并投 PPoPP 的核心叙事基础。

---

## Kernel 选择

**Primary: PageRank**
- 真实 reduction-heavy：每 iteration 对 in-neighbors 做 weighted sum
- FP non-associativity 直接影响结果
- 已有 GPU 实现 baseline (Gunrock, cuGraph)
- 学术 + 工业关注度高
- 是唯一能和 SSSP 构成"漂 vs 不漂"正反对照的候选

**不选的候选及原因:**
- BFS: SSSP 在 unit-weight 上的退化，不走出 min-plus 半环
- Bottleneck path: 全 exact 操作，结果可预测
- INT64 任何算法: 整数精确结合，trivially byte-exact
- Triangle Counting: 整数计数，FP 不影响

**GO/NO-GO gate:** RE0 实验确认 PageRank 跨 vendor 真的漂移。如果不漂，重新评估。

---

## 核心 thesis

### 独立版 (10 页)
> GPU-parallel iterative graph reductions (PageRank) produce byte-different outputs across heterogeneous platforms due to FP non-associative summation. Reduction-tree certificates with provenance witnesses enable O(E log V) verification with principled tolerance derivation from reduction depth and FP precision, distinguishing legitimate FP drift from algorithmic corruption.

### 合并版 (2.2 部分, ~2.5 页)
> PageRank 的 sum-reduction 外层算子在 GPU 上因并行 reduce 顺序不同而跨 vendor 漂移——与 SSSP 的 byte-exact 形成对照。边界在于代数结构：idempotent exact meet (min) vs non-associative sum。

---

## 合并决策框架

**决策时间: 2026-08-01**

### 判断 1: PageRank 数据质量
- 漂移量化完整、有 interesting pattern → 合并有料
- 只是"确认漂了"没有深度 → 不合并

### 判断 2: 10 页容量
- 试写合并版 outline，SSSP 3 页 + PageRank 2.5 页 + 框架/方法论 4.5 页能成立 → 合并
- SSSP 的 Theorem 5.1 压缩后不 convincing → 不合并

### 判断 3: 叙事自然度
- "不漂 + 漂 → 边界在代数结构" 讲得顺 → 合并
- 两部分各说各的 → 不合并

### 合并后的去向
- 合并 → PPoPP 2027 一篇分类框架论文
- 不合并 → 2.1 单独投 PPoPP 2027; 2.2 数据保留，补充 certificate 部分，独立投 ASPLOS 2028 / PPoPP 2028

---

## 接收概率估计

| 方案 | Venue | 接收概率 |
|---|---|---|
| 2.2 独立投 | PPoPP 2027 | 30-40% |
| 2.1 + 2.2 合并投 | PPoPP 2027 | 45-55% |
| 2.2 独立投（被拒后改投） | ASPLOS 2028 | 25-35% |

---

## Anti-patterns (避免重蹈)

1. 不预设 drift 存在 — RE0 必须先跑
2. 不在 ground truth 缺乏时 lock 详细 spec
3. 不 over-engineer certificate（合并版可能不需要完整 certificate）
4. 不为了合并而牺牲 2.1 的深度
5. 不到 8 月 1 日之前做合并决策
