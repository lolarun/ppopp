# Paper 2.2 — Problem Statement & Core Thesis (v2)

**Document purpose**: Lock framing for Reduction certificate paper. Framing depends on RE0 confirming drift exists.

---

## Section 1: The Underlying Phenomenon

### 1.1 业界 folklore 的证据链

"GPU 并行计算结果跨硬件漂移"不是论文自己竖的稻草人，有硬证据支撑：

**官方文档级:**
- NVIDIA CCCL 3.1 定义三级确定性 (gpu-to-gpu / run-to-run / not-guaranteed)，专门为此开发 deterministic reduction API，代价 20-30%
- PyTorch 官方："Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms"
- cuBLAS: "same bit-wise results" 限定 "same architecture and same number of SMs"

**直接测量:**
- Shanmugavelu et al. (SC'24 Workshop): 1000 次相同输入产生 1000 组独一无二的模型权重
- Davidson et al. (GPU SSSP): "distances are updated in parallel in a non-deterministic order"
- TAO (2025): "cross-platform non-determinism is intrinsic to production GPU stacks"

**关键区分:** 上述证据主要针对 **sum-reduction 类操作**（矩阵乘、梯度累加、PageRank 等）。Paper 2.1 发现 min-plus SSSP 不在此类——但这个区分从未被直接测量过。Paper 2.2 在 sum-reduction 端做直接测量，与 Paper 2.1 构成完整的正反对照。

### 1.2 为什么 reduction 和 path computation 不同（理论）

Paper 2.1 发现 SSSP byte-reproducible，原因:
- min-plus semiring: min 是 exact 选择操作，不涉及舍入
- 加法只沿单条路径发生，不跨路径 reduce
- atomic CAS 保证 per-vertex update 的 linearizability

Reduction class 不同:
- sum semiring: FP 加法不满足结合律
- Reduction 是**跨多条边/路径**的聚合，不是沿单条路径
- 并行 reduce 的顺序由 warp scheduling 决定，跨硬件不同

**这个理论区分是两篇论文放在一起时的核心贡献。**

### 1.3 方法论空白

当 PageRank 输出跨平台不同时，现有方法:
- Tolerance-based comparison: 需要 hand-tuned ε，太紧拒绝合法 drift，太松放过真错误
- Reproducible BLAS (Demmel et al.): byte-exact 但 20-30% 性能代价
- Algorithmic redundancy: 重跑 + 多数投票，成本高

**都不是 certificate-based correctness validation。**

---

## Section 2: Problem Statement Variants

### Variant A: Tolerance-derived certificate (独立版推荐)

> "GPU-parallel iterative graph reduction (PageRank) on heterogeneous platforms produces byte-different outputs due to FP non-associative summation. Conventional tolerance-based comparison uses hand-tuned epsilon that is often miscalibrated. We propose reduction-tree certificates with provenance witnesses enabling principled tolerance derivation from reduction depth and FP precision, with O(E log V) verifier cost."

**适用场景:** 2.2 独立投 PPoPP/ASPLOS

### Variant B: 正反对照 framing (合并版推荐)

> "GPU graph computation exhibits a sharp determinism boundary determined by algebraic structure. Path-class algorithms (SSSP, min-plus semiring) achieve cross-vendor byte-exact results; reduction-class algorithms (PageRank, sum semiring) exhibit measurable cross-vendor drift. We characterize this boundary through controlled experiments on NVIDIA and AMD GPUs, identify scheduling purity as the secondary boundary within the byte-exact regime, and demonstrate that the algebraic structure of the outer operator (exact meet vs non-associative sum) is the primary determinant."

**适用场景:** 2.1 + 2.2 合并投 PPoPP

### Variant C: 分类框架 framing (合并版升级)

> "We propose a three-dimensional classification framework — (algebraic structure, atomic discipline, scheduling purity) — for predicting GPU graph computation determinism. Empirical validation on two canonical algorithms (SSSP and PageRank) across NVIDIA and AMD GPUs demonstrates the framework's predictive power: it correctly classifies byte-exact, race-tolerant, and drift-prone regimes."

**适用场景:** 合并版如果 PageRank 数据足够丰富

---

## Section 3: 推荐 framing 选择

**独立版:** Variant A — 聚焦 tolerance-derived certificate
**合并版:** Variant B — 正反对照，如果数据足够丰富可升级到 Variant C

**决策时间: 2026-08-01，基于 PageRank 数据质量。**

---

## Section 4: Core Thesis

### 独立版 (一句话)
> GPU-parallel iterative graph reductions produce byte-different outputs across heterogeneous platforms due to FP non-associative summation; reduction-tree certificates enable principled verification distinguishing legitimate drift from algorithmic corruption.

### 合并版 (一句话)
> GPU graph computation determinism is predictable from algebraic structure: idempotent-meet algorithms (SSSP) are byte-exact while non-associative-sum algorithms (PageRank) drift, with scheduling purity as a secondary boundary within the exact regime.

### Claim 组件

**独立版 5 claims:**
1. Drift exists: PageRank 跨 vendor byte-different (RE0 验证)
2. Certificate captures provenance: 不只是 output value
3. Verifier complexity bounded: O(E log V)
4. Tolerance is principled: derived from reduction depth + FP precision
5. Coverage meaningful: 抓到 hand-tuned tolerance 漏掉的错误

**合并版 4 claims:**
1. SSSP byte-exact cross-vendor (from 2.1)
2. PageRank drift cross-vendor (from 2.2)
3. 边界在代数结构 (min exact vs sum non-associative)
4. Scheduling purity 是 byte-exact regime 内的二级边界 (from 2.1 §5)

---

## Section 5: What this paper does NOT claim

- 不 claim reproducible computation（不要求 byte-exact，接受 drift 但验证正确性）
- 不 claim certificate 泛化到所有图算法
- 不 claim new PageRank algorithm
- 不 claim formal verification（defer to Layer 3）
- 不 claim performance improvement（我们加了 overhead，justify 其价值）
- 合并版不 claim min-plus 结论泛化到所有 idempotent semiring（只验证了 SSSP）

---

## Section 6: Key terms (lock vocabulary)

| Term | Meaning | Don't confuse with |
|---|---|---|
| Reduction tree | 并行 reduction 操作的层次结构 | Computational graph |
| Provenance witness | 记录哪些输入贡献了哪些输出 | Data provenance (数据系统) |
| Reduction depth | Reduction tree 的层数 | Iteration count |
| Principled tolerance | ε derived from depth + precision | Hand-tuned epsilon |
| Drift mass | 跨平台输出差异的聚合幅度 | Drift rate |
| Algebraic boundary | 决定确定性的代数结构分界线 | Hardware boundary |
| Scheduling purity | Work-set 选择不依赖被 race 污染的信号 | Scheduling fairness |

---

## Section 7: RE0 Pre-launch verification (CRITICAL)

**Paper 2.2 的第一件事，在 commit 任何 framing 之前:**

跑 PageRank on NVIDIA + AMD，2-3 个 dataset，byte-compare 输出。

**Decision logic:**
- Drift > 1% of vertices byte-different → GO, Variant A/B framing
- Drift 0.1-1% → GO but framing 需要 careful quantification
- Drift < 0.1% → **STOP**: hypothesis falsified
  - Option 1: 这本身是一个 surprise finding（类似 SSSP），可以 reframe
  - Option 2: 切换到 connected components / betweenness centrality
  - Option 3: 不做 2.2，2.1 单独投

**RE0 必须在 2026-05 第三周完成。**
