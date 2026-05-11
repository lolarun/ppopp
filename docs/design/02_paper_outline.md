# Paper 2.2 — Paper Outline (v2)

**Target (独立版):** 10 pages, ACM acmart sigplan, PPoPP 2027
**Target (合并版):** ~2.5 pages within merged 10-page paper
**Note:** 独立版和合并版同步设计，8 月决定走哪条路

---

## A. 独立版结构 (10 pages PPoPP)

| Section | 篇幅 | 角色 |
|---|---|---|
| §1 Introduction | 1 页 | Hook, motivate, contribution list |
| §2 Background | 1 页 | Reductions, FP non-associativity, prior methods |
| §3 Theoretical Foundation | 1.5 页 | 为什么 reduction 漂而 SSSP 不漂 |
| §4 Certificate Design | 1.5 页 | Reduction tree certificate + tolerance derivation |
| §5 Verifier | 1 页 | O(E log V) verifier + soundness |
| §6 Drift Characterization | 1.5 页 | Cross-vendor/cross-run 漂移量化 |
| §7 Verifier Coverage | 1 页 | Detection rate vs tolerance methods |
| §8 Performance | 0.75 页 | Emission overhead + verifier cost |
| §9 Related Work + Discussion | 0.5 页 | Position vs ReproBLAS, certifying algorithms |
| §10 Conclusion | 0.25 页 | |
| **Total** | **10 页** | |

### §1 Introduction (1 页)

**P1 — Phenomenon:**
GPU PageRank 在不同硬件上产生不同输出。业界已知（NVIDIA CCCL 三级确定性、PyTorch disclaimer），通过 tolerance 绕过。

**P2 — Gap:**
Tolerance 需要 hand-tuned ε。太紧拒绝合法 drift，太松放过真错误。ReproBLAS byte-exact 但性能代价 20-30%。都不是 principled certificate-based verification。

**P3 — Approach:**
Reduction-tree certificates + provenance witnesses。Tolerance 从 reduction depth × FP precision 推导，不是 hand-tuned。Verifier O(E log V)，比 recompute 便宜。

**P4 — Connection to path-class:**
Paper 2.1 (cite) 发现 SSSP byte-exact cross-vendor。本文在 reduction 端做对照——PageRank 漂。边界在代数结构：min (exact) vs sum (non-associative)。两篇合起来构成 GPU 图计算确定性的完整分类。

**P5 — Contributions:**
- C1: Cross-vendor PageRank 漂移的直接测量与量化
- C2: Reduction-tree certificate design with principled tolerance derivation
- C3: O(E log V) verifier with soundness argument
- C4: Theoretical explanation: semiring structure predicts determinism regime

### §2 Background (1 页)

- §2.1 PageRank 数学公式 + GPU 并行实现要点
- §2.2 FP non-associativity: 为什么并行 reduce 顺序影响结果
- §2.3 Prior methods: tolerance / ReproBLAS / redundancy

### §3 Theoretical Foundation (1.5 页)

- §3.1 Semiring perspective: path-class (min, +) vs reduction-class (+, ×)
- §3.2 为什么 min 是 exact 而 sum 不是：外层算子的代数性质决定确定性
- §3.3 推论：预测其他算法（BFS = exact, betweenness centrality = drift, connected components = exact）

### §4 Certificate Design (1.5 页)

- §4.1 Certificate = (y, R, T)：output values + reduction tree + tolerance bound
- §4.2 Compact reduction tree encoding（provenance summary, O(E log V) space）
- §4.3 Tolerance derivation: ε(v) = depth(R, v) × machine_eps × max|contribution at v|
- §4.4 Soundness: 如果 certificate 满足 invariants，则 |y - y_true|∞ ≤ T

### §5 Verifier (1 页)

- §5.1 算法：walk provenance, check tolerance, verify DAG consistency
- §5.2 Complexity: O(E log V)
- §5.3 Soundness argument (cite Demmel FP error bounds)
- §5.4 Limitations: what verifier catches vs misses

### §6 Drift Characterization (1.5 页)

- §6.1 Cross-vendor drift: NV vs AMD, per-dataset, per-precision
- §6.2 Cross-run drift: same GPU, multiple runs
- §6.3 Drift magnitude distribution across vertex set
- §6.4 Comparison with theoretical prediction (§3 predicts drift, §4.3 bounds it)

### §7 Verifier Coverage (1 页)

- §7.1 Controlled error injection: 4 types, detection rate
- §7.2 Verifier vs hand-tuned tolerance: false positive / false negative comparison
- §7.3 Real-world library audit (Gunrock/cuGraph PageRank, if time)

### §8 Performance (0.75 页)

- §8.1 Certificate emission overhead (target < 20%)
- §8.2 Verifier cost vs recompute (target 5-10x faster)

### §9 Related Work + Discussion (0.5 页)

Position vs ReproBLAS, certifying algorithms (McConnell et al.), GPU PageRank optimization literature

### §10 Conclusion (0.25 页)

---

## B. 合并版结构 (PageRank 部分 ~2.5 页)

合并论文总结构:

| Section | 来源 | 篇幅 |
|---|---|---|
| §1 Introduction | 新写（分类框架叙事） | 1.5 页 |
| §2 Framework | 新写 | 1 页 |
| §3 Case A: SSSP | 2.1 压缩 | 3 页 |
| §4 Case B: PageRank | **2.2 压缩** | **2.5 页** |
| §5 Cross-case analysis | 新写 | 1 页 |
| §6 Methodology + Discussion | 合并 | 1 页 |
| **Total** | | **10 页** |

### PageRank 在合并版中保留什么 (§4, 2.5 页)

**必须保留:**
- §4.1 Cross-vendor drift 存在 + 量化（L∞, L2, byte-different fraction）(0.75 页)
- §4.2 为什么漂: sum non-associativity + parallel reduce ordering (0.5 页)
- §4.3 Cross-run variance（同卡多次 run）(0.25 页)
- §4.4 Drift magnitude vs 理论预测 (0.5 页)
- §4.5 可选: fixed-tree reduction 对照（byte-exact 但性能代价多少）(0.5 页)

**合并时砍掉:**
- Certificate design（§4 独立版）→ 移到 future work
- Verifier algorithm（§5 独立版）→ 移到 future work
- Verifier coverage（§7 独立版）→ 移到 future work
- Error injection experiment → 移到 future work
- Performance evaluation → 只保留 drift 量化，不做 verifier 性能

**合并版的 PageRank 贡献变为:**
- 漂移的直接测量 + 量化
- 与 SSSP byte-exact 的正反对照
- 代数结构解释

Certificate 和 verifier 留给 2.2 独立版（如果不合并）或 Layer 2.2 后续论文。

---

## C. 写作 checklist

### 独立版
- [ ] §3 theoretical section 是论文核心 — 必须严谨
- [ ] 数字具体（"73% of vertices byte-different"），不用 "substantial drift"
- [ ] 不 claim reproducibility（和 Paper 2.1 不同）
- [ ] Honest about verifier limitations
- [ ] 引用 Paper 2.1 简要，不 over-reference

### 合并版
- [ ] PageRank 部分自包含（reviewer 不需要读过 2.1 论文）
- [ ] 正反对照叙事清晰（不是两篇论文的物理拼接）
- [ ] SSSP 部分不因压缩而丢失 scheduling purity 的论证力度
- [ ] Framework section (§2) 是新贡献，不只是 summary
