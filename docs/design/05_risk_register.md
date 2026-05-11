# Paper 2.2 — Risk Register (v2)

**Document purpose**: 识别 Paper 2.2 风险 + 合并相关风险
**Review cadence**: 每两周 (4 个月压缩周期)

---

## 风险总览

| ID | 风险 | 概率 | 影响 | 阶段 |
|---|---|---|---|---|
| R1 | PageRank 不漂移 | 30-40% | Critical | Phase 0 |
| R2 | 4 个月时间不够 | 40% | High | 全程 |
| R3 | 合并后 SSSP 深度被稀释 | 50% | High | Phase 3A |
| R4 | 合并版 10 页塞不下 | 45% | High | Phase 3A |
| R5 | PageRank 数据没有 interesting pattern | 35% | Medium | Phase 1-2 |
| R6 | ReproBLAS 区分度不够 (独立版) | 50% | Medium | Phase 3B |
| R7 | GPU 访问中断 | 15% | High | 全程 |
| R8 | 与 Paper 2.1 时间冲突 | 30% | Medium | 全程 |
| R9 | Tolerance derivation theorem 有 gap (独立版) | 30% | High | Phase 3B |
| R10 | PPoPP reviewer 不关心 reproducibility | 35% | Medium | 提交后 |

---

## R1: PageRank 不漂移 (CRITICAL)

**概率**: 30-40%
**影响**: Critical — 杀死整个 2.2 framing

**场景**: RE0 发现 PageRank 跨 vendor drift < 0.1%。类似 Paper 2.1 中 SSSP 的 surprise。

**为什么概率不低**: Paper 2.1 已经证明 min-plus 不漂。如果你的 PageRank 实现恰好用了 deterministic 的 reduce path（比如 HIP 某些版本的 atomic add 在特定条件下走确定性路径），可能也不漂。

**Mitigation**:
- RE0 是 hard gate，Phase 0 第二周出结果
- 如果不漂:
  - **最佳方案**: 这本身是一个 finding — "连 sum-reduction 在 GPU 图算法上也不漂"。与 SSSP 合并后论文的 story 变成 "GPU 图计算在 atomic discipline 下的确定性比预期更广泛"
  - **备选 1**: 检查是否因为实现走了确定性路径（禁用 atomic、用 non-deterministic parallel reduce 重试）
  - **备选 2**: 切换到 betweenness centrality（更重的 FP reduction，更可能漂）
  - **备选 3**: 2.1 单独投

**Trigger**: RE0 结果 (05-25)

---

## R2: 4 个月时间不够

**概率**: 40%
**影响**: High

**场景**: PageRank kernel 调试 + 实验 + 写作 + (可能的) 合并重构，4 个月太紧。

**Mitigation**:
- Phase 0-2 (实验) 和 Phase 3 (写作/合并) 有清晰分界
- 合并版不需要 certificate 和 verifier，工作量大幅减少
- 实验规模已从 v1 的 700-1500 GPU-hours 压缩到 50-300
- 复用 Paper 2.1 大量 infrastructure

**如果仍然来不及**:
- 砍 RE3 (fixed-tree 对照) — nice-to-have 不是必须
- 砍 RE2 的 detailed attribution — 保留 RE2a (cross-run) + RE2c (cross-vendor)
- 合并版: PageRank 部分只保留 drift matrix + 一段 mechanism 解释

**Trigger**: Phase 1 末 (06-28) 评估进度

---

## R3: 合并后 SSSP 深度被稀释

**概率**: 50%
**影响**: High

**场景**: SSSP 从独立版 ~7 页压缩到合并版 3 页后，scheduling purity 的 three-algorithm boundary (Δ-stepping / BF / async push) 论证变得不 convincing。Theorem 5.1 的证明被移到 supplementary，reviewer 不看。

**Mitigation**:
- 8 月合并决策前试压缩，评估 scheduling purity 论证是否仍然站得住
- Theorem 5.1: 正文保留 theorem statement + 1 段 proof intuition (为什么 BF 的 scheduling purity 使它 race-tolerant)
- 完整证明放 supplementary（PPoPP 允许 supplementary material）
- 三算法对比表压缩为一张 compact table

**Decision input**: 合并决策会议 (08-01) 的第 4 项 agenda

---

## R4: 合并版 10 页塞不下

**概率**: 45%
**影响**: High

**场景**: SSSP 3 页 + PageRank 2.5 页 + Introduction 1.5 页 + Framework 1 页 + Methodology 1 页 + Cross-case 1 页 = 10 页。没有任何 buffer。如果 SSSP 或 PageRank 任一部分需要更多篇幅，溢出。

**Mitigation**:
- 严格的 section 篇幅预算，写作时逐 section 检查
- Methodology 可以压缩到 0.5 页（复用 Paper 2.1 的 setup description）
- Related work 合并到 Introduction 或 Framework 中，不单独 section
- Cross-case analysis 如果太长，核心数据用 table 呈现

**如果仍然塞不下**:
- 不合并 → Phase 3B

---

## R5: PageRank 数据没有 interesting pattern

**概率**: 35%
**影响**: Medium

**场景**: RE1 显示 PageRank 跨 vendor drift 存在但 uniform — 所有 dataset、所有精度的 drift 幅度差不多，没有 structure-dependent variation。论文变成"我们测了，漂了，漂了这么多，完。"

**Mitigation**:
- 选择结构多样的 dataset（power-law vs uniform degree, dense vs sparse）
- FP32 vs FP64 对比可能产生差异
- RE2e (atomic vs non-atomic) 可能产生 interesting finding
- 如果真的没有 pattern: 合并版仍然有价值（正反对照的一端），独立版可能太薄

**Trigger**: Phase 1 末 RE1 数据分析

---

## R6: 与 ReproBLAS 区分度不够 (独立版)

**概率**: 50%
**影响**: Medium

**场景**: Reviewer 说 "Demmel et al. 已经解决了 reproducible reduction；你的 certificate 跟 ReproBLAS 有什么区别？"

**Mitigation**:
- 清晰定位: ReproBLAS 目标是 byte-exact reproduction（高性能代价）；我们目标是 accept drift but verify correctness（低代价）
- 不同 goal: 他们是 reproducibility，我们是 correctness validation
- RE3 fixed-tree 对照可以量化 "byte-exact 的代价 vs certificate 的代价"

**仅影响独立版。合并版不做 certificate，不需要回答这个问题。**

---

## R7: GPU 访问中断

**概率**: 15%
**影响**: High

**Scenario**: AMD Developer Cloud / Colab 服务中断

**Mitigation**:
- 多个访问渠道: AMD Developer Cloud + Colab + (如果需要) 临时租用
- 实验数据本地备份
- NV 端实验可以全部在 Colab 完成（T4 免费）

---

## R8: 与 Paper 2.1 时间冲突

**概率**: 30%
**影响**: Medium

**场景**: Paper 2.1 的 LaTeX 转换 / Intro 重写 / Colab 实验占用了太多时间，2.2 进度落后。

**Mitigation**:
- Phase 0-1 期间 2.1 和 2.2 完全独立，不应冲突
- 如果 2.1 LaTeX 工作量超预期: 优先 2.1（已有完整实验数据），2.2 可以 delay
- Phase 3 如果合并: 两条线自然汇合，不再冲突

---

## R9: Tolerance derivation theorem 有 gap (独立版)

**概率**: 30%
**影响**: High

**场景**: ε(v) = depth(R, v) × machine_eps × max|contribution at v| 在某些 dataset 上 bound 不住实际 drift。

**Mitigation**:
- 合并版不做 tolerance derivation，不受影响
- 独立版: 早期在 RE1 数据上验证 bound 是否成立
- 如果 bound 太紧: refine theorem（加 additional terms）
- 如果 bound 太松: 文档为 known limitation

**仅影响独立版。**

---

## R10: PPoPP reviewer 不关心 reproducibility

**概率**: 35%
**影响**: Medium

**场景**: PPoPP PC 对 reproducibility 话题不感兴趣，觉得 "niche"。

**Mitigation**:
- PPoPP scope 明确包含 "parallel algorithms", "synchronization", "formal analysis" — 与论文 match
- 合并版的 scheduling purity + 代数结构分类框架是 parallel algorithm 的核心话题
- Introduction 用 deployment 场景（regulated workload, cross-vendor migration）motivate，不只是学术兴趣

---

## 风险调整后的预期结果

| 结果 | 估计概率 |
|---|---|
| RE0 STOP (PageRank 不漂) | 15-20% |
| 合并投 PPoPP 2027 且接收 | 25-30% |
| SSSP 单独投 PPoPP 2027 且接收 | 15-20% |
| 投出但被拒，改投 2028 venue | 20-25% |
| 时间不够，推迟到下一 cycle | 10-15% |

**PPoPP 2027 的总接收概率: ~40-50%**（合并版 + 独立版的加权平均）

---

## 风险 review schedule

| 日期 | Review 内容 |
|---|---|
| 05-25 | R1 resolved (RE0 结果) |
| 06-28 | R2 (时间), R5 (数据 pattern), R8 (2.1 冲突) 评估 |
| 07-27 | R3, R4 (合并可行性) 预评估 |
| 08-01 | **合并决策 — R3, R4, R5 最终评估** |
| 08-15 | R2 (时间) 最终评估，决定是否砍实验 |
