# Paper 2.2 — Timeline & Milestones (v2)

**执行周期**: 2026-05-12 to 2026-09 初
**PPoPP 2027 deadline**: ~2026-09-01 (estimated, based on PPoPP 2026 pattern)
**合并决策点**: 2026-08-01

---

## 高层时间线

| 阶段 | 时间 | 周数 | 焦点 |
|---|---|---|---|
| Phase 0 | 05-12 → 05-25 | 2 周 | RE0 GO/NO-GO + PageRank kernel |
| Phase 1 | 05-26 → 06-28 | 5 周 | PageRank 跑通双平台 + RE1 drift baseline |
| Phase 2 | 06-29 → 07-27 | 4 周 | RE2 mechanism attribution + RE3 对照 |
| **决策点** | **08-01** | — | **合并 or 独立** |
| Phase 3A (合并) | 08-01 → 09-01 | 4 周 | 合并论文重构 + 打磨 |
| Phase 3B (独立) | 08-01 → 09-01 | 4 周 | 独立版 certificate + verifier + 写作 |

---

## Phase 0: GO/NO-GO (05-12 → 05-25, 2 周)

### Week 1 (05-12 → 05-18)

**目标**: 最简 PageRank 跑通 + RE0 启动

- [ ] 写 50-行 HIP push-based PageRank kernel
- [ ] 本地 CUDA 编译验证（如有 NVIDIA GPU）或 Colab T4
- [ ] 准备 3 个 dataset: web-google, livejournal, RMAT-22
- [ ] 复用 Paper 2.1 的 dataset infrastructure

### Week 2 (05-19 → 05-25)

**目标**: RE0 完成，GO/NO-GO 决策

- [ ] NVIDIA 上跑 PageRank × 3 datasets × FP32
- [ ] AMD MI300X 上跑同样配置
- [ ] Byte-compare 输出: byte_diff_fraction, max_Linf, L2_norm
- [ ] **GO/NO-GO 决策**

**Phase 0 交付物**: RE0 结果 + GO/STOP 决策

**如果 STOP**: 评估替代方案（见 01_problem_thesis.md §7）

---

## Phase 1: Drift Baseline (05-26 → 06-28, 5 周)

### Week 3-4 (05-26 → 06-08)

**目标**: PageRank kernel 生产化

- [ ] Push-pull hybrid 优化（如果 push-only 性能差 >5x vs cuGraph）
- [ ] HIP unified source 双平台编译验证
- [ ] 与 CPU NetworkX PageRank 对比验证正确性
- [ ] JSONL run logging 集成（复用 Paper 2.1）

### Week 5-6 (06-09 → 06-22)

**目标**: RE1 — full drift baseline

- [ ] 扩展到 6-8 datasets
- [ ] FP32 + FP64 双精度
- [ ] NVIDIA A10 + AMD MI300X + Colab T4
- [ ] 每配置 5 runs
- [ ] 生成 drift matrix table (类似 Paper 2.1 的 17-cell matrix)

### Week 7 (06-23 → 06-28)

**目标**: RE1 数据分析 + 初步写作

- [ ] 分析 drift pattern: 哪些 dataset 漂得多？FP32 vs FP64 差异？
- [ ] rank order stability: top-100 PageRank 排名跨 vendor 是否一致？
- [ ] 写 §6 Drift Characterization 初稿 (独立版) / §4 初稿 (合并版)

**Phase 1 交付物**: 完整 drift matrix + 初步分析 + 部分写作

---

## Phase 2: Mechanism + 对照 (06-29 → 07-27, 4 周)

### Week 8-9 (06-29 → 07-13)

**目标**: RE2 — drift mechanism attribution

- [ ] RE2a: 同 GPU 多次 run (cross-run variance)
- [ ] RE2c: 同算法, NV vs AMD (cross-vendor)
- [ ] RE2e: Atomic add vs non-atomic store (对称 Paper 2.1 的 strict vs relaxed)
- [ ] 分析: drift 主要来自哪个因素？

### Week 10-11 (07-14 → 07-27)

**目标**: RE3 — fixed-tree reduction 对照

- [ ] 实现 fixed-tree reduction PageRank variant
- [ ] 跑 cross-vendor 比较: fixed-tree 版本是否 byte-exact？
- [ ] 量化性能代价: fixed-tree vs standard
- [ ] 如果 fixed-tree byte-exact → 强对照数据（确定性可以买到，代价是 X%）

**Phase 2 交付物**: Mechanism attribution 数据 + fixed-tree 对照 + 分析

---

## 合并决策点: 2026-08-01

### 输入
- RE0: drift 存在 ✓/✗
- RE1: drift matrix 完整度 + 数据质量
- RE2: mechanism 分析深度
- RE3: fixed-tree 对照结果
- Paper 2.1 SSSP: LaTeX 版本状态

### 决策流程

```
RE1 数据质量够吗？
├── 不够 → 不合并 → Phase 3B
└── 够
    ├── 试写合并版 outline (2 天)
    │   ├── 10 页塞不下 → 不合并 → Phase 3B
    │   └── 塞得下
    │       ├── 叙事自然 → 合并 → Phase 3A
    │       └── 叙事不自然 → 不合并 → Phase 3B
    └──
```

### 决策会议 agenda (08-01, self-review)
1. 列出 PageRank 所有实验结果的 key findings
2. 用 3 句话概括 PageRank 对合并论文的贡献
3. 试写合并版 abstract (~200 words)
4. 评估 SSSP 部分压缩到 3 页后 scheduling purity 论证是否仍然 convincing
5. 做决策

---

## Phase 3A: 合并版 (08-01 → 09-01, 4 周)

### Week 12-13 (08-01 → 08-17)

**目标**: 合并论文结构重构

- [ ] 写 §1 Introduction: 分类框架叙事
- [ ] 写 §2 Framework: (algebraic structure, atomic discipline, scheduling purity) 三维
- [ ] 压缩 Paper 2.1 SSSP 到 §3 (3 页): 17-cell matrix + strict vs relaxed + Theorem 5.1 (proof → supplementary)
- [ ] 整合 PageRank 到 §4 (2.5 页): drift matrix + mechanism + 对照

### Week 14-15 (08-18 → 09-01)

**目标**: 打磨 + 提交

- [ ] §5 Cross-case analysis: 框架预测力验证
- [ ] §6 Methodology + Discussion
- [ ] 内审: 10 页限制检查
- [ ] LaTeX 排版 (acmart sigplan 10pt)
- [ ] Double-blind 清理
- [ ] **Submit PPoPP 2027**

---

## Phase 3B: 独立版 (08-01 → 09-01, 4 周)

如果不合并，PageRank 独立推进，SSSP 单独投 PPoPP。

### Week 12-13 (08-01 → 08-17)

**目标**: Certificate + verifier 实现

- [ ] Certificate emission: 在 PageRank kernel 中记录 reduction tree
- [ ] Verifier 实现 (CPU, single-threaded)
- [ ] RE4: verifier soundness validation
- [ ] RE5: verifier vs tolerance comparison (error injection)

### Week 14 (08-18 → 08-24)

**目标**: 独立版写作

- [ ] 完整 10 页独立论文初稿
- [ ] RE6: certificate emission overhead
- [ ] RE7: verifier cost

### Week 15 (08-25 → 09-01)

**目标**: 打磨 + 提交

- [ ] 内审 + 修改
- [ ] LaTeX 排版
- [ ] SSSP 单独投 PPoPP 2027
- [ ] PageRank 独立投？或留给 ASPLOS 2028？

---

## 与 Paper 2.1 的并行关系

| 时间 | Paper 2.1 SSSP | Paper 2.2 PageRank |
|---|---|---|
| 05-12 → 06-15 | LaTeX 转换 + Intro 重写 | RE0 + kernel 实现 |
| 06-15 → 07-31 | 打磨到可投状态 + Colab T4 | RE1 + RE2 + RE3 |
| 08-01 | — | **合并决策** |
| 08-01 → 09-01 | 合并 → 整合 / 独立 → 单独投 | 合并 → 整合 / 独立 → certificate |

**关键约束**: 两条线不能互相阻塞。Phase 0-2 完全独立。Phase 3 才可能需要 coordination（合并版需要同时编辑同一篇论文）。

---

## Weekly check-in template

每周五 self-review:

```
Week N (日期):
- 本周完成: [实验 / 代码 / 写作]
- 数据发现: [任何 surprising 结果]
- 阻塞: [GPU 访问 / 技术问题 / 时间不够]
- 下周计划: [具体任务]
- 合并信号: [正面 / 负面 / 中性]
- 整体进度: [on track / behind / ahead]
```

---

## Critical milestones

| 日期 | Milestone | 影响 |
|---|---|---|
| **05-25** | RE0 GO/NO-GO | 决定 2.2 是否继续 |
| **06-28** | RE1 drift matrix 完成 | 合并版的核心数据 |
| **07-27** | RE2 + RE3 完成 | 合并决策的数据基础 |
| **08-01** | **合并决策** | 决定 Phase 3 走哪条路 |
| **09-01** | **Submit** | PPoPP 2027 |
