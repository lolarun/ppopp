# 02 — Verifier

**Document purpose:** O(V+E) certificate verifier 的具体实现. CPU only,C++17 + OpenMP.

**对应论文章节:** §III.B-D (soundness, verifier, completeness)、§IV.C (verifier impl)、§VII (correctness detection evaluation)

---

## 1. 算法

输入: graph G = (V, E, w), source s, certificate (d, π)

输出: SAT 或 UNSAT_<reason>

```
verify(G, s, d, π):
    # Invariant 1: source distance
    if d[s] != 0:                     return UNSAT_SOURCE_DISTANCE
    if π[s] != INVALID_VID:           return UNSAT_SOURCE_PRED

    # Invariant 2: relaxation condition (∀ edge)
    for each edge (u, v, w) in E:
        if d[u] != INF and d[v] > d[u] + w:
            return UNSAT_RELAXATION

    # Invariant 3: predecessor consistency (∀ v ≠ s)
    for v in V \ {s}:
        if d[v] == INF:
            if π[v] != INVALID_VID:   return UNSAT_UNREACHABLE_PRED
            continue                  # unreachable, no further check
        if π[v] == INVALID_VID:       return UNSAT_REACHABLE_NO_PRED
        u = π[v]
        # 找边 (u, v, w_uv)
        w_uv = lookup_edge_weight(G, u, v)
        if w_uv == NOT_FOUND:         return UNSAT_PRED_NOT_NEIGHBOR
        if d[v] != d[u] + w_uv:       return UNSAT_PRED_DISTANCE_MISMATCH

    # Invariant 4: tree structure (no cycle, all reach s via π)
    if not is_tree_rooted_at_s(π, s):
        return UNSAT_CYCLE

    return SAT
```

**复杂度:** O(V + E) 单次扫描 + O(V + E) tree check = O(V + E) overall.

---

## 2. 文件结构

```
src/verifier/
├── invariants.h           # Verdict enum + helper structs
├── cpu_verifier.cpp       # 主 entry: verify(...)
├── relaxation_check.cpp   # Invariant 2 (并行)
├── predecessor_check.cpp  # Invariant 3 (并行)
└── tree_check.cpp         # Invariant 4 (parallel union-find)
```

---

## 3. 接口

```cpp
// src/verifier/invariants.h
enum class Verdict {
    SAT,
    UNSAT_SOURCE_DISTANCE,
    UNSAT_SOURCE_PRED,
    UNSAT_RELAXATION,
    UNSAT_UNREACHABLE_PRED,
    UNSAT_REACHABLE_NO_PRED,
    UNSAT_PRED_NOT_NEIGHBOR,
    UNSAT_PRED_DISTANCE_MISMATCH,
    UNSAT_CYCLE,               // π contains a cycle (π forms a loop, not a tree)
    UNSAT_DISCONNECTED_TREE,   // π forms a tree but NOT rooted at source
};
// Note: UNSAT_CYCLE and UNSAT_DISCONNECTED_TREE are distinct.
// CYCLE = union-find detects same root before merging (actual cycle in π).
// DISCONNECTED_TREE = π is acyclic but some reachable vertex's root ≠ source root.

struct VerifyResult {
    Verdict verdict;
    std::optional<vid_t>  witness_vertex;   // 第一个失败顶点 (debug 用)
    std::optional<eid_t>  witness_edge;     // 第一条失败边
    double                wall_time_ms;
};

template<typename W>
VerifyResult verify(
    const CSR<W>&           g,
    vid_t                   source,
    std::span<const W>      d,
    std::span<const vid_t>  pi,
    int                     num_threads = 0   // 0 = OpenMP default
);
```

---

## 4. 并行化策略

### 4.1 Invariant 2 (relaxation, 主 cost)

这是最贵的一步: O(E) 边扫描. OpenMP parallel for over edges:

```cpp
template<typename W>
Verdict check_relaxation_parallel(const CSR<W>& g,
                                  std::span<const W> d) {
    std::atomic<bool> failed{false};
    std::atomic<eid_t> witness_edge{INVALID_EID};

    #pragma omp parallel for schedule(static, 4096)
    for (eid_t u = 0; u < g.n_vertices; ++u) {
        if (failed.load(std::memory_order_relaxed)) continue;
        if (d[u] == Sentinel<W>::inf) continue;        // unreachable u

        for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u+1]; ++e) {
            vid_t v = g.col_indices[e];
            W     w = g.weights[e];
            if (d[v] > d[u] + w) {
                failed.store(true);
                witness_edge.store(e);
                break;
            }
        }
    }
    return failed.load() ? Verdict::UNSAT_RELAXATION : Verdict::SAT;
}
```

注:每个线程查到 fail 后还是会继续 iter 直到 schedule chunk 结束 — 不是 short-circuit perfect,但成本很低 (`failed.load()` cheap on hot cache line)。

### 4.2 Invariant 3 (predecessor consistency)

每个顶点独立检查,parallel for over vertices:

```cpp
#pragma omp parallel for schedule(static, 1024)
for (vid_t v = 0; v < g.n_vertices; ++v) {
    if (v == source) continue;
    // ... per-vertex check
}
```

`lookup_edge_weight(G, u, v)` 在 CSR 上需要 binary search over `col_indices[row_offsets[u] : row_offsets[u+1]]` (假设邻居有序). 如果 CSR 邻居未排序,加一步 W2 完成的 CSR sort pass.

**优化:** 如 `n_v` 大且 `lookup_edge_weight` 命中率低,改用 hash lookup (但 ~1.5x memory). 默认用 binary search,O(log d_u) per lookup,d_u 通常 << V.

### 4.3 Invariant 4 (tree check)

需要确认 π 没有 cycle (除 source 自指,但 π[s] = INVALID),且每个 reachable 顶点经 π 链能到 s.

**算法 (parallel union-find):**

```
for each v ≠ s with π[v] != INVALID:
    union(v, π[v])
after all unions: ∀ reachable v, find(v) == find(s)
```

如果有 cycle,union-find 不会出错,但最终 root 不会是 s 的 root. 用 parallel union-find (Bender et al. 2020-ish) 或 simpler serial (W7 不调优,先正确)。

**Serial 版 (W7 默认):**

```cpp
std::vector<vid_t> parent(g.n_vertices);
std::iota(parent.begin(), parent.end(), 0);
auto find = [&](vid_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
};
for (vid_t v = 0; v < g.n_vertices; ++v) {
    if (v == source) continue;
    if (pi[v] == INVALID_VID) continue;
    vid_t ru = find(pi[v]), rv = find(v);
    if (ru == rv) return Verdict::UNSAT_CYCLE;     // already same root → cycle
    parent[rv] = ru;                                // union
}
// 验证 ∀ reachable v, find(v) == find(source)
vid_t root_s = find(source);
for (vid_t v = 0; v < g.n_vertices; ++v) {
    if (d[v] == Sentinel<W>::inf) continue;
    if (find(v) != root_s) return Verdict::UNSAT_CYCLE;
}
return Verdict::SAT;
```

复杂度: nearly O(V) amortized with path compression.

如 100M-顶点图上 serial > 5 秒,W10 改 parallel union-find. Otherwise leave it.

---

## 5. 性能目标

**目标:** 100M-edge 图上 verify wall-time < 30 秒.

实测预期 (基于 LiveJournal ~70M 边经验):
- Invariant 2 (relaxation): ~5 秒 (memory-bound, 大约 10 GB/s effective bandwidth)
- Invariant 3 (predecessor): ~3 秒
- Invariant 4 (tree): ~1 秒
- 总计: < 10 秒

10x faster than recompute (Δ-stepping on same graph ~1-2 minutes CPU multi-threaded) → 满足论文 §VIII.B claim "verifier << recompute".

---

## 6. 测试策略

### 6.1 Soundness sanity (W7)

**已知正确输入 → 必须 SAT:**

1. CPU Boost.Graph dijkstra 输出 → verify SAT
2. CPU Δ-stepping 输出 → verify SAT
3. 10 个手算小图 → verify SAT

如果任一返回 UNSAT,说明 verifier 实现有 bug,优先修。

### 6.2 注错测试 (W7 + 配合 E4)

注入 4 类错误,验证 verifier reject 对应类:

| 错误类型 | 注入操作 | 期望 verdict |
|---|---|---|
| Wrong distance | `d[v] += δ` (δ > 0) | UNSAT_RELAXATION 或 UNSAT_PRED_DISTANCE_MISMATCH |
| Wrong predecessor | `π[v] = random_other` | UNSAT_PRED_DISTANCE_MISMATCH |
| Inconsistent | `d[v]` 与 `π[v]` 解耦 | UNSAT_PRED_DISTANCE_MISMATCH |
| Missed unreachable | 设 `d[v] = INF` 但实际可达 | UNSAT_RELAXATION (一定有边能 relax) |
| Cycle in π | 制造 π[a]=b, π[b]=a | UNSAT_CYCLE |

每类至少跑 100 次随机图 + 随机注错位置,verifier 必须 100% catch.

### 6.3 Cross-platform sanity (W7+)

NVIDIA 输出 + AMD 输出 都过 verifier,都应 SAT (除非有真 bug,见 E5)。

---

## 7. Soundness argument (论文 §III.B)

**不**自己证 novel theorem. 直接 cite + reuse:

- **Reference:** McConnell, Mehlhorn, Näher, Schweitzer. "Certifying algorithms." *Computer Science Review* 5.2 (2011).
- 该 reference 已包含 SSSP certificate 的 soundness 证明 (relaxation invariant + tree → distance correct)
- 论文 §III.B 写: "我们的 verifier 实现 invariant 1-4 of [McConnell et al.]; soundness 由 [McConnell et al. Theorem 4.2] 直接给出. 我们 narrow theorem 到 connected graphs with non-negative weights, 这是我们的工作 scope."

**Risk R8 mitigation:** 不写 novel formal proof. 只 implement + cite. 即使 reviewer 要求,也可指向 McConnell 等的 textbook-level 结果。

---

## 8. GPU verifier (defer)

**默认不写**. 但保留接口 hook:

```cpp
// src/verifier/gpu_verifier.hip   (空文件 or stub)
template<typename W>
VerifyResult verify_gpu(...);  // not implemented in W7
```

W10 实测后,**仅当** CPU verifier 在最大 dataset 上 > 5 分钟才启用,~3 天移植 invariant 2 到 HIP kernel (其他 invariant 量小,留 CPU)。

---

## 9. CLI / harness 集成

```bash
# Standalone usage
verify_cert \
  --graph=twitter.csr \
  --certificate=twitter_run42.cert \
  --source=0 \
  --threads=16 \
  --output=verify_report.json

# Harness 内集成 (默认 run_sssp 完跑 verifier)
run_sssp ... --verify=true   # 把 verifier 时间也写进 run log
```

`verify_report.json` 输出格式:

```json
{
  "verdict": "SAT",
  "wall_time_ms": 234.1,
  "phases_ms": {"invariant_2": 145.3, "invariant_3": 67.2, "invariant_4": 21.6},
  "witness": null
}
```

UNSAT 时 witness 含具体 vertex/edge:

```json
{
  "verdict": "UNSAT_RELAXATION",
  "wall_time_ms": 18.4,
  "witness": {"vertex": 12345, "edge_id": 99887, "u": 12340, "v": 12345, "w": 1.5,
              "d_u": 10.0, "d_v": 12.0}
}
```

Debug 用,论文里不展示 witness 细节。
