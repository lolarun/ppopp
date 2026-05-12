# 01 — SSSP Baseline + Certificate Emission

**Document purpose:** Δ-stepping 算法在 CPU 和 GPU (HIP unified) 上的具体实现,以及如何加 predecessor π emission. 工程蓝图,不是论文 §III/§IV 的完整规范。

**对应论文章节:** §II.A (Δ-stepping)、§III (Certificate)、§IV.A-B (Implementation)

---

## 1. CPU Δ-stepping reference (W1-W2)

**目的:** 作为 ground truth 用于 verifier 测试和 GPU 实现的 sanity check. **不**追求性能,只要正确。

**实现:** 直接 wrap Boost.Graph 的 `dijkstra_shortest_paths`,加一个 thin Δ-stepping 实现做对比。

```cpp
// src/sssp/cpu_dijkstra.cpp
template<typename W>
Certificate<W> dijkstra_cpu(const CSR<W>& g, vid_t source);

// src/sssp/cpu_delta_stepping.cpp
template<typename W>
Certificate<W> delta_stepping_cpu(const CSR<W>& g, vid_t source, W delta);
```

**测试:** 10 个 hand-crafted 小图 (有的有 ties,有的有 unreachable 顶点),hand-compute 答案,assert d 和 π 都对。

---

## 2. GPU Δ-stepping (HIP unified, W3-W5)

### 2.1 算法

标准 Δ-stepping (Meyer & Sanders 2003):

```
d[s] = 0; d[v] = ∞ for v ≠ s
B[0] = {s}; i = 0
while ∃ non-empty bucket:
    while B[i] non-empty:
        // light edges relaxation
        R = {(v, w(v, u)) | v ∈ B[i], (v, u) light edge}
        B[i] = ∅
        relax_all(R)        // updates d, π, may put vertices into buckets
    // heavy edges relaxation
    H = {(v, w(v, u)) | v removed from B[i], (v, u) heavy edge}
    relax_all(H)
    i += 1
```

GPU mapping: 每 phase 一个 kernel,bucket 用 device-side array + atomic counter.

### 2.2 文件结构

```
src/sssp/
├── delta_stepping.hip          # 主 entry point + driver loop
├── delta_stepping_kernels.hip  # 各 GPU kernel
├── bucket.hip                  # bucket 数据结构 + 操作
└── tiebreak.h                  # 一致性 tiebreak rule
```

### 2.3 主 entry point

```cpp
// src/sssp/delta_stepping.hip
template<typename W>
Certificate<W> delta_stepping_gpu(
    const CSR<W>& g_host,
    vid_t source,
    W delta,
    bool emit_certificate     // 关键 flag: false = baseline, true = augmented
);
```

实现要点:
- 一份 kernel 实现两种模式,通过 `if constexpr` 或 template bool 编译期分叉,**不**写两份 kernel
- Host-side 接收 baseline / augmented 都走同一函数,只是返回 `Certificate<W>` 中 π 是否填充

### 2.4 GPU 内存布局

| Buffer | Size | Type | Lifetime |
|---|---|---|---|
| `d_d` (distance) | n_v | W | 整个 SSSP |
| `d_pi` (predecessor) | n_v | vid_t | 整个 SSSP, **仅当 emit_certificate=true** |
| `d_row_offsets` | n_v + 1 | eid_t | 整个 SSSP |
| `d_col_indices` | n_e | vid_t | 整个 SSSP |
| `d_weights` | n_e | W | 整个 SSSP |
| `d_bid` (bucket_id) | n_v | int | per-vertex bucket index = ⌊d[v]/Δ⌋; INT_MAX = inactive |
| `d_front` | n_v (overprovision) | vid_t | current phase-A frontier |
| `d_removed` | n_v (overprovision) | vid_t | accumulated removed set for phase B |
| `d_fsz` / `d_rsz` | 1 each | int | frontier / removed set sizes (atomic counters) |
| `d_min_bid` | 1 | int | current i_min (parallel reduction result) |
| `d_updated` | 1 | int | any_updated flag for phase-A convergence |

**Note:** Old ping-pong design (d_bucket_a/b) replaced by bucket_id[] array + explicit frontier.
This is the correct Δ-stepping bucket structure; the per-phase ping-pong was algorithmically incorrect.

**Memory overhead from emission:** `+sizeof(vid_t) * n_v` = 4 bytes per vertex. 在 100M-顶点图上是 400MB,GPU 容易吃下。

### 2.5 Tiebreak rule

**规则:** 当多个 predecessor 给出相同的 d(v) 值时,保留 vertex_id **最小**的那个作为 π(v)。

**为什么需要:** Heterogeneous execution 下不同 GPU relaxation 顺序不同,如果不固定 tiebreak,π 可以合法地不同 → 不能直接比较 π,只能比较 verifier 是否都 accept。**固定 tiebreak 简化 cross-platform comparison**:同图同 source 同 weight 的 π 应该完全一致 (除 FP 精度差异)。

**实现:**

```cpp
// 当尝试 relax (u, v, w) 时:
//   new_d = d[u] + w
//   if new_d < d[v]:        // strict improvement
//       atomic update d[v], π[v] = u
//   elif new_d == d[v] && u < π[v]:   // tie, smaller vid wins
//       atomic update π[v] = u (d 不变)
```

实现用 `atomicCAS` on packed `(d, pi)` 64-bit word,或 lock-free pair-update,见 §3.

### 2.6 Light/heavy edge 划分

边 (u, v, w) light ↔ w ≤ Δ. Δ 选择影响性能但不影响正确性:

- 默认 Δ = max_weight / avg_degree (heuristic from literature)
- 可通过 CLI `--delta=N` 覆盖

不调 Δ 做 SOTA 性能,论文不靠这个。

---

## 3. Certificate emission augmentation (W6)

### 3.1 核心问题

Baseline 只更新 `d`. Augmented 需要在 relax 成功时一起更新 `π`. 关键是**保持 (d, π) 一致** —— d[v] 必须等于 d[π[v]] + w(π[v], v) (verifier invariant 4)。

### 3.2 Atomic 更新方案

**问题:** GPU relaxation 是并发的。两个 thread 同时 relax v,可能一个写 d、一个写 π,出现 d 和 π 不匹配。

**实际实现方案 (FP32 + FP64 统一):**

```cpp
// 1. 用 CAS-loop atomicMinFP 更新 dist[v]
W old = atomicMinFP(&dist[v], nd);

// 2. 严格改善 (nd < old): 无条件覆盖 pi[v]
if (nd < old) {
    atomicExch(&pi[v], u);         // first writer wins; subsequent strict wins overwrite
    atomicMin(&bucket_id[v], new_bid);
}
// 3. 平局 (nd == old): CAS-loop 保证最小 vid 赢
else if (nd == old) {
    atomicMinPi(&pi[v], u);        // CAS loop: install u only if u < current pi[v]
}
```

`atomicMinFP` 是 CAS loop (HIP 无原生 FP atomicMin),对 FP32 和 FP64 都用同一路径。`atomicMinPi` 是 32-bit CAS loop,对 vid_t=uint32_t 操作。

**为什么不用 64-bit packed atomic (原方案 A):**
- FP32+vid_t 打包 64-bit CAS 可行,但 FP64+vid_t 需要 96-bit,无法单一 atomic
- 双-atomic 方案更简单且对两种精度统一
- Correctness 通过 tiebreak CAS (atomicMinPi) 保证,不依赖 d/π 原子配对
- 最终态保证正确:verifier 在 SSSP 完全收敛后跑

**测试:** W6 写完后,跑 1000 次随机种子 random graph,每次都 verifier 必须 SAT。如果出现 UNSAT,大概率是 atomic race,debug。

### 3.3 Source 顶点的 π

`π[source] = INVALID_VID` (sentinel,定义为 `0xFFFFFFFF`)。

Verifier 第一条 invariant: `d[source] == 0 && pi[source] == INVALID_VID`.

### 3.4 Unreachable 顶点

`d[v] = INF_W` (sentinel,FP32 用 `+∞`,FP16 用 max representable),`π[v] = INVALID_VID`.

Verifier invariant: `pi[v] == INVALID_VID ⟺ d[v] == INF_W` (除 source).

### 3.5 Performance target

Emission 开销目标 < 15% (论文 thesis claim). 经验估计:
- 64-bit packed atomic on FP32: ~5-10% overhead (atomic 同 cacheline,无额外 round trip)
- Double-atomic on FP64: ~15-25% overhead (两次 atomic + reload)

**Tuning 优先级 (W6 末测后按需):**
1. Packed atomic over split atomic
2. Memory layout: AoS (DistPred[]) vs SoA (d[], pi[]) — packed CAS 强制 AoS
3. Cache hint (`__ldg`) for d_arr reads
4. Reduce false sharing: pad to cacheline

如 W6 末仍 > 25%,启用 fallback: post-hoc π reconstruction (见 §3.6)。

### 3.6 Fallback: post-hoc π reconstruction

如 in-flight emission 始终 > 25% overhead,改成:

1. SSSP 跑 baseline (只输出 d)
2. 跑完后,一次性扫描所有边 O(E),对每个 v 找一个满足 `d[v] == d[u] + w(u, v)` 的 u 作为 π[v]

```cpp
// FIX #3: use relative epsilon, NOT exact equality.
// d[v] == d[u] + w is exact-equality and fails on FP round-trip.
template<typename W>
void reconstruct_pi(const CSR<W>& g, vid_t source,
                    const std::vector<W>& dist, std::vector<vid_t>& pi) {
    const W eps8 = W{8} * std::numeric_limits<W>::epsilon();
    pi.assign(g.n_vertices, INVALID_VID);
    // pi[source] = INVALID_VID  (verifier invariant 1)
    for (vid_t u = 0; u < g.n_vertices; ++u) {
        if (dist[u] >= Sentinel<W>::inf / W{2}) continue;
        for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u+1]; ++e) {
            vid_t v = g.col_indices[e];
            W     w = g.weights[e];
            W    nd = dist[u] + w;
            W    dv = dist[v];
            if (dv >= Sentinel<W>::inf / W{2}) continue;
            W tol  = eps8 * (dv > W{1} ? dv : W{1});
            W diff = nd > dv ? nd - dv : dv - nd;
            if (diff <= tol && (pi[v] == INVALID_VID || u < pi[v]))
                pi[v] = u;  // tiebreak: smallest vid wins
        }
    }
}
```

复杂度 O(E),通常 < 1 秒。**这条路径在论文 §IV.B 作为 design alternative 讨论**——既是 fallback 又是论点 ("emission cost can be amortized to O(E) post-pass").

---

## 4. Multi-precision 实现

### 4.1 模板实例化

```cpp
// src/sssp/delta_stepping.hip
template Certificate<float>  delta_stepping_gpu<float>(...);
template Certificate<double> delta_stepping_gpu<double>(...);
template Certificate<__half> delta_stepping_gpu<__half>(...);
```

放在 `.hip` 文件末尾 explicit instantiation,header 只声明。

### 4.2 FP16 注意事项

- `__half` 在 NVIDIA 上 native (Volta+),AMD 上 HIP 提供等价 type
- 累加可能 overflow:`d + w` 可能溢出 FP16 表示范围。**Mitigation:** 累加用 FP32,结果存回 FP16 (mixed precision),或限制 FP16 实验只跑小图小权重
- Atomic on `__half`:NVIDIA 7.0+ / AMD CDNA2+ 支持原生,旧硬件需 packed 32-bit emulation

### 4.3 Sentinel 值

```cpp
template<typename W> struct Sentinel;
template<> struct Sentinel<float>  { static constexpr float  inf = std::numeric_limits<float>::infinity(); };
template<> struct Sentinel<double> { static constexpr double inf = std::numeric_limits<double>::infinity(); };
template<> struct Sentinel<__half> { static __half inf() { return __half_raw{0x7C00}; } };
```

---

## 5. 单元测试 + 集成测试

### 5.1 Unit (Catch2)

```cpp
// tests/test_delta_stepping_small.cpp
TEST_CASE("trivial graph: single edge") { ... }
TEST_CASE("disconnected graph: unreachable") { ... }
TEST_CASE("ties: smaller vid wins") { ... }
TEST_CASE("source has no outgoing edges") { ... }
```

每个用 hand-built CSR,assert (d, π) 完全等于 hand-computed。

### 5.2 Integration (medium)

W3 起,DIMACS 最小 4 个 road network (n_v < 1M),与 Boost.Graph dijkstra 比较 d,verifier 必须 SAT。

### 5.3 Cross-platform parity (W5 起)

每次 push,自动跑 NVIDIA + AMD 上同一 dataset (小):

- Verifier 必须都 SAT
- d 不要求 byte-exact (允许 FP drift)
- π 在 tiebreak 规则下应该一致 (除 FP near-tie 边界 case)
- 记录 drift 数据进 `drift_baseline.csv` 作为 E1 早期数据

### 5.4 Soundness fuzzer (W7 起)

随机生成图 (RMAT) → 跑 SSSP → 注入随机错误 (E4 同代码) → verifier 必须 reject. 如果 verifier accept 了被注错的输出,要么 verifier bug 要么注错 trivial,要 case-by-case 看。

---

## 6. 性能调优策略 (W4)

只在 W4 集中调一次,目标 NVIDIA 上达 Gunrock 50-70%. **不**追 SOTA。

调优 checklist (优先级降序):
1. Memory coalescing in edge relaxation kernel
2. Bucket size threshold (small bucket → kernel overhead 高,合并)
3. Light/heavy split 分两个 kernel,heavy 较少调用
4. Persistent kernel (避免 kernel launch overhead) — only if needed
5. Warp-level shuffle for bucket compaction
6. Dynamic parallelism — **不用** (复杂度不值)

如 W4 末仍 < Gunrock 30%,触发 §00 §2.2 fallback。

---

## 7. 已知限制 (写进论文 §IX)

- FP near-tie boundary: 当 d[v] 与 d[u] + w 在 FP epsilon 范围内,verifier 可能假性 reject 或 accept。论文标注此为 known limitation
- 我们不解决 negative weights (Δ-stepping 不支持)
- 我们不解决 huge graph (> GPU memory) — out-of-core 不在 scope
- Tiebreak 规则下,论文中的 π 不是 "唯一正确的 SSSP 树",只是 "确定性挑出的一棵 SSSP 树"
