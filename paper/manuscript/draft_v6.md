# Cross-Vendor Bit-Reproducibility in GPU Graph Algorithms: It's the Atomics, Not the Arithmetic

<!-- PPoPP 2027 submission. ACM acmart sigplan, 10 pages, double-blind. -->
<!-- Draft v6, 2026-05-16 -->
<!-- Changes from v5: narrowed overclaim ("sole source" → "sole observed source in studied -->
<!-- kernels"); CRC32 metric clarified as presentation checksum with direct bytewise -->
<!-- comparison; FP preconditions (NaN, signed zero, subnormal) added to §3 and Theorem -->
<!-- 5.1; performance table expanded to 3 GPUs × 6 datasets; §6.3 intra-vendor variance -->
<!-- table populated for all GPUs from artifact re-run data; memory model assumptions made -->
<!-- explicit in Theorem 5.1 conditions -->

---

## Abstract

GPU graph algorithms are widely assumed to produce non-deterministic output across hardware platforms due to floating-point non-associativity and non-deterministic atomic scheduling. We challenge this assumption through a systematic cross-vendor empirical study of two algorithms that sit at opposite ends of the reduction spectrum: single-source shortest paths (SSSP), whose min-plus semiring makes distance computation path-functional, and PageRank, whose sum-reduction over neighbor contributions is order-dependent under IEEE 754 arithmetic.

On six graphs spanning road, web, social, and synthetic classes, across NVIDIA A10, Tesla T4, and AMD MI300X GPUs at both FP32 and FP64 precision, we find that SSSP produces byte-identical distance vectors cross-vendor under disciplined atomic compare-and-swap, while PageRank under standard atomic-add scatter exhibits drift on 24--100% of vertices depending on graph structure. Through a three-stage "onion-peeling" experiment --- removing atomic adds from the scatter loop, then from reduction kernels, then from all paths --- we isolate atomic-add scheduling as the *sole observed* source of drift in these kernels: a fully deterministic PageRank variant with zero atomic adds produces byte-identical output across all three GPUs from both vendors (180 runs, 6 datasets, 2 precisions). For the operations exercised in these kernels, IEEE 754 arithmetic is bit-exact across NVIDIA and AMD silicon when computation order is fixed.

We formalize the discriminating condition as an algebraic framework: the outer operator of the computation's semiring determines the determinism regime. Min (idempotent, commutative, order-independent) yields reproducibility by construction; sum (non-associative under finite precision) requires explicit scheduling control. Within the reproducible regime, we identify a finer boundary --- *scheduling purity* --- that determines whether relaxing atomic discipline (replacing CAS with non-atomic stores) breaks correctness: algorithms whose work-set selection depends on per-vertex signals corrupted by intra-iteration races break, while those that unconditionally scan all vertices tolerate the relaxation.

---

## 1 Introduction

GPU-parallel graph algorithms are deployed in production systems for social-network analytics, recommendation engines, and infrastructure monitoring. A recurring question in heterogeneous deployments --- where different GPU models, vendors, or driver versions coexist --- is whether the same algorithm on the same input produces the same output. The conventional wisdom is that it does not: floating-point arithmetic is non-associative, atomic operations resolve concurrent races non-deterministically, and both effects compound in iterative graph computations. NVIDIA's own documentation distinguishes three levels of determinism (bitwise, numerically-equivalent, non-deterministic) and classifies most parallel reductions as non-deterministic.

This assumption has practical consequences. Tolerance-based comparison ($\|y_A - y_B\|_\infty < \varepsilon$) requires choosing $\varepsilon$ --- too tight rejects legitimate platform variation, too loose admits silent errors. Golden-output comparison (requiring byte-identity) fails on algorithms that legitimately produce different-but-equally-valid outputs, such as shortest-path predecessors when multiple optimal trees exist. Deterministic-reduction libraries like ReproBLAS [riedy-demmel-reproblas] guarantee reproducibility but impose 20--30% runtime overhead and require algorithm-level integration. Hardware-enforced deterministic atomic buffering [chou-dab-2020, jooybar-gpudet-2013] requires microarchitectural modifications not available on commodity GPUs. None of these approaches answer the fundamental question: *where does the non-determinism actually come from, and how much of it is eliminable on existing hardware?*

We answer this question by studying two graph algorithms that sit at opposite ends of the algebraic spectrum:

- **SSSP** (single-source shortest paths via $\Delta$-stepping [meyer-sanders-2003]): the outer operator is $\min$, which is idempotent and order-independent under IEEE 754. Per-vertex distance updates use atomic compare-and-swap. We find that SSSP produces byte-identical distance vectors across NVIDIA and AMD GPUs on every tested configuration.

- **PageRank** (iterative eigenvector via power method): the outer operator is $\sum$, which is non-associative under finite-precision arithmetic. Per-vertex rank updates use atomic add. We find that PageRank exhibits cross-vendor drift on 24--100% of vertices, with magnitude spanning four orders depending on graph structure.

**PageRank: onion-peeling isolation.** The central finding emerges from a controlled elimination experiment on PageRank. We implement three kernel variants:

1. **Push** (baseline): atomic-add scatter to neighbors. Cross-vendor drift on all datasets.
2. **Pull v1**: CSC-transpose read from in-neighbors, no scatter atomic add. *Still drifts* on 5/6 datasets --- residual atomic adds in auxiliary reduction kernels (dangling-mass sum, convergence residual) are sufficient to break byte-identity.
3. **Pull v2**: zero atomic adds anywhere --- block-level partial sums written to buffer, host-side sequential reduction. **Byte-identical across all three GPUs from both vendors.** 180 runs (6 datasets $\times$ 2 precisions $\times$ 5 seeds $\times$ 3 GPUs) produce identical CRC32 checksums per (dataset, precision) cell, confirmed by direct bytewise comparison.

This "onion-peeling" methodology establishes that atomic-add scheduling is the *sole observed* source of PageRank drift in these kernels. No contribution comes from hardware floating-point implementation differences (which would persist in Pull v2), compiler differences (nvcc vs. amdclang++), or microarchitectural variation (Turing vs. Ampere vs. CDNA3).

**SSSP: scheduling purity.** Within the reproducible regime, we further discover that the per-vertex atomic CAS is not universally necessary for distance correctness. A controlled race-injection probe (`RELAX_ATOMICS`) reveals that removing CAS breaks $\Delta$-stepping on every dataset (5/5 unique distance vectors per seed, all UNSAT) but leaves Bellman-Ford byte-exact and SAT. The natural explanation --- that iterate-to-fixedpoint host loops repair races --- is refuted by a third algorithm (asynchronous push) that has the same host loop structure yet breaks identically to $\Delta$-stepping under relaxed atomics. The true discriminant is *scheduling purity*: whether the next iteration's work-set selection depends on per-vertex signals corrupted by intra-iteration races. This yields a formal theorem (Theorem 5.1) with four operational conditions that precisely predict which algorithms tolerate relaxed atomics.

Combined, the unified conclusion is:

> **In the graph kernels studied, GPU floating-point arithmetic is bit-exact across vendors when computation order is fixed. All observed cross-vendor drift originates from non-deterministic atomic-operation scheduling.**

The following table summarizes the complete causal chain across both algorithms:

| Algorithm | Implementation | Atomic discipline | Result | Root cause |
|---|---|---|---|---|
| SSSP ($\Delta$-stepping) | Standard | Strict CAS | Byte-exact cross-vendor | $\min$ is order-independent |
| SSSP ($\Delta$-stepping) | Standard | Relaxed (no CAS) | Drift (5 unique), all UNSAT | Scheduling purity violated (C4) |
| SSSP (Bellman-Ford) | Standard | Relaxed (no CAS) | Byte-exact, all SAT | Scheduling purity satisfied |
| SSSP (Async push) | Iterate-to-fixedpoint | Relaxed (no CAS) | Drift (5 unique), all UNSAT | Scheduling purity violated (C4) |
| PageRank | Push (`atomicAdd`) | Standard | Drift (24--100% vertices) | Non-deterministic sum scheduling |
| PageRank | Pull v1 (partial) | No scatter atomic | Drift (5/6 datasets) | Residual `atomicAdd` in reductions |
| PageRank | Pull v2 (zero atomic) | Fully deterministic | Byte-exact cross-vendor | All scheduling fixed |

This paper makes five contributions:

1. **Cross-vendor drift characterization** for GPU PageRank across 6 graphs, 2 precisions, and 3 GPUs from 2 vendors, quantifying the relationship between graph structure (degree skewness, dangling ratio) and drift magnitude (four orders of variation).

2. **Mechanism attribution** via three-stage kernel isolation that pins all observed drift to atomic-add scheduling, ruling out hardware FP differences, compiler variation, and microarchitectural effects for the studied kernels.

3. **Cross-vendor byte-identity evidence**: when atomic adds are eliminated, NVIDIA and AMD GPUs produce identical PageRank output (180 runs, zero bit differences), verified by direct bytewise comparison.

4. **Scheduling purity theorem** (Theorem 5.1): four operational conditions that precisely classify which SSSP algorithms tolerate relaxed atomics and which break, validated across three algorithms ($\Delta$-stepping, Bellman-Ford, asynchronous push) on both vendors.

5. **SSSP certificate verification**: a linear-time verifier checking five invariants on (distance, predecessor) certificates that accepts legitimate predecessor variation while detecting algorithmic errors with 99.94% sensitivity (3,420 error injections), plus cross-implementation validation against Gunrock and cuGraph (12/12 byte-identical distances).

---

## 2 Background

### 2.1 Floating-Point Non-Associativity

IEEE 754 [ieee754-2019] floating-point addition is commutative but not associative: $(a + b) + c$ may differ from $a + (b + c)$ when intermediate results are rounded to finite precision [shanmugavelu-fp-noassoc-2024]. For single-precision (FP32, 23-bit mantissa), the unit of least precision (ULP) at unit scale is $2^{-23} \approx 1.2 \times 10^{-7}$; for double-precision (FP64, 52-bit mantissa), $2^{-52} \approx 2.2 \times 10^{-16}$. When a parallel reduction sums $n$ values in tree order, the accumulated rounding error is $O(n \cdot \text{ULP})$ and depends on tree shape --- different execution orders produce different bit patterns even when mathematically equivalent.

The $\min$ operator, by contrast, is associative, commutative, and idempotent: $\min(a, \min(b, c)) = \min(\min(a, b), c) = \min(a, b, c)$ for any ordering, with no rounding error. This asymmetry is the algebraic root of the determinism boundary we characterize.

### 2.2 GPU Atomic Operations

GPU kernels use atomic operations to resolve concurrent writes to shared memory locations. Two primitives are relevant:

- **Atomic compare-and-swap (CAS)**: `atomicCAS(addr, expected, desired)` atomically reads `*addr`, compares with `expected`, and writes `desired` if equal. Used in SSSP to implement $d[v] \leftarrow \min(d[v], d[u] + w)$ via a CAS loop that retries until the minimum is installed. The final value is the global minimum of all attempted updates --- order-independent.

- **Atomic add**: `atomicAdd(addr, val)` atomically reads `*addr` and writes `*addr + val`. Used in PageRank to accumulate neighbor contributions: $\text{pr}[v] \mathrel{+}= \text{pr}[u] / \text{deg}(u)$. The final value is the sum of all contributions, but the *order* in which partial sums are accumulated affects the bit pattern due to FP non-associativity.

Both operations are non-deterministic in scheduling --- which thread's CAS or add executes first depends on warp/wavefront scheduling that varies across runs and architectures [alglave-gpu-2015]. The critical distinction is whether the final value depends on this scheduling order.

### 2.3 PageRank

PageRank computes the stationary distribution of a random walk on a directed graph $G = (V, E)$. At each iteration $t$:
$$\text{pr}^{(t+1)}[v] = \frac{1 - d}{N} + d \left( \sum_{u \in \text{in}(v)} \frac{\text{pr}^{(t)}[u]}{\text{deg}^+(u)} + \frac{D^{(t)}}{N} \right)$$
where $d = 0.85$ is the damping factor, $N = |V|$, and $D^{(t)} = \sum_{u : \text{deg}^+(u)=0} \text{pr}^{(t)}[u]$ is the dangling-vertex mass. Iteration continues until the $L_1$ residual $\|\text{pr}^{(t+1)} - \text{pr}^{(t)}\|_1 < \epsilon$ or a maximum iteration count is reached.

The GPU push-based implementation parallelizes the inner sum via atomic add: each thread reads $\text{pr}[u]$, divides by $\text{deg}^+(u)$, and atomically adds the contribution to every out-neighbor $v$. The dangling mass $D^{(t)}$ and convergence residual $\|\Delta\text{pr}\|_1$ are themselves parallel reductions, typically implemented via shared-memory block reduction followed by a global atomic add.

### 2.4 SSSP via $\Delta$-Stepping

$\Delta$-stepping [meyer-sanders-2003] partitions vertices into distance buckets of width $\Delta$ and processes them in order. Within each bucket, *light* edges ($w \leq \Delta$) are relaxed iteratively until stable; then *heavy* edges ($w > \Delta$) are relaxed once. Each relaxation attempts $d[v] \leftarrow \min(d[v], d[u] + w)$ via atomic CAS on a packed (distance, predecessor) tuple. The $\min$ over concurrent CAS attempts produces the globally minimum distance regardless of scheduling order.

### 2.5 Bellman-Ford and Asynchronous Push

**Bellman-Ford** iterates over all vertices unconditionally: each iteration scans every reachable vertex's outgoing edges and attempts relaxation. The host loop terminates when no thread reports an improvement (`any_updated == 0`). This *iterate-every-vertex* structure means the work-set is not conditioned on any per-vertex state.

**Asynchronous push** uses a double-buffered active-flag array: a vertex is active in iteration $T+1$ if its distance was improved (via atomic CAS) in iteration $T$. The host loop terminates when no vertex is active. Unlike Bellman-Ford, the work-set is derived from per-vertex improvement signals.

Both algorithms share the iterate-to-fixedpoint host-loop structure with $\Delta$-stepping but differ in how they select the active vertex set --- a distinction that becomes critical under relaxed atomics (Section 5).

---

## 3 Algebraic Framework

We classify GPU graph algorithms by the *outer operator* of their computation's semiring structure.

**Definition 3.1 (Reduction class).** A graph algorithm's per-vertex update has the form $x[v] \leftarrow \bigoplus_{u \in N(v)} f(x[u], w(u,v))$ where $\bigoplus$ is the outer reduction operator and $f$ is a vertex-local transformation. We call $\bigoplus$ the algorithm's *reduction operator*.

**Property 3.1 (Order-independence).** A reduction operator $\bigoplus$ is *order-independent under IEEE 754* if, for any multiset $\{a_1, \ldots, a_n\}$ of IEEE 754 values and any two permutations $\sigma, \tau$ of $\{1, \ldots, n\}$:
$$(\cdots((a_{\sigma(1)} \oplus a_{\sigma(2)}) \oplus a_{\sigma(3)}) \oplus \cdots) = (\cdots((a_{\tau(1)} \oplus a_{\tau(2)}) \oplus a_{\tau(3)}) \oplus \cdots)$$

**Observation 3.1.** The IEEE 754 $\min$ operator is order-independent (it is associative, commutative, and idempotent) *under the following preconditions*: (i) all operands are non-NaN (IEEE 754 `minNum` propagates NaN asymmetrically: $\min(\text{NaN}, x) = x$, but chained `minNum` with two NaN inputs is implementation-defined in some language bindings); (ii) signed-zero comparison follows a canonical rule (our implementations use `a < b ? a : b`, which preserves the first operand on ties, including $\min(+0, -0) = +0$ and $\min(-0, +0) = -0$ --- but the tie-breaking is deterministic for a fixed evaluation order, and the final minimum over a multiset is independent of permutation since the true minimum is unique when any non-zero value participates). In our SSSP kernels, distances are initialized to $+\infty$ (positive sentinel) and edge weights are strictly positive, so all intermediate candidates are finite, positive, and non-NaN; the preconditions are satisfied by construction. The IEEE 754 $+$ operator is *not* order-independent (it is commutative but not associative).

**Observation 3.2 (Determinism classification).** Let $A$ be a GPU graph algorithm whose per-vertex update uses reduction operator $\bigoplus$ implemented via atomic operations, where all operands satisfy the preconditions of Observation 3.1 (finite, non-NaN for $\min$; no subnormal flush-to-zero differences across platforms for $+$). If $\bigoplus$ is order-independent under IEEE 754, then $A$ produces deterministic output regardless of atomic scheduling order. If $\bigoplus$ is not order-independent, then $A$'s output may vary across executions unless the scheduling order is explicitly fixed.

*Proof.* When $\bigoplus$ is order-independent, the final value at each vertex is a function of the *multiset* of contributed values, independent of the order in which atomic operations install them. Atomic scheduling permutes the installation order but not the multiset, so the result is invariant. When $\bigoplus$ is not order-independent, there exist multisets and two permutations that produce distinct accumulated values due to intermediate rounding; non-deterministic scheduling may realize either permutation. $\square$

**FP environment assumptions.** Throughout this paper, all GPU kernels compile and execute under default IEEE 754 rounding mode (round-to-nearest-even). We verified via PTX and AMDGCN inspection that no fused multiply-add (FMA) instructions appear in the critical reduction paths of either algorithm (Section 8.4). Subnormal handling is IEEE 754-compliant on all tested GPUs (NVIDIA Turing/Ampere and AMD CDNA3 do not flush subnormals to zero in FP32/FP64 arithmetic by default). Platforms that enable flush-to-zero (FTZ) or denormals-are-zero (DAZ) modes may break the order-independence of $\min$ near zero and the bit-exactness of $+$ on subnormal operands.

This observation classifies algorithms into two regimes but says nothing about what happens *within* the order-independent regime when atomic discipline is relaxed (CAS replaced by non-atomic stores). That finer question --- which algorithms tolerate relaxed atomics --- is addressed by the scheduling purity theorem in Section 5.

**Predictions.** The framework classifies untested algorithms:

| Algorithm | Outer operator | Predicted class |
|---|---|---|
| BFS (unweighted SSSP) | $\min$ over integers | Reproducible (exact integer arithmetic) |
| Betweenness centrality | $\sum$ (dependency accumulation) | Drift (same mechanism as PageRank) |
| Connected components | $\min$ (label propagation) | Reproducible |
| Triangle counting | $\sum$ (count accumulation) | Reproducible (integer atomic add is exact) |
| GNN aggregation (sum/mean) | $\sum$ | Drift |
| GNN aggregation (max) | $\max$ | Reproducible |

The framework also predicts that algorithms in the drift class can be made reproducible by replacing non-deterministic atomic reduction with deterministic alternatives (e.g., fixed reduction tree, sequential host-side accumulation), at a potential performance cost. We validate this prediction empirically for PageRank in Section 7.

---

## 4 Experimental Methodology

### 4.1 Hardware

| GPU | Vendor | Architecture | Compiler | VRAM |
|---|---|---|---|---|
| NVIDIA A10 | NVIDIA | Ampere (sm_86) | nvcc / CUDA 12.8 | 24 GB |
| Tesla T4 | NVIDIA | Turing (sm_75) | nvcc / CUDA 12.8 | 16 GB |
| MI300X VF | AMD | CDNA3 (gfx942) | amdclang++ / ROCm 7.2 | 24 GB (1/8 slice) |

The MI300X VF is a 1/8 virtual-function slice of a full MI300X (192 GB). Drift measurements are unaffected by virtualization (drift depends on atomic scheduling order, not throughput), but performance numbers are not representative of full MI300X. We validated that a second MI300X VF with SPX 1/1 partition allocation (191.7 GiB) produces byte-identical strict-mode `d_hash` on every overlapping cell, confirming that VF partition size does not affect algorithmic semantics. Bare-metal scheduling behavior on a non-virtualized MI300X may differ.

All three GPUs share a single source tree compiled via HIP-to-CUDA compatibility shim on NVIDIA and native HIP on AMD. The same binary CSR input files are used across all platforms (SHA-256 verified).

### 4.2 Datasets

| Dataset | $|V|$ | $|E|$ | Type | $d_{\max}/d_{\text{med}}$ | Dangling ratio |
|---|---|---|---|---|---|
| road-CA | 1.97M | 5.53M | Road network | 4 | 0% |
| web-Google | 916K | 5.11M | Web graph | ~3,000 | 28.3% |
| LiveJournal | 4.85M | 69.0M | Social network | ~2,500 | 12.1% |
| wiki-Talk | 2.39M | 5.02M | Communication | ~50,000 | 93.8% |
| as-Skitter | 1.70M | 11.1M | Internet topology | ~12,000 | 5.2% |
| RMAT-22 | 4.19M | 67.1M | Synthetic (Graph500) | ~50,000 | 3.8% |

Datasets span four structural classes. We report $d_{\max}/d_{\text{med}}$ (ratio of maximum to median out-degree) as a quantitative measure of degree skewness. As Section 6 shows, drift magnitude correlates with degree skewness for most datasets, but wiki-Talk is an exception: its extreme drift is driven by its 93.8% dangling ratio rather than degree concentration. The dangling ratio (fraction of vertices with zero out-degree) is also the primary predictor of Pull v1's residual drift (Section 7).

### 4.3 Metrics

- **byte_diff_fraction**: fraction of vertices whose output bytes differ between two runs.
- **max $L_\infty$**: maximum element-wise absolute difference across all vertices.
- **CRC32**: checksum of the full output vector, reported as a compact presentation shorthand for tabular compactness. Byte-identity is verified by direct element-wise comparison (equivalent to `memcmp` over the full output array); CRC32 agreement alone is not used as proof of byte-identity. All CRC32 matches reported in this paper correspond to verified byte-identical vectors.
- **d_hash**: CRC32 of the reachable-distance subvector (SSSP); excludes unreachable sentinels. Same verification protocol: byte-identity confirmed by direct comparison, CRC32 reported for presentation.

### 4.4 Experimental Design

Each experiment runs 5 seeds per (dataset, precision, GPU) cell. Pairwise comparison produces:
- **Intra-vendor pairs**: $\binom{5}{2} = 10$ pairs per (dataset, precision, GPU).
- **Cross-vendor pairs**: $5 \times 5 = 25$ pairs per (dataset, precision, vendor pair).

PageRank uses three kernel variants: push (atomic-add scatter), pull v1 (no scatter atomic, residual reduction atomics), pull v2 (zero atomic adds). SSSP uses three algorithms ($\Delta$-stepping, Bellman-Ford, asynchronous push) under both strict atomic CAS and a controlled `RELAX_ATOMICS` build that swaps CAS for non-atomic store.

---

## 5 SSSP: Reproducible Under Atomic Discipline

### 5.1 Cross-Vendor Distance Byte-Identity

GPU $\Delta$-stepping produces byte-identical reachable-distance vectors across NVIDIA A10, Tesla T4, and AMD MI300X on all 6 datasets at both FP32 and FP64 precision, across all 5 seeds --- 60 (dataset, precision, seed) cells, all three GPUs producing identical `d_hash` in every cell. The same `d_hash` (CRC32 of the reachable-distance subvector, confirmed by direct bytewise comparison) across all three GPUs from both vendors confirms that the $\min$-based relaxation produces order-independent results. All three algorithms ($\Delta$-stepping, Bellman-Ford, asynchronous push) produce identical `d_hash` per dataset under strict atomics on all three platforms, confirming cross-algorithm and cross-vendor distance-vector equality.

### 5.2 Predecessor Divergence Is Structural

While distances are byte-exact, predecessor arrays $\pi$ diverge on 5 of 17 cross-vendor configurations --- all involving FP32 road-network inputs where near-ties ($d[u] + w \approx d[v]$) produce multiple equally-valid shortest-path trees. Per-vertex classification on the two analyzed configurations (6,094 differing-$\pi$ vertices total) shows 100% of differing-$\pi$ vertices have at least two incoming edges achieving the FP-equal minimum candidate. This is expected: under FP weights, $\pi$ is race-determined (the first thread to install a tied distance wins), and the classic SSSP correctness condition requires only that $(d, \pi)$ satisfies edge-inequality invariants, not that $\pi$ be unique.

### 5.3 Breaking Reproducibility: The Relaxed-Atomics Probe

A controlled experiment (`RELAX_ATOMICS=ON`) replaces the per-vertex atomic CAS with a non-atomic load-compare-store, preserving the early-out check but removing atomicity. We apply this probe to three algorithms that span the design space:

| Algorithm | Host loop | Work-set selection | Strict | Relaxed | (C4) status |
|---|---|---|---|---|---|
| Bellman-Ford | iterate-to-fixedpoint | every reachable vertex every iteration | SAT, 1 unique | SAT, 1 unique | satisfied |
| $\Delta$-stepping | bucket phases | bucket membership from `d[v]` | SAT, 1 unique | UNSAT, 5 unique | **violated** |
| Async push | iterate-to-fixedpoint | active flag from racy `outcome` | SAT, 1 unique | UNSAT, 5 unique | **violated** |

Each cell represents 6 datasets $\times$ 5 seeds = 30 runs. "1 unique" means all seeds produce the same `d_hash`; "5 unique" means every seed produces a distinct `d_hash`. Results are cross-vendor symmetric (NVIDIA and AMD produce the same partition).

The results reveal a three-way surprise:

1. **$\Delta$-stepping breaks immediately** under relaxed atomics: 30/30 UNSAT, 5 unique `d_hash` per dataset. The non-atomic store allows concurrent threads to overwrite each other's distance improvements, corrupting the bucket-membership predicate.

2. **Bellman-Ford remains byte-exact and SAT** under the same relaxed primitive: 30/30 SAT, 1 unique `d_hash` per dataset. Bellman-Ford scans every reachable vertex every iteration regardless of per-vertex state, so racy stores are repaired by the next full sweep.

3. **Asynchronous push breaks identically to $\Delta$-stepping** despite sharing Bellman-Ford's iterate-to-fixedpoint host loop: 30/30 UNSAT, 5 unique per dataset. This refutes the naive hypothesis that "iterate-to-fixedpoint $\Rightarrow$ race-tolerant." The true discriminant is not the host-loop structure but the work-set selection mechanism.

Both $\Delta$-stepping and async push prune the active vertex set based on per-vertex signals (bucket membership derived from a possibly-stale `d[v]`; active flag set by the racy `atomicRelax` outcome). Under relaxed atomics, these signals become unreliable: a racy store can overwrite a thread's improvement, but only the originally-committing thread sets the active flag. The vertex whose stale distance ends up in memory may not be re-armed, losing correctness signals. Bellman-Ford doesn't prune, so it tolerates the noise.

### 5.4 Scheduling Purity: Theorem Statement

We formalize the discriminating condition as four operational requirements:

**Theorem 5.1 (Race tolerance under scheduling purity).** Let $\mathcal{A}$ be an iterate-to-fixedpoint host loop on a graph $G = (V, E, w)$ with non-negative edge weights and finite (non-NaN, non-$-\infty$) initial distances, equipped with the relaxation kernel. Suppose the execution environment satisfies: IEEE 754 round-to-nearest-even mode, no flush-to-zero, and single-copy atomicity for naturally-aligned 64-bit global stores (see C1). Suppose further that conditions (C1)--(C4) below hold. If at iteration $T$ the host observes `any_updated == 0` and proceeds to terminate, then the host's observed distance vector $d$ equals the optimum $d^\star$.

**Conditions.**

- **(C1) No-tearing per-vertex store.** Each per-vertex packed `(d, π)` (or scalar `d`) load and store executes as a single-copy whole-word memory operation at a naturally-aligned address. On the GPUs we test, the 64-bit packed store at `-O2` lowers to `st.global.u64` (NVIDIA PTX) / `global_store_b64` (AMD AMDGCN). The NVIDIA PTX ISA guarantees single-copy atomicity for naturally-aligned stores up to 64 bits [lustig-ptx-2019]. For AMD CDNA3, we treat naturally-aligned 64-bit global stores as an operational assumption, verified by inspecting the emitted `global_store_b64` instruction; no tearing was observed across any of our experimental runs. This condition is mechanically verifiable by inspecting the compiler's ISA output --- it does not require trusting the compiler's correctness in general, only confirming that the emitted instruction is on the vendor's single-copy-atomic list.

- **(C2) Terminal-iteration edge coverage.** The kernel of the terminating iteration $T$ inspects every edge $(u, v)$ with $d[u] < +\infty$ at $T$'s start. Bellman-Ford satisfies (C2) by construction; algorithms that prune the work set per iteration must additionally satisfy (C4).

- **(C3) Strict progress flag.** Any thread that executes past the kernel's early-out check atomically sets a global progress flag (`atomicOr(any_updated, 1)`) before its non-atomic store. The host reads this flag after a release-acquire barrier (`cudaDeviceSynchronize` / `hipDeviceSynchronize`) and uses `flag == 0` as the termination criterion.

- **(C4) Scheduling purity.** The kernel's choice of "active vertex set for iteration $T+1$" depends only on values that are release-acquire visible to all threads at the iteration boundary --- not on per-vertex committed-improvement signals computed under relaxed-atomic stores within iteration $T$. Equivalently: for every vertex $v$, the predicate "$v \in \mathrm{active}_{T+1}$" must be derivable from the host's view of memory at the post-kernel barrier, not from intra-iteration racy local signals.

*Proof sketch.* The argument proceeds in two parts.

*Convergence under (C1)--(C3).* Under (C1), no vertex observes a torn value --- every load returns a well-formed distance written by some single thread. Under (C2), the terminal iteration inspects every finite-distance edge; if any edge violates the optimality condition $d[v] \le d[u] + w(u,v)$, the responsible thread would attempt an improvement and set the progress flag via (C3), contradicting the host's observation that `any_updated == 0`. Therefore at termination, $d$ satisfies $d[v] \le d[u] + w(u,v)$ for all edges with $d[u] < +\infty$. Combined with $d[s] = 0$ and non-negative weights, this is precisely the Bellman optimality condition [bellman-1958]: $d = d^\star$.

*Role of (C4).* Without (C4), an algorithm may fail to satisfy (C2) even though (C3) reports no progress. If the work-set selection depends on per-vertex signals computed from racy intra-iteration stores, a vertex $v$ whose distance was overwritten by a stale value may be incorrectly marked inactive. When $v$ is omitted from the next iteration's work set, edges from $v$ are not inspected, and the thread that would have detected the optimality violation is never launched --- so (C3) reports quiescence even though $d \neq d^\star$. (C4) prevents this by requiring the active-set predicate to depend only on values visible at the iteration barrier, ensuring that racy intra-iteration stores cannot cause premature pruning.

The full formal proof is in the supplementary material. $\square$

**Condition diagnosis for each algorithm:**

| Condition | Bellman-Ford | $\Delta$-stepping | Async push |
|---|---|---|---|
| (C1) No-tearing | ✓ (64-bit store) | ✓ (64-bit store) | ✓ (64-bit store) |
| (C2) Edge coverage | ✓ (every vertex scanned) | ✓ under strict; ✗ under relaxed (stale bucket) | ✓ under strict; ✗ under relaxed (stale active flag) |
| (C3) Progress flag | ✓ (`atomicOr`) | ✓ (`atomicOr`) | ✓ (`atomicOr`) |
| (C4) Scheduling purity | ✓ (no pruning) | **✗** (bucket from `d[v]`) | **✗** (active from racy outcome) |

Bellman-Ford satisfies all four conditions --- it tolerates relaxed atomics. $\Delta$-stepping and async push violate (C4) --- their work-set depends on per-vertex signals corrupted by intra-iteration races under relaxed stores. This matches the empirical results exactly.

### 5.5 Certificate-Based Verification

We ship a linear-time verifier [mehlhorn-mcconnell-naher-schweitzer-2011] that checks five invariants on the (distance, predecessor) certificate:

- **(R1)** Source constraint: $d[s] = 0$ and $\pi[s] = \bot$.
- **(R2)** Predecessor-distance consistency: $|d[v] - (d[\pi[v]] + w(\pi[v], v))| \le 4096\epsilon \cdot \max(|d[v]|, 1)$ for every reachable $v$.
- **(R3)** Relaxation inequality: $d[v] \le d[u] + w(u, v)$ for every edge, with the same FP tolerance.
- **(R4)** Acyclicity: the predecessor relation forms an acyclic forest rooted at $s$ on the reachable set.
- **(R5)** Unreachable consistency: every $v$ with $\pi[v] = \bot$ and $v \neq s$ has $d[v] = \infty$.

The verifier returns SAT if all rules pass, otherwise UNSAT\_$X$ where $X$ identifies the failing rule. Under exact arithmetic, (R1)--(R5) constitute a complete Bellman-optimality check [bellman-1958]: (R3) gives the upper bound, (R2)+(R4) exhibit a witnessing path, and the argument follows the standard soundness theorem [alkassar-bohme-mehlhorn-rizkallah-2014].

**FP tolerance regime.** The $4096\epsilon$ tolerance on (R2) and (R3) trades soundness for completeness along the diameter axis. On long-diameter graphs with small FP32 weights (e.g., road networks with gaussian weight distributions), path-accumulated rounding can exceed the tolerance, producing UNSAT on a valid certificate. We observe this boundary on `ny_road` and `usa_road` under FP32 with both uniform and gaussian weight distributions; the same graphs under FP64 produce SAT. The four UNSAT cells (2 graphs $\times$ 2 weight distributions) are identical across all three GPUs --- the verifier rejects the same configurations on A10, T4, and MI300X, confirming that the boundary is a property of the graph diameter $\times$ precision interaction, not of the hardware. This characterizes the verifier's operating envelope --- a finding, not a defect. A diameter-aware tolerance proportional to $\text{diameter} \cdot \min(w) \cdot \epsilon$ would tighten one regime at the cost of widening the other.

**Verifier vs. golden-output comparison.** Across the full artifact re-run (4,223 runs on three GPUs), the verifier returns 4,211 SAT and 12 UNSAT --- the 12 being the same 4 road-FP32 boundary cells on each of the 3 GPUs. The verifier accepts all strict-CAS configurations outside this boundary (including the 5 with divergent $\pi$) and rejects all 60 relaxed-atomic configurations. A naive cross-vendor golden-output validator that computes `array_equal(π_NV, π_AMD)` produces 5 false rejections on the divergent-$\pi$ configurations. The verifier's structural acceptance of multiple valid trees is not a tolerance artifact but a consequence of the SSSP solution space on FP-rounded weights.

**Error injection coverage.** We inject five kinds of errors into verified-SAT certificates: distance perturbation ($d[v] \times (1 + \delta)$, $\delta \in [-0.10, 0.10]$), random predecessor replacement, simultaneous (d, π) perturbation, missed-unreachable injection (replacing sentinel with plausible finite distance), and 2-cycle installation in the predecessor relation. Across 3,720 injection cases (4 datasets, 5 error kinds, 30--100 seeds per cell), 300 are graph-property exceptions (missed-unreachable on fully-connected road graphs with no unreachable vertices to corrupt). Of the remaining **3,420 signal-bearing cases, the verifier detects 3,418 (99.94%)**. The two missed cases are sub-$\epsilon$ distance perturbations on `ny_road` FP32 where the perturbation falls within the $4096\epsilon$ tolerance band --- the documented soundness-completeness trade-off.

Cycle and predecessor-random injections are pure verifier-only signal: byte-comparison on `d` sees zero difference (the injection did not touch `d`), but the verifier rejects. In this regime, certificate verification is strictly more sensitive than golden-output comparison.

**Stress test.** RMAT-18 stress tests produce SAT on every run across all three platforms: 1,000/1,000 on A10, 1,000/1,000 on T4, and 1,965/1,965 on MI300X VF --- 3,965 total, zero UNSAT, zero false-positive rejection.

### 5.6 Cross-Implementation Audit

To test whether byte-exact distances are specific to our implementation or structural to disciplined GPU SSSP, we apply our verifier to two independently-developed CUDA SSSP libraries: **Gunrock 2.2.0** [wang-gunrock-2016] (UC Davis, asynchronous frontier-based) and **NVIDIA RAPIDS cuGraph 25.10** [nvidia-cugraph] (Python-fronted, cuDF-backed). Both libraries are independent of our HIP code and of each other in team, host orchestration, and kernel design.

**Result: 12/12 byte-identical distances** across road, web, social, and RMAT graphs spanning 264K to 24M vertices. Per-dataset `d_hash` matches our SAT-anchored certificates exactly. The byte-exact distance property is cross-implementation across at least two independent CUDA libraries: the claim --- disciplined atomic CAS plus min-plus semiring yields byte-exact distances under non-negative weights --- is structural to the algorithm class, not to the code we wrote.

---

## 6 PageRank: Drift Characterization

### 6.1 Cross-Vendor Drift Exists and Is Universal

Every cross-GPU PageRank push comparison shows non-zero byte difference. Across 900 cross-GPU pairs (6 datasets $\times$ 2 precisions $\times$ 75 pairs per cell: 25 A10$\leftrightarrow$MI300X, 25 T4$\leftrightarrow$MI300X, 25 A10$\leftrightarrow$T4), the minimum byte_diff_fraction is 24.1% (road-CA FP32) and the maximum is 100.0% (wiki-Talk FP32). There are no byte-identical cross-GPU pairs under the push kernel --- including A10$\leftrightarrow$T4 pairs (same vendor, different microarchitecture), confirming that scheduling differences within NVIDIA's own product line produce drift comparable to cross-vendor drift.

### 6.2 Two Structural Drivers of Drift

Drift magnitude is governed by two structural properties that act through different mechanisms:

**Degree skewness drives per-vertex drift magnitude.** Hub vertices in power-law graphs receive many concurrent atomic-add contributions, maximizing the reduction-order variance at each destination. We observe four orders of magnitude variation in max $L_\infty$ across the degree-skewness axis:

| Dataset | $d_{\max}/d_{\text{med}}$ | Dangling ratio | FP32 max $L_\infty$ | FP64 max $L_\infty$ |
|---|---|---|---|---|
| road-CA | 4 | 0% | $3.41 \times 10^{-13}$ | $7.41 \times 10^{-22}$ |
| LiveJournal | ~2,500 | 12.1% | $6.26 \times 10^{-10}$ | $5.15 \times 10^{-19}$ |
| web-Google | ~3,000 | 28.3% | $2.39 \times 10^{-9}$ | $1.64 \times 10^{-17}$ |
| as-Skitter | ~12,000 | 5.2% | $1.19 \times 10^{-9}$ | $1.36 \times 10^{-18}$ |
| RMAT-22 | ~50,000 | 3.8% | $4.13 \times 10^{-9}$ | $1.74 \times 10^{-17}$ |
| wiki-Talk | ~50,000 | 93.8% | $1.60 \times 10^{-8}$ | $1.22 \times 10^{-18}$ |

Road-CA ($d_{\max}/d_{\text{med}} = 4$, near-uniform degree) has 300--4000$\times$ smaller max $L_\infty$ than high-skewness graphs.

**Dangling ratio drives byte_diff_fraction saturation.** wiki-Talk has the highest FP32 max $L_\infty$ despite not having the highest degree skewness. This anomaly is explained by its 93.8% dangling ratio: most PageRank mass concentrates on a small active set, while the 93.8% dangling majority sit near the FP32 noise floor ($\sim 10^{-7}$). Any perturbation from the non-deterministic dangling-mass sum pushes these vertices across ULP boundaries, producing 100% byte_diff_fraction at FP32 --- every vertex's representation differs. At FP64, byte_diff drops to 99.1% median (individual pairs as low as 9.5%) because the wider mantissa lifts some vertex values above the noise floor, recovering partial byte-identity.

The two effects are separable: degree skewness controls the *magnitude* of per-vertex drift ($L_\infty$), while dangling ratio controls the *extent* of drift propagation (byte_diff_fraction). Road-CA (0% dangling, low skewness) has both the smallest $L_\infty$ and the smallest byte_diff_fraction.

### 6.3 Precision Scaling and Intra-Vendor Variance

**Precision scaling.** FP64 drift magnitude drops approximately $10^9\times$ relative to FP32, consistent with the ULP ratio ($2^{-23} / 2^{-52} \approx 5.4 \times 10^8$). However, byte_diff_fraction remains comparable (e.g., web-Google: 99.4% FP32 vs. 94.3% FP64) because FP64's wider mantissa means even tiny perturbations flip many low-order bits. This has a direct implication for tolerance-based validation: a hand-tuned $\varepsilon = 10^{-6}$ that works for FP32 is eight orders of magnitude too loose for FP64, where the relevant drift scale is $10^{-18}$. Principled tolerance must be parameterized by machine epsilon, not hand-tuned.

**Intra-vendor variance.** Same-GPU, different-seed pairwise comparison reveals scheduler-dependent variance across all three GPUs:

| Dataset | A10 nv-nv byte_diff median | T4 nv-nv byte_diff median | MI300X amd-amd byte_diff median |
|---|---|---|---|
| road-CA | 20.8% | 16.3% | 15.2% |
| web-Google | 77.0% | 11.3% | **96.8%** |
| LiveJournal | 29.3% | 50.1% | **55.3%** |
| as-Skitter | 87.2% | 35.4% | **97.9%** |
| RMAT-22 | 32.8% | 71.3% | 44.2% |
| wiki-Talk | 100.0% | 97.1% | 100.0% |

CRC32 analysis confirms that all 10 intra-GPU seed pairs on every dataset produce distinct output on all three GPUs (no two seeds ever produce byte-identical push output). The MI300X VF wavefront scheduler produces higher intra-vendor drift than either NVIDIA GPU on 3 of 6 datasets (bolded), consistent across precisions. T4 (Turing) and A10 (Ampere) show different intra-vendor profiles despite being same-vendor --- confirming that intra-vendor variance is a scheduling-hardware effect, not a floating-point precision effect. Note that the MI300X VF results reflect a virtualized 1/8 partition; bare-metal scheduling behavior on a full MI300X may differ.

---

## 7 Mechanism Attribution: Isolating the Drift Source

### 7.1 The Onion-Peeling Methodology

We implement three PageRank kernel variants that progressively eliminate atomic adds:

| Variant | Scatter loop | Dangling-mass sum | Convergence residual |
|---|---|---|---|
| Push (baseline) | `atomicAdd` | `atomicAdd` | `atomicAdd` |
| Pull v1 | No atomic (CSC read) | `atomicAdd` | `atomicAdd` |
| Pull v2 | No atomic (CSC read) | Host-side sequential | Host-side sequential |

Pull v1 reads each vertex's in-neighbors via the CSC transpose and writes to its own slot --- no scatter atomic add. Pull v2 additionally replaces the block-level `atomicAdd` in the dangling-mass and convergence-residual kernels with per-block partial sums written to a device buffer, copied to host, and summed sequentially. Pull v2 contains **zero** atomic-add operations.

### 7.2 Pull v1: Partial Isolation (Still Drifts)

Pull v1 achieves byte-identity on road-CA (the only dataset with zero dangling vertices) but still drifts on 5/6 datasets. The residual drift source is the `atomicAdd` in `pr_dangling_sum`: each block's shared-memory reduction produces a partial sum that is atomically added to a global accumulator. The dangling mass feeds into the base term $(1-d)/N + d \cdot D/N$ that is added to *every* vertex's rank each iteration --- a non-deterministic dangling sum propagates drift to the entire rank vector.

| Dataset | Dangling ratio | Pull v1 byte-identical? |
|---|---|---|
| road-CA | 0% | ✓ (natural control) |
| RMAT-22 | 3.8% | ✗ |
| as-Skitter | 5.2% | ✗ |
| LiveJournal | 12.1% | ✗ |
| web-Google | 28.3% | ✗ |
| wiki-Talk | 93.8% | ✗ |

Any non-zero dangling ratio introduces a non-deterministic `atomicAdd` in the dangling-mass reduction. Road-CA (0% dangling) is the only dataset where Pull v1 achieves byte-identity --- a natural control confirming the diagnosis.

### 7.3 Pull v2: Complete Isolation (Byte-Identical Cross-Vendor)

Pull v2 eliminates all atomic adds. Result:

| Dataset | FP32 CRC32 (A10 = T4 = MI300X) | FP64 CRC32 (A10 = T4 = MI300X) |
|---|---|---|
| as-Skitter | `dcb27f17` | `da5b9150` |
| LiveJournal | `dfeef6b0` | `9ee07b86` |
| RMAT-22 | `eef3b85c` | `611a5829` |
| road-CA | `ce22664c` | `64b45a2a` |
| web-Google | `91b78c48` | `e6f00d32` |
| wiki-Talk | `27b3fe0c` | `7814d28a` |

All 180 runs (3 GPUs $\times$ 6 datasets $\times$ 2 precisions $\times$ 5 seeds) produce identical CRC32 per (dataset, precision) cell, confirmed by direct bytewise comparison of the full output vectors. Convergence metrics also match exactly: identical iteration counts and final $L_1$ residuals across all three GPUs.

### 7.4 Implications of Cross-Vendor Byte-Identity

The Pull v2 result eliminates three candidate drift sources:

1. **Hardware FP implementation differences**: if NVIDIA and AMD FP units produced different rounding for the same operation sequence, Pull v2 would show vendor-dependent CRCs. It does not.
2. **Compiler differences**: nvcc and amdclang++ produce different machine code, but the computation is bit-identical. IEEE 754 compliance is exact at the instruction level.
3. **Microarchitectural variation**: Turing (sm_75), Ampere (sm_86), and CDNA3 (gfx942) represent three distinct microarchitectures across two ISAs [alglave-gpu-2015]. Byte-identity across all three proves that the hardware arithmetic is ISA-invariant for the operations used.

The complete causal chain:

| Layer removed | Variant | Result | Conclusion |
|---|---|---|---|
| None | Push | Drift | Non-determinism present |
| Scatter `atomicAdd` | Pull v1 | Still drifts (5/6) | Scatter is not the only source |
| All `atomicAdd` | Pull v2 | Byte-identical cross-vendor | `atomicAdd` scheduling is the sole observed source |

### 7.5 Cost of Determinism

Pull v2 replaces device-side atomic reduction with host-side sequential summation, incurring a device-to-host copy and sequential sum per kernel invocation. We measure wall time (ms/iter) for the three variants across all datasets on A10 and T4 (MI300X VF performance excluded from this table as it is not representative of full MI300X; see Section 4.1):

| GPU | Dataset | Push | Pull v1 | Pull v2 | v2/v1 |
|---|---|---|---|---|---|
| A10 | road-CA | 0.24 | 0.21 | 0.21 | 1.01$\times$ |
| A10 | web-Google | 0.30 | 3.30 | 3.31 | 1.00$\times$ |
| A10 | as-Skitter | 3.53 | 1.99 | 1.97 | 0.99$\times$ |
| A10 | LiveJournal | 8.49 | 16.03 | 16.07 | 1.00$\times$ |
| A10 | wiki-Talk | 5.90 | 1.53 | 1.53 | 1.00$\times$ |
| A10 | RMAT-22 | 12.09 | 78.85 | 78.74 | 1.00$\times$ |
| T4 | road-CA | 0.41 | 0.47 | 0.48 | 1.01$\times$ |
| T4 | web-Google | 1.32 | 6.19 | 6.32 | 1.02$\times$ |
| T4 | as-Skitter | 6.23 | 2.19 | 3.35 | 1.53$\times$\* |
| T4 | LiveJournal | 30.19 | 31.74 | 31.84 | 1.00$\times$ |
| T4 | wiki-Talk | 7.36 | 1.93 | 1.93 | 1.00$\times$ |
| T4 | RMAT-22 | 26.14 | 86.92 | 84.84 | 0.98$\times$ |

\*T4 as-Skitter converges in only 13 iterations; the fixed per-iteration host-copy overhead (device$\to$host transfer + sequential sum) is proportionally larger. Averaged across 5 seeds, the overhead is 15% (mean v2/v1 = 1.15$\times$), with high run-to-run variance due to the short iteration count.

Pull v2 wall time is within 2% of Pull v1 on 11 of 12 A10/T4 dataset cells. The host-reduction cost (copying ~256 partial sums and summing sequentially) is negligible relative to the main SpMV-like gather kernel, which dominates each iteration. The single outlier (T4 as-Skitter) is explained by the unusually short convergence: 13 iterations magnify constant per-iteration overhead. On long-running workloads (the common case), the determinism overhead is consistently below measurement noise.

The dominant performance difference is between push and pull access patterns (CSR scatter vs. CSC gather), not between atomic and deterministic reduction. Push is faster on high-skewness graphs where scattered writes are amortized across many neighbors; pull is faster on low-degree datasets (road-CA, wiki-Talk) where sequential reads dominate. This architectural choice is independent of the determinism mechanism.

### 7.6 Software-Stack Drift: CUDA 12.8 vs. 13.0

As a supplementary probe, we ran the push kernel under CUDA 12.8 and CUDA 13.0 on the same hardware with the same inputs, on both A10 and T4. Result: all 24 comparison cells (6 datasets $\times$ 2 precisions $\times$ 2 GPUs) show different CRC32 values between CUDA versions --- neither GPU reproduces its CUDA 12.8 output under CUDA 13.0. The atomic-add scheduling order differs between driver versions, extending the drift attribution from hardware scheduling to the full software stack. This further validates the atomic-scheduling explanation: any layer that changes scheduling (hardware, driver, compiler, runtime) can change the output.

---

## 8 Discussion

### 8.1 Unified Principle

The SSSP and PageRank results converge on a single principle: **In the graph kernels studied, GPU floating-point arithmetic is bit-exact across vendors when computation order is fixed; all observed cross-vendor drift originates from non-deterministic atomic-operation scheduling.** This principle has two faces:

- SSSP's atomic CAS implements $\min$, which is order-independent $\Rightarrow$ byte-exact by construction.
- PageRank's atomic add implements $\sum$, which is order-dependent $\Rightarrow$ drift under non-deterministic scheduling, but byte-exact when scheduling is determinized (Pull v2).

The principle is falsifiable: any algorithm where Pull-v2-style deterministic reduction still shows cross-vendor drift would refute it. We have not observed such a case. The scope of this claim is limited to the operations exercised in the studied kernels (addition, comparison, division on normal-range FP values); algorithms involving transcendental functions, FMA-eligible sequences (Section 8.4), or subnormal-range operands may exhibit additional cross-vendor divergence.

### 8.2 Two Boundaries

The paper characterizes two distinct boundaries:

1. **The determinism boundary** (Section 3, Observation 3.2): order-independent operators ($\min$) yield reproducible output; order-dependent operators ($\sum$) yield drift. This is the coarse-grained classification.

2. **The scheduling purity boundary** (Section 5, Theorem 5.1): within the order-independent class, algorithms that satisfy scheduling purity tolerate relaxed atomics; those that violate it break. This is the fine-grained classification within the reproducible regime.

Together, these two boundaries provide a practical taxonomy for the studied class of atomic-reduction graph kernels: an algorithm's determinism class is determined by its reduction operator, and its robustness to implementation shortcuts (relaxed atomics) is determined by its scheduling purity.

### 8.3 Predictions for Untested Algorithms

The algebraic framework (Section 3) predicts:
- **BFS**: exact (integer $\min$, no FP rounding).
- **Betweenness centrality**: drift (sum-reduction of dependency scores).
- **Connected components**: reproducible (label propagation via $\min$).
- **Graph neural network aggregation**: drift if using sum/mean pooling; reproducible if using max pooling.

The scheduling purity theorem (Section 5) further predicts: within the reproducible class, any algorithm that prunes its work set based on per-vertex racy signals will break under relaxed atomics, while algorithms with unconditional iteration (e.g., Jacobi-style sweeps) will tolerate them.

SSSP and PageRank instantiate the two core operator classes studied in this paper ($\min$ and $\sum$); validating the predictions on additional graph primitives is future work.

### 8.4 Limitations

1. **Three GPUs, two vendors.** Our cross-vendor finding rests on A10, T4, and MI300X VF. Additional architectures (e.g., Intel Arc, NVIDIA Hopper) would strengthen generalization.

2. **MI300X VF (1/8 slice).** Performance measurements on the MI300X VF are not representative of the full GPU. Drift measurements are unaffected (drift depends on atomic scheduling, not throughput). A second VF with SPX 1/1 partition produces byte-identical results, but bare-metal scheduling behavior may differ.

3. **Two graph primitives.** We test SSSP ($\min$) and PageRank ($\sum$). The framework's predictions for other algorithms (betweenness centrality, GNN aggregation, connected components) are untested. While the algebraic classification is general, the empirical validation covers only these two algorithm classes.

4. **FMA contraction.** IEEE 754 permits fused multiply-add (FMA) to produce differently-rounded results than separate multiply and add. We inspected the PTX and AMDGCN output for our kernels and confirmed no FMA instructions appear in the critical reduction paths (the inner loops use only independent additions). Algorithms with FMA-eligible inner loops (e.g., $a \cdot b + c$ sequences) may exhibit drift even under deterministic scheduling if FMA contraction differs across vendors.

5. **Verifier FP boundary.** The SSSP verifier flags certificates as UNSAT when accumulated FP rounding exceeds its tolerance --- observed on long-diameter graphs with small-weight distributions under FP32. This characterizes the verifier's operating envelope, not an algorithmic defect (Section 5.5).

6. **Scope of "bit-exact arithmetic" claim.** Our cross-vendor byte-identity result (Pull v2) exercises IEEE 754 addition, comparison, and division on normal-range FP32/FP64 values under round-to-nearest-even mode. We do not test transcendental functions, extended-precision intermediates, or subnormal/denormalized operands. Platforms with non-default FP modes (FTZ/DAZ) or different FMA contraction policies may break byte-identity even with fixed scheduling.

---

## 9 Related Work

**Hardware-enforced determinism.** Chou et al. [chou-dab-2020] propose Deterministic Atomic Buffering (DAB), a microarchitectural extension that orders atomic operations to achieve run-to-run determinism on a single GPU, evaluated on PageRank and betweenness centrality. Jooybar et al. [jooybar-gpudet-2013] propose GPUDet, a full-instruction-ordering scheme for deterministic GPU execution. Our work differs in three dimensions: (i) we characterize *cross-vendor* determinism across NVIDIA and AMD, not single-GPU run-to-run reproducibility; (ii) we achieve determinism via algorithmic choice (Pull v2) on commodity hardware without microarchitectural modification, at <2% incremental overhead relative to the non-deterministic pull baseline (Section 7.5); (iii) we identify the algebraic boundary ($\min$ vs. $\sum$) that predicts which algorithms need deterministic treatment at all --- min-plus algorithms require no intervention, a distinction absent from hardware-enforced approaches that order *all* atomics uniformly.

**GPU determinism in software frameworks.** NVIDIA classifies reductions as non-deterministic and offers `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` and CUB's deterministic scan as partial mitigations; the CCCL 3.x library adds deterministic-reduce primitives with documented overhead. PyTorch documents `torch.use_deterministic_algorithms()` with known gaps. Our work differs by attributing determinism to the algebraic structure of the computation rather than to framework-level flags, and by demonstrating that graph-specific reductions can be made deterministic at <2% incremental cost over the pull baseline, without general-purpose deterministic-reduction libraries.

**Reproducible floating-point.** ReproBLAS [riedy-demmel-reproblas] and its successors [demmel-nguyen-2013, iakymchuk-exblas-2015] guarantee bitwise reproducibility for BLAS operations via long accumulators or pre-rounded fixed-point intermediate representations, at 20--30% overhead. Our Pull v2 achieves reproducibility for graph reductions at <2% incremental overhead over the pull-gather baseline by exploiting the block-reduce + host-sum structure, though the pull formulation itself may be slower or faster than push depending on graph structure (Section 7.5).

**Certifying algorithms.** McConnell et al. [mehlhorn-mcconnell-naher-schweitzer-2011] formalize the certifying-algorithm paradigm: an algorithm emits a certificate that a simpler checker can verify. Alkassar et al. [alkassar-bohme-mehlhorn-rizkallah-2014] formally verify the LEDA shortest-path checker in Isabelle/Simpl. Our SSSP verifier follows this paradigm, extending it to GPU-parallel SSSP with FP-weighted graphs. The verifier's advantage over golden-output comparison is that it accepts legitimate predecessor variation while detecting invalid distance vectors.

**Cross-vendor GPU graph analytics.** Gunrock [wang-gunrock-2016], RAPIDS cuGraph [nvidia-cugraph], and Pannotia [che-pannotia-2013] provide GPU graph primitives; we verify that Gunrock and cuGraph $\Delta$-stepping produce byte-identical distances to our implementation on 6/6 datasets (Section 5.6). VanAusdal and Burtscher [vanausdal-burtscher-2025] compare graph algorithm implementation styles (including PageRank reduction strategies) across NVIDIA and AMD GPUs, focusing on throughput; their observation that global-atomic-add performance differs between vendors is consistent with our finding of higher intra-vendor scheduling variance on MI300X (Section 6.3). Flor et al. [flor-adgraph-2025] port nvGRAPH to AMD ROCm (adGRAPH) and benchmark performance parity. Both are performance studies; our contribution is the orthogonal axis of cross-vendor *correctness* and bit-reproducibility.

**Asynchronous iteration theory.** Bertsekas [bertsekas-1982, bertsekas-1983] establishes convergence conditions for asynchronous distributed computation of fixed points, including distributed Bellman-Ford. Our scheduling purity theorem (Theorem 5.1) can be viewed as the GPU-parallel analogue: condition (C4) identifies when the GPU's non-deterministic thread scheduling preserves the fixedpoint convergence guarantee.

**GPU memory models.** Alglave et al. [alglave-gpu-2015] and Lustig et al. [lustig-ptx-2019] formalize weak memory behaviors and programming assumptions on GPUs. Our condition (C1) relies on the PTX single-copy atomicity guarantee [lustig-ptx-2019] and operational verification of the same property on AMD CDNA3; (C4) relies on the release-acquire semantics of `cudaDeviceSynchronize` / `hipDeviceSynchronize` as the iteration barrier.

**Dataflow analysis.** The algebraic framework (Section 3) echoes Kildall's [kildall-1973] lattice-theoretic classification of program analyses and Cousot & Cousot's [cousot-cousot-1977] abstract interpretation framework. The $\min$/$\sum$ dichotomy maps to the meet-over-all-paths vs. accumulative-transfer distinction in dataflow analysis.

**Microarchitectural reproducibility.** Lindsay et al. [lindsay-counterpoint-2026] use hardware event counters to refute microarchitectural assumptions, demonstrating that empirical falsification is a viable methodology for architecture research. Our three-algorithm race-injection probe follows a similar falsification methodology applied to algorithmic rather than microarchitectural assumptions.

---

## 10 Conclusion

We characterized two boundaries that govern non-determinism in GPU graph algorithms. The *determinism boundary*, rooted in the reduction operator's order-independence under IEEE 754, separates algorithms whose output is invariant to atomic scheduling ($\min$-based SSSP) from those that drift ($\sum$-based PageRank). The *scheduling purity boundary*, formalized as four operational conditions in Theorem 5.1, further determines which algorithms within the reproducible class tolerate relaxed atomic discipline.

Through cross-vendor experiments on three GPUs from two vendors, we established that all observed drift in the studied kernels originates from non-deterministic atomic-add scheduling --- not from hardware floating-point differences, compiler variation, or microarchitectural effects. For the non-deterministic class, we demonstrated that deterministic reduction eliminates drift with negligible incremental cost over the pull baseline (Section 7.5), while the larger push-vs-pull performance tradeoff remains graph-dependent, achieving cross-vendor byte-identity across 180 runs with zero bit differences. For the deterministic class, we provided a certificate-based verifier that handles the legitimate output variation (predecessor non-determinism) that golden-output comparison cannot, with 99.94% error-detection sensitivity across 3,420 injection cases.

---

## References

[alglave-gpu-2015] Alglave, J., Batty, M., Donaldson, A.F., Gopalakrishnan, G., Ketema, J., Poetzl, D., Sorensen, T., and Wickerson, J. "GPU Concurrency: Weak Behaviours and Programming Assumptions." ASPLOS, pp. 577--591, 2015. DOI: 10.1145/2694344.2694391

[alkassar-bohme-mehlhorn-rizkallah-2014] Alkassar, E., Böhme, S., Mehlhorn, K., and Rizkallah, C. "A Framework for the Verification of Certifying Computations." J. Automated Reasoning 52(3):241--273, 2014. DOI: 10.1007/s10817-013-9289-2

[bellman-1958] Bellman, R. "On a Routing Problem." Quarterly of Applied Mathematics 16(1):87--90, 1958. DOI: 10.1090/qam/102435

[bertsekas-1982] Bertsekas, D.P. "Distributed Dynamic Programming." IEEE Trans. Automatic Control 27(3):610--616, 1982. DOI: 10.1109/TAC.1982.1102980

[bertsekas-1983] Bertsekas, D.P. "Distributed Asynchronous Computation of Fixed Points." Mathematical Programming 27(1):107--120, 1983. DOI: 10.1007/BF02591967

[che-pannotia-2013] Che, S., Beckmann, B.M., Reinhardt, S.K., and Skadron, K. "Pannotia: Understanding Irregular GPGPU Graph Applications." IISWC, pp. 185--195, 2013. DOI: 10.1109/IISWC.2013.6704684

[chou-dab-2020] Chou, Y.-H., Mohan, J., Yashwanth, B.R., Kim, H., and Patt, Y.N. "Deterministic Atomic Buffering." MICRO, pp. 404--417, 2020. DOI: 10.1109/MICRO50266.2020.00043

[cousot-cousot-1977] Cousot, P. and Cousot, R. "Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints." POPL, pp. 238--252, 1977. DOI: 10.1145/512950.512973

[demmel-nguyen-2013] Demmel, J. and Nguyen, H.D. "Fast Reproducible Floating-Point Summation." IEEE 21st Symposium on Computer Arithmetic (ARITH), pp. 163--172, 2013. DOI: 10.1109/ARITH.2013.9

[iakymchuk-exblas-2015] Iakymchuk, R., Graillat, S., Defour, D., and Collange, C. "ExBLAS: Reproducible and Accurate BLAS Library." HAL tech report hal-01202396, 2015.

[ieee754-2019] IEEE. "IEEE Standard for Floating-Point Arithmetic." IEEE Std 754-2019, 2019. DOI: 10.1109/IEEESTD.2019.8766229

[jooybar-gpudet-2013] Jooybar, H., Fung, W.W.L., O'Connor, M., Devietti, J., and Aamodt, T.M. "GPUDet: A Deterministic GPU Architecture." ASPLOS, pp. 1--12, 2013. DOI: 10.1145/2451116.2451118

[kildall-1973] Kildall, G.A. "A Unified Approach to Global Program Optimization." POPL, pp. 194--206, 1973. DOI: 10.1145/512927.512945

[lindsay-counterpoint-2026] Lindsay, N., Trippel, C., Khandelwal, A., and Bhattacharjee, A. "CounterPoint: Using Hardware Event Counters to Refute and Refine Microarchitectural Assumptions." ASPLOS Vol. 2, 2026. Best Paper Award. DOI: 10.1145/3779212.3790145

[lustig-ptx-2019] Lustig, D., Sahasrabuddhe, S., and Giroux, O. "A Formal Analysis of the NVIDIA PTX Memory Consistency Model." ASPLOS, pp. 257--270, 2019. DOI: 10.1145/3297858.3304043

[mehlhorn-mcconnell-naher-schweitzer-2011] McConnell, R.M., Mehlhorn, K., Näher, S., and Schweitzer, P. "Certifying Algorithms." Computer Science Review 5(2):119--161, 2011. DOI: 10.1016/j.cosrev.2010.09.009

[meyer-sanders-2003] Meyer, U. and Sanders, P. "Δ-stepping: A Parallelizable Shortest Path Algorithm." J. Algorithms 49(1):114--152, 2003. DOI: 10.1016/S0196-6774(03)00076-2

[nvidia-cugraph] NVIDIA RAPIDS Team. "cuGraph: GPU-Accelerated Graph Analytics." RAPIDS open-source library, 2024. https://github.com/rapidsai/cugraph

[riedy-demmel-reproblas] Ahrens, P., Demmel, J., Nguyen, H.D., and Riedy, E.J. "ReproBLAS: Reproducible BLAS." BeBOP project, 2016. https://bebop.cs.berkeley.edu/reproblas/

[shanmugavelu-fp-noassoc-2024] Shanmugavelu, S., Taillefumier, M., Culver, C., Hernandez, O., Coletti, M., and Sedova, A. "Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications." SC'24 Workshops (SCW), 2024. DOI: 10.1109/SCW63240.2024.00028

[wang-gunrock-2016] Wang, Y., Davidson, A., Pan, Y., Wu, Y., Riffel, A., and Owens, J.D. "Gunrock: A High-Performance Graph Processing Library on the GPU." PPoPP, pp. 11:1--11:12, 2016. DOI: 10.1145/2851141.2851145

[flor-adgraph-2025] Flor, R., Fernandes, L.G., and Schnorr, L.M. "adGRAPH: GPU Architectures in Graph Analytics." EDBT, 2025. DOI: 10.48786/edbt.2025.20

[vanausdal-burtscher-2025] VanAusdal, B. and Burtscher, M. "Comparing Graph Algorithm Styles on NVIDIA and AMD GPUs." SC'25 Workshop on Graphs, Architectures, Programming, and Learning (GrAPL), 2025.
