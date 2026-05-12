#include "cpu_verifier.h"
#include <omp.h>
#include <atomic>
#include <vector>
#include <numeric>
#include <chrono>
#include <cmath>

// ── Helpers ──────────────────────────────────────────────────────────────────

// Relative-epsilon FP comparison for predecessor distance mismatch.
// Returns true if |a - b| > ε·max(|a|,|b|,1).
//
// Tolerance constant: each FP add of (d[u]+w) introduces ~1 ULP of relative
// error. After K accumulated additions along an SSSP path, the worst-case
// relative error is bounded by K * eps. The relevant K is the longest reachable
// path in edge count, which equals the SSSP-tree depth from the source.
//
// Empirically (see findings F10): kEps = 8 * eps was too tight on usa_road
// (24M v, depth ~5000+) under gaussian weight FP32 — accumulated rounding
// exceeded the 8-eps tolerance and produced false UNSAT_PRED_DISTANCE_MISMATCH
// even though the algorithm was correct. Bumped to 4096 * eps (≈4.9e-4 for
// FP32, ≈9.1e-13 for FP64) which absorbs accumulated rounding for realistic
// SSSP depths while still rejecting algorithmically-invalid certificates
// (which would differ by >>1 ULP from correct values).
template<typename W>
static bool fp_ne(W a, W b) {
    constexpr W kEps = W{4096} * std::numeric_limits<W>::epsilon();
    W scale = std::max({std::abs(a), std::abs(b), W{1}});
    return std::abs(a - b) > kEps * scale;
}

// ── Invariant checks ─────────────────────────────────────────────────────────

template<typename W>
static VerifyResult check_source(
    vid_t source,
    std::span<const W>     d,
    std::span<const vid_t> pi)
{
    if (d[source] != W{0})
        return {Verdict::UNSAT_SOURCE_DISTANCE, source, std::nullopt};
    if (pi[source] != INVALID_VID)
        return {Verdict::UNSAT_SOURCE_PRED, source, std::nullopt};
    return {Verdict::SAT};
}

template<typename W>
static VerifyResult check_relaxation(
    const CSR<W>&      g,
    std::span<const W> d,
    int                num_threads)
{
    std::atomic<bool>  failed{false};
    std::atomic<vid_t> wit_v{INVALID_VID};
    std::atomic<eid_t> wit_e{eid_t(-1)};

    #pragma omp parallel for schedule(static, 4096) num_threads(num_threads) \
        if(num_threads > 0)
    for (eid_t u = 0; u < (eid_t)g.n_vertices; ++u) {
        if (failed.load(std::memory_order_relaxed)) continue;
        if (d[u] == Sentinel<W>::inf) continue;

        for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            if (failed.load(std::memory_order_relaxed)) break;
            vid_t v = g.col_indices[e];
            W     w = g.weights[e];
            if (d[v] > d[u] + w) {
                failed.store(true);
                wit_v.store((vid_t)u);
                wit_e.store(e);
            }
        }
    }

    if (failed.load())
        return {Verdict::UNSAT_RELAXATION, wit_v.load(), wit_e.load()};
    return {Verdict::SAT};
}

// Encode Verdict as int for atomic storage (Verdict enum values fit in int).
// UNSET sentinel: -1 (no failure yet).
static constexpr int VERDICT_UNSET = -1;

template<typename W>
static VerifyResult check_predecessor(
    const CSR<W>&          g,
    vid_t                  source,
    std::span<const W>     d,
    std::span<const vid_t> pi,
    int                    num_threads)
{
    // atomic<int> stores the first Verdict that triggers; -1 = none yet.
    std::atomic<int>   first_verdict{VERDICT_UNSET};
    std::atomic<vid_t> wit_v{INVALID_VID};

    auto try_fail = [&](Verdict verd, vid_t v) {
        int expected = VERDICT_UNSET;
        // Only record the FIRST failure (CAS from unset).
        first_verdict.compare_exchange_strong(expected, (int)verd,
                                              std::memory_order_relaxed);
        // Always try to record a witness vertex (best-effort).
        vid_t exp_v = INVALID_VID;
        wit_v.compare_exchange_strong(exp_v, v, std::memory_order_relaxed);
    };

    #pragma omp parallel for schedule(static, 1024) num_threads(num_threads) \
        if(num_threads > 0)
    for (eid_t v = 0; v < (eid_t)g.n_vertices; ++v) {
        if (first_verdict.load(std::memory_order_relaxed) != VERDICT_UNSET) continue;
        if ((vid_t)v == source) continue;

        if (d[v] == Sentinel<W>::inf) {
            // Unreachable: pi must be INVALID_VID
            if (pi[v] != INVALID_VID)
                try_fail(Verdict::UNSAT_UNREACHABLE_PRED, (vid_t)v);
            continue;
        }

        // Reachable: pi must be a valid neighbor with correct distance
        if (pi[v] == INVALID_VID) {
            try_fail(Verdict::UNSAT_REACHABLE_NO_PRED, (vid_t)v);
            continue;
        }

        vid_t u    = pi[v];
        W     w_uv = g.edge_weight(u, (vid_t)v);

        if (w_uv == Sentinel<W>::inf) {
            try_fail(Verdict::UNSAT_PRED_NOT_NEIGHBOR, (vid_t)v);
            continue;
        }

        // d[v] must equal d[u] + w(u,v)  (within FP tolerance)
        if (fp_ne(d[v], d[u] + w_uv))
            try_fail(Verdict::UNSAT_PRED_DISTANCE_MISMATCH, (vid_t)v);
    }

    int fv = first_verdict.load();
    if (fv != VERDICT_UNSET)
        return {(Verdict)fv, wit_v.load(), std::nullopt};
    return {Verdict::SAT};
}

// Union-Find (path compression + union by rank)
struct UF {
    std::vector<vid_t> parent;
    std::vector<int>   rank;

    explicit UF(vid_t n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    vid_t find(vid_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }

    // Returns false if x and y are already in the same set (→ cycle)
    bool unite(vid_t x, vid_t y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank[x] < rank[y]) std::swap(x, y);
        parent[y] = x;
        if (rank[x] == rank[y]) rank[x]++;
        return true;
    }
};

template<typename W>
static VerifyResult check_tree(
    const CSR<W>&          g,
    vid_t                  source,
    std::span<const W>     d,
    std::span<const vid_t> pi)
{
    UF uf(g.n_vertices);

    for (vid_t v = 0; v < g.n_vertices; ++v) {
        if (v == source) continue;
        if (pi[v] == INVALID_VID) continue;  // unreachable, skip

        if (!uf.unite(v, pi[v]))
            return {Verdict::UNSAT_CYCLE, v, std::nullopt};
    }

    // Every reachable vertex must be in the same component as source
    vid_t root_s = uf.find(source);
    for (vid_t v = 0; v < g.n_vertices; ++v) {
        if (d[v] == Sentinel<W>::inf) continue;
        if (uf.find(v) != root_s)
            return {Verdict::UNSAT_DISCONNECTED_TREE, v, std::nullopt};
    }

    return {Verdict::SAT};
}

// ── Public entry ─────────────────────────────────────────────────────────────

template<typename W>
VerifyResult verify(
    const CSR<W>&          g,
    vid_t                  source,
    std::span<const W>     d,
    std::span<const vid_t> pi,
    int                    num_threads)
{
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    VerifyResult r = check_source(source, d, pi);
    if (!r.sat()) { r.wall_time_ms = elapsed(); return r; }

    r = check_relaxation(g, d, num_threads);
    if (!r.sat()) { r.wall_time_ms = elapsed(); return r; }

    r = check_predecessor(g, source, d, pi, num_threads);
    if (!r.sat()) { r.wall_time_ms = elapsed(); return r; }

    r = check_tree(g, source, d, pi);
    r.wall_time_ms = elapsed();
    return r;
}

template VerifyResult verify<float> (const CSR<float>&,  vid_t, std::span<const float>,  std::span<const vid_t>, int);
template VerifyResult verify<double>(const CSR<double>&, vid_t, std::span<const double>, std::span<const vid_t>, int);
