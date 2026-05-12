// cpu_delta_stepping.cpp — CPU Δ-stepping reference (sequential)
//
// Meyer & Sanders (2003) sequential variant.
// Bucket structure: std::vector<std::vector<vid_t>> buckets;
//   bucket i holds vertices v with floor(d[v]/delta) == i.
// Phase A: light edges (w <= delta) processed until bucket i_min is empty.
// Phase B: heavy edges (w  > delta) processed once from removed set.

#include "cpu_delta_stepping.h"
#include "tiebreak.h"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

// ── Helpers ──────────────────────────────────────────────────────────────────

template<typename W>
static inline W avg_weight(const CSR<W>& g) {
    if (g.n_edges == 0) return W{1};
    double sum = 0.0;
    for (eid_t e = 0; e < g.n_edges; ++e) sum += (double)g.weights[e];
    return (W)(sum / (double)g.n_edges);
}

// Bucket index for distance d and bucket width delta.
// Returns INT32_MAX for INF values.
template<typename W>
static inline int bid(W d, W delta) {
    if (d >= Sentinel<W>::inf / W{2}) return INT32_MAX;
    return (int)(d / delta);
}

// ── Main algorithm ────────────────────────────────────────────────────────────

template<typename W>
Certificate<W> delta_stepping_cpu(
    const CSR<W>& g, vid_t source, W delta, bool emit_cert)
{
    const vid_t NV  = g.n_vertices;
    const W     INF = Sentinel<W>::inf;

    if (delta == W{0}) delta = avg_weight(g);
    if (delta == W{0}) delta = W{1};

    // Distance + predecessor arrays
    std::vector<W>     dist(NV, INF);
    std::vector<vid_t> pi(NV, INVALID_VID);

    // Bucket structure: dynamic array of buckets.
    // We use a circular / growing scheme: bucket index is unbounded.
    // Implementation: vector<vector<vid_t>> with lazy expansion.
    const int INIT_BUCKETS = 1024;
    std::vector<std::vector<vid_t>> buckets(INIT_BUCKETS);

    auto ensure_bucket = [&](int b) {
        if (b >= (int)buckets.size())
            buckets.resize(b + 1);
    };

    // Relax edge (u -> v, weight w): update dist/pi/bucket if improved.
    auto relax = [&](vid_t u, vid_t v, W w) {
        W nd = dist[u] + w;
        if (nd < dist[v]) {
            // Remove v from its current bucket if active
            int old_b = bid(dist[v], delta);
            if (old_b != INT32_MAX) {
                // Mark as inactive (lazy deletion: we check on pop)
                // We do not eagerly remove; we'll skip stale entries below.
            }
            dist[v] = nd;
            if (emit_cert) pi[v] = u;
            int new_b = bid(nd, delta);
            ensure_bucket(new_b);
            buckets[new_b].push_back(v);
        } else if (emit_cert &&
                   TiebreakRelax<W>::tie_wins(nd, dist[v], u, pi[v])) {
            pi[v] = u;
        }
    };

    // Initialise source
    dist[source] = W{0};
    // pi[source] stays INVALID_VID — verifier invariant 1 requires pi[s] == INVALID_VID.
    // (The source has no predecessor by definition.)
    ensure_bucket(0);
    buckets[0].push_back(source);

    // Main loop
    int i_min = 0;
    while (true) {
        // Find next non-empty bucket
        while (i_min < (int)buckets.size() && buckets[i_min].empty())
            ++i_min;
        if (i_min >= (int)buckets.size()) break;

        // Phase A: light edges — iterate until bucket i_min is drained
        std::vector<vid_t> removed;
        while (!buckets[i_min].empty()) {
            // Drain current bucket into S (may grow mid-iteration)
            std::vector<vid_t> S;
            S.swap(buckets[i_min]);

            for (vid_t u : S) {
                // Stale check: u must still belong to bucket i_min
                if (bid(dist[u], delta) != i_min) continue;
                removed.push_back(u);
                // Light relax
                for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
                    W w = g.weights[e];
                    if (w > delta) continue;  // heavy — skip
                    relax(u, g.col_indices[e], w);
                }
            }
            // buckets[i_min] may have grown from relaxations above; loop again
        }

        // Phase B: heavy edges from removed set
        for (vid_t u : removed) {
            for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
                W w = g.weights[e];
                if (w <= delta) continue;  // light — skip
                relax(u, g.col_indices[e], w);
            }
        }
    }

    Certificate<W> cert;
    cert.d  = std::move(dist);
    cert.pi = emit_cert ? std::move(pi) : std::vector<vid_t>(NV, INVALID_VID);
    return cert;
}

// ── Post-hoc π reconstruction ─────────────────────────────────────────────────

template<typename W>
void reconstruct_pi(
    const CSR<W>&         g,
    vid_t                 source,
    const std::vector<W>& dist,
    std::vector<vid_t>&   pi)
{
    const vid_t NV  = g.n_vertices;
    const W     INF = Sentinel<W>::inf;
    // Relative epsilon: 8 * machine_epsilon * max(|d|, 1)
    const W eps8    = W{8} * std::numeric_limits<W>::epsilon();

    pi.assign(NV, INVALID_VID);
    // pi[source] stays INVALID_VID — verifier invariant 1 requires pi[s] == INVALID_VID.

    for (vid_t u = 0; u < NV; ++u) {
        if (dist[u] >= INF / W{2}) continue;  // u unreachable
        for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            vid_t v = g.col_indices[e];
            W     w = g.weights[e];
            W     nd = dist[u] + w;
            W     dv = dist[v];
            if (dv >= INF / W{2}) continue;
            // epsilon comparison: |nd - dv| <= eps * max(|dv|, 1)
            W tol = eps8 * (dv > W{1} ? dv : W{1});
            W diff = nd > dv ? nd - dv : dv - nd;
            if (diff <= tol) {
                // u is a valid predecessor of v
                // Apply tiebreak: smallest vid wins
                if (pi[v] == INVALID_VID || u < pi[v]) {
                    pi[v] = u;
                }
            }
        }
    }
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template Certificate<float>  delta_stepping_cpu<float> (const CSR<float>&,  vid_t, float,  bool);
template Certificate<double> delta_stepping_cpu<double>(const CSR<double>&, vid_t, double, bool);

template void reconstruct_pi<float> (const CSR<float>&,  vid_t, const std::vector<float>&,  std::vector<vid_t>&);
template void reconstruct_pi<double>(const CSR<double>&, vid_t, const std::vector<double>&, std::vector<vid_t>&);
