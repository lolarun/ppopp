// rmat_generator.cpp — Graph500 RMAT generator.
//
// Generates directed RMAT edges using the Graph500 specification:
//   a=0.57, b=0.19, c=0.19, d=0.05
// n_vertices = 2^scale; n_edges = edgefactor * n_vertices.
// Self-loops and duplicates are removed.
// Edge weights: uniform FP in [0.001, 1.0] using cfg.seed.

#include "rmat_generator.h"
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cassert>

namespace {

// Generate a single RMAT edge for a 2^scale graph.
std::pair<uint32_t, uint32_t> rmat_edge(
    int scale, double a, double b, double c, double /*d*/,
    std::mt19937_64& rng)
{
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    uint32_t u = 0, v = 0;
    for (int bit = scale - 1; bit >= 0; --bit) {
        double r = ud(rng);
        uint32_t ub = 0, vb = 0;
        if      (r < a)         { ub = 0; vb = 0; }
        else if (r < a + b)     { ub = 0; vb = 1; }
        else if (r < a + b + c) { ub = 1; vb = 0; }
        else                    { ub = 1; vb = 1; }
        u |= (ub << bit);
        v |= (vb << bit);
    }
    return {u, v};
}

} // anonymous namespace

template<typename W>
CSR<W> generate_rmat(const RMATConfig& cfg) {
    const uint32_t n_v = (uint32_t)1 << cfg.scale;
    const eid_t    n_e_target = (eid_t)cfg.edgefactor * n_v;

    double d = 1.0 - cfg.a - cfg.b - cfg.c;
    assert(d > 0.0);

    std::mt19937_64 rng(cfg.seed);

    struct RawEdge { uint32_t u, v; };
    std::vector<RawEdge> raw;
    raw.reserve(n_e_target);

    // Generate ~n_e_target raw edges (over-generate slightly to cover removal)
    eid_t gen = (eid_t)(n_e_target * 1.05) + 16;
    for (eid_t i = 0; i < gen; ++i) {
        auto [u, v] = rmat_edge(cfg.scale, cfg.a, cfg.b, cfg.c, d, rng);
        if (u != v) raw.push_back({u, v});
    }

    // Sort + deduplicate
    std::sort(raw.begin(), raw.end(), [](const RawEdge& a, const RawEdge& b) {
        return a.u < b.u || (a.u == b.u && a.v < b.v);
    });
    raw.erase(std::unique(raw.begin(), raw.end(), [](const RawEdge& a, const RawEdge& b) {
        return a.u == b.u && a.v == b.v;
    }), raw.end());

    // Trim to target (may be slightly under if too many dupes; acceptable)
    if ((eid_t)raw.size() > n_e_target) raw.resize(n_e_target);

    eid_t n_e = (eid_t)raw.size();

    // Build row offsets
    std::vector<eid_t> row_off(n_v + 1, 0);
    for (const auto& e : raw) row_off[e.u + 1]++;
    for (uint32_t i = 0; i < n_v; ++i) row_off[i + 1] += row_off[i];

    // Assign synthetic FP weights
    std::mt19937 wrng(cfg.seed + 1000000u);  // separate stream from edge gen
    std::uniform_real_distribution<double> dist_w(0.001, 1.0);

    std::vector<vid_t> col(n_e);
    std::vector<W>     wgt(n_e);

    std::vector<eid_t> pos(row_off.begin(), row_off.end());
    for (const auto& e : raw) {
        eid_t p = pos[e.u]++;
        col[p]  = e.v;
        wgt[p]  = (W)dist_w(wrng);
    }

    CSR<W> g;
    g.n_vertices  = n_v;
    g.n_edges     = n_e;
    g.row_offsets = std::move(row_off);
    g.col_indices = std::move(col);
    g.weights     = std::move(wgt);
    return g;
}

template CSR<float>  generate_rmat<float> (const RMATConfig&);
template CSR<double> generate_rmat<double>(const RMATConfig&);
