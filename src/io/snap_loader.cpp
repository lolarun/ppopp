// snap_loader.cpp — Load SNAP edge-list graphs.
//
// SNAP format (directed, 0-indexed after remap):
//   Lines starting with '#' are comments.
//   Data lines: <src> <dst>    (may be 0-indexed or 1-indexed)
//
// We auto-detect indexing: if any id == 0 we treat as 0-indexed,
// otherwise 1-indexed.
//
// Weights: SNAP files are unweighted.  We assign uniform FP in [0.001, 1.0]
// using the provided seed (default 42).  This matches snap_to_dimacs.py.

#include "snap_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace {

struct RawEdge { uint32_t u, v; };

} // anonymous namespace

template<typename W>
CSR<W> load_snap(const char* path, unsigned weight_seed) {
    FILE* fp = fopen(path, "r");
    if (!fp) throw std::runtime_error(std::string("snap_loader: cannot open ") + path);

    std::vector<RawEdge> raw;
    raw.reserve(1 << 20);

    char line[128];
    bool has_zero = false;
    uint32_t max_id = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        uint32_t u, v;
        if (sscanf(line, "%u %u", &u, &v) != 2) continue;
        if (u == v) continue;  // skip self-loops
        if (u == 0 || v == 0) has_zero = true;
        max_id = std::max(max_id, std::max(u, v));
        raw.push_back({u, v});
    }
    fclose(fp);

    if (raw.empty()) throw std::runtime_error("snap_loader: empty graph");

    // Remap to 0-indexed
    uint32_t offset = has_zero ? 0 : 1;
    uint32_t n_v    = max_id - offset + 1;

    // Build vertex id remap (SNAP ids may not be contiguous)
    // Fast path: check if ids are already contiguous 0-based
    std::vector<RawEdge> edges;
    edges.reserve(raw.size());
    for (auto& e : raw) {
        uint32_t u = e.u - offset;
        uint32_t v = e.v - offset;
        edges.push_back({u, v});
    }

    // Sort + deduplicate
    std::sort(edges.begin(), edges.end(), [](const RawEdge& a, const RawEdge& b) {
        return a.u < b.u || (a.u == b.u && a.v < b.v);
    });
    edges.erase(std::unique(edges.begin(), edges.end(), [](const RawEdge& a, const RawEdge& b) {
        return a.u == b.u && a.v == b.v;
    }), edges.end());

    eid_t n_e = (eid_t)edges.size();

    // Build row offsets
    std::vector<eid_t> row_off(n_v + 1, 0);
    for (const auto& e : edges) row_off[e.u + 1]++;
    for (vid_t i = 0; i < n_v; ++i) row_off[i + 1] += row_off[i];

    // Assign synthetic FP weights
    std::mt19937 rng(weight_seed);
    std::uniform_real_distribution<double> dist_w(0.001, 1.0);

    std::vector<vid_t> col(n_e);
    std::vector<W>     wgt(n_e);

    std::vector<eid_t> pos(row_off.begin(), row_off.end());
    for (const auto& e : edges) {
        eid_t p  = pos[e.u]++;
        col[p]   = e.v;
        wgt[p]   = (W)dist_w(rng);
    }

    CSR<W> g;
    g.n_vertices  = n_v;
    g.n_edges     = n_e;
    g.row_offsets = std::move(row_off);
    g.col_indices = std::move(col);
    g.weights     = std::move(wgt);
    return g;
}

template CSR<float>  load_snap<float> (const char*, unsigned);
template CSR<double> load_snap<double>(const char*, unsigned);
