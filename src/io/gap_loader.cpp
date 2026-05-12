// gap_loader.cpp — Load GAP benchmark graphs (.mtx Matrix Market)
//
// Supported: Matrix Market edge list (.mtx), undirected or directed.
// Lines starting with '%' are comments.  Header line:
//   %%MatrixMarket matrix coordinate [real|integer|pattern] [general|symmetric]
// Data lines: <row> <col> [value]   (1-indexed)
//
// For GAP road graphs (.mtx with integer weights), we cast to W.
// For unweighted (.mtx pattern), we assign uniform FP in [0.001, 1.0].
// Symmetric flag: we store both directions (directed representation).
//
// .wsg (Galois binary) is NOT yet implemented; only .mtx is handled.

#include "gap_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>

namespace {

struct Edge {
    uint32_t u, v;
    double   w;
};

// Parse Matrix Market file.  Returns edge list (directed, 0-indexed).
std::vector<Edge> parse_mtx(const char* path, bool& has_weights, bool& symmetric) {
    FILE* fp = fopen(path, "r");
    if (!fp) throw std::runtime_error(std::string("gap_loader: cannot open ") + path);

    std::vector<Edge> edges;
    has_weights = false;
    symmetric   = false;
    bool header_done = false;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '%') {
            // Parse banner line
            if (strncmp(line, "%%MatrixMarket", 14) == 0) {
                char obj[64], fmt[64], field[64], sym[64];
                // "%%MatrixMarket matrix coordinate [field] [symmetry]"
                if (sscanf(line + 15, "%63s %63s %63s %63s",
                           obj, fmt, field, sym) >= 2) {
                    has_weights = (strncasecmp(field, "pattern", 7) != 0);
                    symmetric   = (strncasecmp(sym,   "symmetric", 9) == 0);
                }
            }
            continue;
        }
        if (!header_done) {
            // Skip size line: rows cols nnz
            header_done = true;
            continue;
        }
        uint32_t r, c;
        double   w = 1.0;
        if (has_weights)
            sscanf(line, "%u %u %lf", &r, &c, &w);
        else
            sscanf(line, "%u %u", &r, &c);
        if (r == c) continue;  // skip self-loops
        edges.push_back({r - 1, c - 1, w});
        if (symmetric && r != c)
            edges.push_back({c - 1, r - 1, w});
    }
    fclose(fp);
    return edges;
}

// Build CSR from edge list.
template<typename W>
CSR<W> build_csr(std::vector<Edge>& edges, uint32_t n_v, unsigned seed) {
    // Sort by source
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.u < b.u || (a.u == b.u && a.v < b.v);
    });
    // Deduplicate (keep first occurrence)
    edges.erase(std::unique(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.u == b.u && a.v == b.v;
    }), edges.end());

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_w(0.001, 1.0);

    CSR<W> g;
    g.n_vertices = n_v;
    g.n_edges    = (eid_t)edges.size();
    g.row_offsets.resize(n_v + 1, 0);
    g.col_indices.resize(g.n_edges);
    g.weights.resize(g.n_edges);

    for (const auto& e : edges) g.row_offsets[e.u + 1]++;
    for (vid_t i = 0; i < n_v; ++i) g.row_offsets[i + 1] += g.row_offsets[i];

    std::vector<eid_t> pos(g.row_offsets.begin(), g.row_offsets.end());
    for (const auto& e : edges) {
        eid_t p = pos[e.u]++;
        g.col_indices[p] = e.v;
        g.weights[p]     = (e.w == 1.0 && seed != 0)
                           ? (W)dist_w(rng)   // unweighted → synthetic FP
                           : (W)e.w;
    }
    return g;
}

} // anonymous namespace

template<typename W>
CSR<W> load_gap(const char* path, unsigned weight_seed) {
    // Determine extension
    const char* ext = strrchr(path, '.');
    if (!ext) throw std::runtime_error("gap_loader: unknown file extension");

    if (strcasecmp(ext, ".mtx") == 0) {
        bool has_weights, symmetric;
        auto edges = parse_mtx(path, has_weights, symmetric);
        if (edges.empty()) throw std::runtime_error("gap_loader: empty graph");

        // Determine n_vertices from max vertex id
        uint32_t n_v = 0;
        for (const auto& e : edges) {
            n_v = std::max(n_v, std::max(e.u, e.v) + 1);
        }

        // If has_weights, treat as original; else assign synthetic
        // For original weights, pass seed=0 so build_csr uses e.w directly
        unsigned eff_seed = has_weights ? 0 : weight_seed;
        return build_csr<W>(edges, n_v, eff_seed);
    }

    throw std::runtime_error(
        std::string("gap_loader: unsupported format ") + ext);
}

template CSR<float>  load_gap<float> (const char*, unsigned);
template CSR<double> load_gap<double>(const char*, unsigned);
