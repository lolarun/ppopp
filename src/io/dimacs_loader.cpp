#include "dimacs_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <tuple>

template<typename W>
CSR<W> load_dimacs(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);

    vid_t n_v = 0;
    eid_t n_e = 0;
    std::vector<std::tuple<vid_t, vid_t, W>> edges;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        char type = line[0];

        if (type == 'c') continue;

        if (type == 'p') {
            // p sp <n_v> <n_e>
            std::istringstream ss(line);
            std::string tok;
            ss >> tok >> tok >> n_v >> n_e;
            edges.reserve(n_e);
        } else if (type == 'a') {
            // a <u> <v> <w>  (1-indexed; w may be integer or decimal float)
            vid_t u, v; double w;
            std::istringstream ss(line);
            char ch;
            ss >> ch >> u >> v >> w;
            edges.emplace_back(u - 1, v - 1, static_cast<W>(w));
        }
    }

    if (n_v == 0)
        throw std::runtime_error("Invalid DIMACS file: no 'p' line found");

    // Build CSR
    CSR<W> g;
    g.n_vertices = n_v;
    g.n_edges    = edges.size();
    g.row_offsets.assign(n_v + 1, 0);
    g.col_indices.resize(g.n_edges);
    g.weights.resize(g.n_edges);

    // Count out-degrees
    for (auto& [u, v, w] : edges)
        g.row_offsets[u + 1]++;

    // Prefix sum
    for (vid_t u = 0; u < n_v; ++u)
        g.row_offsets[u + 1] += g.row_offsets[u];

    // Fill edges (use a copy of row_offsets as cursor)
    std::vector<eid_t> cursor(g.row_offsets.begin(), g.row_offsets.end());
    for (auto& [u, v, w] : edges) {
        eid_t pos = cursor[u]++;
        g.col_indices[pos] = v;
        g.weights[pos]     = w;
    }

    g.sort_neighbors();
    return g;
}

// Explicit instantiations
template CSR<float>  load_dimacs<float> (const std::string&);
template CSR<double> load_dimacs<double>(const std::string&);
