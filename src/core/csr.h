#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

// Binary CSR format produced by scripts/snap_to_csr.py and gen_rmat.py:
//   uint64_t magic    = 0x52455F435352304B  ('RE_CSR0K')
//   uint64_t N        (vertex count)
//   uint64_t E        (directed edge count)
//   int32_t  row_ptr[N+1]
//   int32_t  col_idx[E]
// CSR rows index OUTGOING edges (row u -> destinations of u's out-edges).
// PageRank push-style needs OUT-edges per vertex, so this layout is direct.

struct CsrGraph {
    int64_t N = 0;
    int64_t E = 0;
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_idx;
};

inline CsrGraph load_csr(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "load_csr: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    uint64_t magic = 0, N = 0, E = 0;
    if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x52455F435352304BULL) {
        std::fprintf(stderr, "load_csr: bad magic in %s (got 0x%016llx)\n",
                     path.c_str(), (unsigned long long)magic);
        std::exit(1);
    }
    if (std::fread(&N, sizeof(N), 1, f) != 1) std::exit(1);
    if (std::fread(&E, sizeof(E), 1, f) != 1) std::exit(1);

    CsrGraph g;
    g.N = (int64_t)N;
    g.E = (int64_t)E;
    g.row_ptr.resize(N + 1);
    g.col_idx.resize(E);
    if (std::fread(g.row_ptr.data(), sizeof(int32_t), N + 1, f) != (N + 1)) std::exit(1);
    if (std::fread(g.col_idx.data(), sizeof(int32_t), E, f) != E) std::exit(1);
    std::fclose(f);
    return g;
}
