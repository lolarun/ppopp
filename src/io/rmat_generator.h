#pragma once
// rmat_generator.h — In-memory RMAT (R-MAT) synthetic graph generator.
//
// Generates Graph500-compatible RMAT graphs:
//   n_vertices = 2^scale,  n_edges ≈ edgefactor * n_vertices
//   Parameters: a=0.57, b=0.19, c=0.19, d=0.05  (Graph500 default)
//
// Edge weights: uniform random FP in [0.001, 1.0], seed fixed by caller.
// Self-loops and duplicate edges are removed; the graph is directed.

#include "../core/csr_sssp.h"

struct RMATConfig {
    int      scale       = 25;    // 2^scale vertices
    int      edgefactor  = 32;    // edges per vertex
    double   a = 0.57, b = 0.19, c = 0.19;  // d = 1-a-b-c
    unsigned seed        = 42;
};

// Generate RMAT CSR.  Returns directed, no self-loops, no duplicate edges.
template<typename W>
CSR<W> generate_rmat(const RMATConfig& cfg = RMATConfig{});

extern template CSR<float>  generate_rmat<float> (const RMATConfig&);
extern template CSR<double> generate_rmat<double>(const RMATConfig&);
