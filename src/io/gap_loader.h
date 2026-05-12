#pragma once
// gap_loader.h — Load GAP benchmark graphs (Matrix Market / Galois binary).
//
// Supports:
//   .mtx  — Matrix Market edge list (1-indexed, may have values or not)
//   .wsg  — Galois weighted serialised graph (binary, GAP-road / kron-25)
//
// If the source file has no edge weights, synthetic FP weights are assigned:
//   uniform random in [0.001, 1.0] with the provided seed (default 42).
// Road graphs that carry original integer distances keep them as-is (cast to W).

#include "../core/csr_sssp.h"

// Load a GAP/MTX graph.  Infers format from file extension.
// weight_seed: used only when the source has no weights.
template<typename W>
CSR<W> load_gap(const char* path, unsigned weight_seed = 42);

extern template CSR<float>  load_gap<float> (const char*, unsigned);
extern template CSR<double> load_gap<double>(const char*, unsigned);
