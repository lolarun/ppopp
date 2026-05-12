#pragma once
#include "../core/csr_sssp.h"
#include <string>

// Load a DIMACS .gr file (9th DIMACS Challenge format):
//   c <comment>
//   p sp <n_vertices> <n_edges>
//   a <u> <v> <w>        (1-indexed, directed)
//
// Weights cast to W. Vertices remapped to 0-indexed.
template<typename W = float>
CSR<W> load_dimacs(const std::string& path);
