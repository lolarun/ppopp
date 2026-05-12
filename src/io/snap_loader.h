#pragma once
// snap_loader.h — Load SNAP edge-list graphs.
//
// SNAP format: lines starting with '#' are comments; each data line is
//   <src_id> <dst_id>
// Vertices are 0-indexed in SNAP; this loader remaps to contiguous 0-based ids.
//
// Edge weights are NOT present in SNAP files.  Weights are assigned as
// uniform random FP in [0.001, 1.0] with the provided seed (default 42).
// This matches the snap_to_dimacs.py preprocessing step.

#include "../core/csr_sssp.h"

// Load a SNAP edge-list file.
// weight_seed: RNG seed for synthetic edge weights.
template<typename W>
CSR<W> load_snap(const char* path, unsigned weight_seed = 42);

extern template CSR<float>  load_snap<float> (const char*, unsigned);
extern template CSR<double> load_snap<double>(const char*, unsigned);
