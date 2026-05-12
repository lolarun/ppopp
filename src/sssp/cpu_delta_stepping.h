#pragma once
// cpu_delta_stepping.h — CPU Δ-stepping reference implementation.
//
// Single-threaded, bucket-based.  Used as:
//   1. A correctness oracle on machines without a GPU.
//   2. A baseline timing reference for verifier overhead experiments.
//
// Algorithm: Meyer & Sanders (2003), sequential variant.

#include "../core/csr_sssp.h"

// Run CPU Δ-stepping SSSP on graph g from source.
//
// delta      : bucket width.  Pass 0 to auto-select (= average edge weight).
// emit_cert  : when true, pi[] is populated (predecessor tree).
template<typename W>
Certificate<W> delta_stepping_cpu(
    const CSR<W>& g,
    vid_t         source,
    W             delta     = W{0},
    bool          emit_cert = true);

// Post-hoc predecessor reconstruction from distance array only.
// Scans all edges and picks the predecessor u that minimises
//   |d[v] - d[u] - w(u,v)|
// using relative epsilon comparison (8 * eps * max(|d[v]|, 1)).
// Useful when emit_cert was false but pi is needed afterwards.
template<typename W>
void reconstruct_pi(
    const CSR<W>&    g,
    vid_t            source,
    const std::vector<W>& dist,
    std::vector<vid_t>&   pi);

extern template Certificate<float>  delta_stepping_cpu<float> (const CSR<float>&,  vid_t, float,  bool);
extern template Certificate<double> delta_stepping_cpu<double>(const CSR<double>&, vid_t, double, bool);

extern template void reconstruct_pi<float> (const CSR<float>&,  vid_t, const std::vector<float>&,  std::vector<vid_t>&);
extern template void reconstruct_pi<double>(const CSR<double>&, vid_t, const std::vector<double>&, std::vector<vid_t>&);
