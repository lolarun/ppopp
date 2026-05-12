#pragma once
// delta_stepping.h — GPU Δ-stepping declaration (HIP unified, NVIDIA + AMD)
//
// Only compiled when USE_GPU is defined (i.e. GPU_BACKEND != NONE).
// The actual implementation is in delta_stepping.hip.

#ifdef USE_GPU

#include "../core/csr_sssp.h"

// Run GPU Δ-stepping SSSP on graph g from source.
//
// delta     : bucket width; heuristic default = average edge weight
// emit_cert : when true, predecessor array pi is tracked in-flight via
//             atomicExch.  When false, pi[] in the returned Certificate is
//             all INVALID_VID (saves ~10-15% SSSP time).
//
// Returns Certificate<W>{d, pi} where d[v] = shortest distance from source,
// pi[v] = predecessor on shortest-path tree (INVALID_VID if source or
// unreachable or emit_cert==false).
template<typename W>
Certificate<W> delta_stepping_gpu(
    const CSR<W>& g,
    vid_t         source,
    W             delta,
    bool          emit_cert);

// Explicit instantiations provided in delta_stepping.hip
extern template Certificate<float>  delta_stepping_gpu<float> (const CSR<float>&,  vid_t, float,  bool);
extern template Certificate<double> delta_stepping_gpu<double>(const CSR<double>&, vid_t, double, bool);

#endif // USE_GPU
