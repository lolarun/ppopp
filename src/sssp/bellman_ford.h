#pragma once
// bellman_ford.h — GPU Bellman-Ford declaration (HIP unified, NVIDIA + AMD)
//
// Only compiled when USE_GPU is defined (i.e. GPU_BACKEND != NONE).
// The actual implementation is in bellman_ford.hip.
//
// Sibling to delta_stepping_gpu — same min-plus relaxation primitives, but
// processes ALL edges every iteration (high contention) rather than only
// frontier edges from the active bucket.  Used for E11 cross-algorithm
// validation of the §II.E "atomic discipline is the enabler" claim.

#ifdef USE_GPU

#include "../core/csr_sssp.h"

template<typename W>
Certificate<W> bellman_ford_gpu(const CSR<W>& g, vid_t source, bool emit_cert);

extern template Certificate<float>  bellman_ford_gpu<float> (const CSR<float>&,  vid_t, bool);
extern template Certificate<double> bellman_ford_gpu<double>(const CSR<double>&, vid_t, bool);

#endif // USE_GPU
