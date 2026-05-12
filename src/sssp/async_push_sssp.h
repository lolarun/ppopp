#pragma once
// async_push_sssp.h — GPU asynchronous push-based SSSP declaration (HIP unified).
//
// Only compiled when USE_GPU is defined.  Implementation in async_push_sssp.hip.
//
// Third algorithm class for the §5 boundary characterization (A1.38).  Sits
// between Δ-stepping (bucket-once) and Bellman-Ford (iterate-to-fixedpoint):
//
//   - Like Δ-stepping: not every vertex is touched every iteration; only
//     "active" vertices (those whose d was lowered last round) participate
//     in the next relaxation kernel.
//   - Like Bellman-Ford: the host loop iterates to a global fixedpoint check
//     (any_active == 0); a vertex can be re-activated arbitrarily many times.
//   - Unlike Δ-stepping: no bucket structure, no Δ heuristic, no light/heavy
//     edge split.  Order of relaxations is fully race-determined within each
//     kernel launch.
//   - Unlike Bellman-Ford: doesn't visit every reachable vertex every
//     iteration; only currently-active ones.
//
// Per §5.3 Lemma 5.1, async push-based SSSP should EMPIRICALLY behave like
// Bellman-Ford under relaxed per-vertex atomics: the global fixedpoint check
// (still using atomicOr on the active flag) re-validates the edge inequality
// at termination, repairing race-induced transient inconsistencies.  This
// extends the §5 boundary from 2 algorithm instances to 3.

#ifdef USE_GPU

#include "../core/csr_sssp.h"

template<typename W>
Certificate<W> async_push_sssp_gpu(const CSR<W>& g, vid_t source, bool emit_cert);

extern template Certificate<float>  async_push_sssp_gpu<float> (const CSR<float>&,  vid_t, bool);
extern template Certificate<double> async_push_sssp_gpu<double>(const CSR<double>&, vid_t, bool);

#endif // USE_GPU
