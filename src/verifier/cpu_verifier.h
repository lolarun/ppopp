#pragma once
#include "invariants.h"
#include "../core/csr_sssp.h"
#include <span>
#include <vector>

// O(V+E) certificate verifier, OpenMP-parallel.
// Checks all four SSSP invariants:
//   1. Source: d[s]=0, pi[s]=INVALID_VID
//   2. Relaxation: ∀(u,v,w): d[v] <= d[u] + w
//   3. Predecessor consistency: pi[v] is a valid neighbor with correct distance
//   4. Tree structure: pi forest forms a tree rooted at source (no cycles)
//
// num_threads=0 uses OMP_NUM_THREADS default.
template<typename W>
VerifyResult verify(
    const CSR<W>&          g,
    vid_t                  source,
    std::span<const W>     d,
    std::span<const vid_t> pi,
    int                    num_threads = 0
);

// Convenience overload: accepts vectors directly (avoids span CTAD const issues).
template<typename W>
inline VerifyResult verify(
    const CSR<W>&              g,
    vid_t                      source,
    const std::vector<W>&      d,
    const std::vector<vid_t>&  pi,
    int                        num_threads = 0)
{
    return verify(g, source,
                  std::span<const W>(d),
                  std::span<const vid_t>(pi),
                  num_threads);
}
