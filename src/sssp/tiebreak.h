#pragma once
#include "../core/csr_sssp.h"

// Canonical tiebreak rule for SSSP predecessor selection.
//
// When relaxing edge (u → v) with new distance nd = d[u] + w:
//   - Strict improvement (nd < d[v])  → always update (d, pi)
//   - Tie         (nd == d[v])        → update pi only if u < current pi[v]
//
// This makes pi deterministic on a single platform for a fixed execution.
// Cross-platform: d[v] may differ due to FP non-associativity, so pi may
// legitimately differ across vendors even with this rule.  Both outputs
// remain SAT under the verifier.
//
// Used in: cpu_dijkstra, delta_stepping_cpu, delta_stepping_gpu.

template<typename W>
struct TiebreakRelax {
    // Returns true if the relaxation should update (d[v], pi[v]).
    // Caller must check both cases: strict_improvement or tie_wins.
    static bool strict_improvement(W nd, W dv) {
        return nd < dv;
    }
    static bool tie_wins(W nd, W dv, vid_t u, vid_t piv) {
        return nd == dv && u < piv;
    }
};
