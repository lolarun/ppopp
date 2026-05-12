#pragma once
#include "../core/csr_sssp.h"
#include <random>

enum class ErrorKind {
    DISTANCE_PERTURB,      // d[v] += δ  (δ > 0, random magnitude)
    PREDECESSOR_RANDOM,    // pi[v] = random other vertex
    INCONSISTENT,          // d[v] changed but pi[v] left unchanged → mismatch
    MISSED_UNREACHABLE,    // d[v] set to INF for a reachable vertex
    CYCLE,                 // create pi[a]=b, pi[b]=a
};

// Inject n_errors errors of given kind into a (copy of) certificate.
// Returns the mutated certificate.
template<typename W>
Certificate<W> inject_errors(
    const CSR<W>&          g,
    const Certificate<W>&  original,
    ErrorKind              kind,
    int                    n_errors,
    int                    seed);
