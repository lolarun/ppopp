#pragma once
#include "../core/csr_sssp.h"

// Standard Dijkstra using a min-heap.
// O((V + E) log V). Used as ground-truth reference.
template<typename W>
Certificate<W> dijkstra_cpu(const CSR<W>& g, vid_t source);
