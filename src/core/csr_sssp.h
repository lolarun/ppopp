#pragma once
#include <vector>
#include <cstdint>
#include <limits>
#include <cassert>
#include <algorithm>

using vid_t = uint32_t;
using eid_t = uint64_t;

constexpr vid_t INVALID_VID = ~vid_t{0};

template<typename W>
struct Sentinel {
    static constexpr W inf = std::numeric_limits<W>::infinity();
};
// __half doesn't have numeric_limits; specialize in precision.h when needed

template<typename W>
struct CSR {
    vid_t n_vertices{0};
    eid_t n_edges{0};
    std::vector<eid_t> row_offsets;  // size n_vertices + 1
    std::vector<vid_t> col_indices;  // size n_edges
    std::vector<W>     weights;      // size n_edges

    bool empty() const { return n_vertices == 0; }

    // Degree of vertex u
    eid_t degree(vid_t u) const {
        assert(u < n_vertices);
        return row_offsets[u + 1] - row_offsets[u];
    }

    // Sort adjacency lists by col_index (needed for binary-search lookup in verifier)
    void sort_neighbors() {
        for (vid_t u = 0; u < n_vertices; ++u) {
            eid_t begin = row_offsets[u];
            eid_t end   = row_offsets[u + 1];
            // Sort (col, weight) pairs together
            std::vector<std::pair<vid_t, W>> tmp;
            tmp.reserve(end - begin);
            for (eid_t e = begin; e < end; ++e)
                tmp.push_back({col_indices[e], weights[e]});
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (eid_t e = begin; e < end; ++e) {
                col_indices[e] = tmp[e - begin].first;
                weights[e]     = tmp[e - begin].second;
            }
        }
    }

    // Binary search for edge (u, v); returns weight or Sentinel::inf if not found
    W edge_weight(vid_t u, vid_t v) const {
        eid_t lo = row_offsets[u], hi = row_offsets[u + 1];
        while (lo < hi) {
            eid_t mid = lo + (hi - lo) / 2;
            if (col_indices[mid] == v) return weights[mid];
            if (col_indices[mid] <  v) lo = mid + 1;
            else                        hi = mid;
        }
        return Sentinel<W>::inf;
    }
};

template<typename W>
struct Certificate {
    std::vector<W>     d;   // size n_vertices; Sentinel<W>::inf = unreachable
    std::vector<vid_t> pi;  // size n_vertices; INVALID_VID = source or unreachable
};
