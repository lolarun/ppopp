#include "cpu_dijkstra.h"
#include <queue>
#include <vector>

template<typename W>
Certificate<W> dijkstra_cpu(const CSR<W>& g, vid_t source) {
    using State = std::pair<W, vid_t>;  // (dist, vertex)

    Certificate<W> cert;
    cert.d.assign(g.n_vertices, Sentinel<W>::inf);
    cert.pi.assign(g.n_vertices, INVALID_VID);

    cert.d[source] = W{0};

    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.push({W{0}, source});

    // Tracks which vertices have been finalised. Prevents the tiebreak from
    // setting pi[v]=u when v is already settled and u is a descendant of v —
    // which creates a cycle in the predecessor forest on large graphs where
    // float32 rounding makes two distinct path lengths compare equal.
    std::vector<bool> settled(g.n_vertices, false);

    while (!pq.empty()) {
        auto [du, u] = pq.top(); pq.pop();

        if (du > cert.d[u]) continue;  // stale entry
        if (settled[u]) continue;      // already finalised (handles equal-dist duplicates)
        settled[u] = true;

        for (eid_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            vid_t v  = g.col_indices[e];
            W    w   = g.weights[e];
            W    nd  = du + w;

            if (nd < cert.d[v]) {
                cert.d[v]  = nd;
                cert.pi[v] = u;
                pq.push({nd, v});
            } else if (!settled[v] && nd == cert.d[v] && u < cert.pi[v]) {
                // Tiebreak: smaller vid wins, only while v is still unsettled
                cert.pi[v] = u;
            }
        }
    }

    return cert;
}

template Certificate<float>  dijkstra_cpu<float> (const CSR<float>&,  vid_t);
template Certificate<double> dijkstra_cpu<double>(const CSR<double>&, vid_t);
