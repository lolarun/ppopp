#include <catch2/catch_test_macros.hpp>
#include "../src/core/csr_sssp.h"
#include "../src/sssp/cpu_dijkstra.h"
#include "../src/verifier/cpu_verifier.h"

// Build a small CSR by hand
static CSR<float> make_graph(
    vid_t n,
    const std::vector<std::tuple<vid_t,vid_t,float>>& edges)
{
    CSR<float> g;
    g.n_vertices = n;
    g.n_edges    = edges.size();
    g.row_offsets.assign(n + 1, 0);
    for (auto& [u, v, w] : edges) g.row_offsets[u + 1]++;
    for (vid_t u = 0; u < n; ++u) g.row_offsets[u+1] += g.row_offsets[u];
    g.col_indices.resize(edges.size());
    g.weights.resize(edges.size());
    std::vector<eid_t> cur(g.row_offsets.begin(), g.row_offsets.end());
    for (auto& [u, v, w] : edges) {
        eid_t pos = cur[u]++;
        g.col_indices[pos] = v;
        g.weights[pos]     = w;
    }
    g.sort_neighbors();
    return g;
}

// ── SAT cases ─────────────────────────────────────────────────────────────────

TEST_CASE("trivial: single edge, source=0") {
    auto g = make_graph(2, {{0, 1, 3.0f}});
    auto cert = dijkstra_cpu(g, 0);
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.sat());
    REQUIRE(cert.d[0] == 0.0f);
    REQUIRE(cert.d[1] == 3.0f);
    REQUIRE(cert.pi[0] == INVALID_VID);
    REQUIRE(cert.pi[1] == 0);
}

TEST_CASE("linear chain: 0→1→2→3") {
    auto g = make_graph(4, {
        {0,1,1.0f},{1,2,2.0f},{2,3,3.0f}
    });
    auto cert = dijkstra_cpu(g, 0);
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.sat());
    REQUIRE(cert.d[3] == 6.0f);
}

TEST_CASE("disconnected: vertex 3 unreachable from 0") {
    auto g = make_graph(4, {
        {0,1,1.0f},{1,2,1.0f}  // vertex 3 isolated
    });
    auto cert = dijkstra_cpu(g, 0);
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.sat());
    REQUIRE(cert.d[3] == Sentinel<float>::inf);
    REQUIRE(cert.pi[3] == INVALID_VID);
}

TEST_CASE("source with no outgoing edges") {
    auto g = make_graph(3, {{1,2,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.sat());
    REQUIRE(cert.d[0] == 0.0f);
    REQUIRE(cert.d[1] == Sentinel<float>::inf);
}

TEST_CASE("diamond: two paths of equal length, tiebreak applies") {
    // 0→1 (1.0), 0→2 (1.0), 1→3 (1.0), 2→3 (1.0)
    // d[3]=2.0, pi[3] should be 1 (smaller vid)
    auto g = make_graph(4, {
        {0,1,1.0f},{0,2,1.0f},{1,3,1.0f},{2,3,1.0f}
    });
    auto cert = dijkstra_cpu(g, 0);
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.sat());
    REQUIRE(cert.d[3] == 2.0f);
    REQUIRE(cert.pi[3] == 1);  // tiebreak: smaller vid
}

// ── UNSAT cases ───────────────────────────────────────────────────────────────

TEST_CASE("UNSAT: wrong source distance") {
    auto g = make_graph(2, {{0,1,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    cert.d[0] = 1.0f;  // corrupt source
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.verdict == Verdict::UNSAT_SOURCE_DISTANCE);
}

TEST_CASE("UNSAT: relaxation violation") {
    auto g = make_graph(3, {{0,1,1.0f},{1,2,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    cert.d[2] = 5.0f;  // too large — violates d[2] <= d[1]+1
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.verdict == Verdict::UNSAT_RELAXATION);
}

TEST_CASE("UNSAT: predecessor not a neighbor") {
    auto g = make_graph(3, {{0,1,1.0f},{1,2,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    cert.pi[2] = 0;  // 0 is not a neighbor of 2
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.verdict == Verdict::UNSAT_PRED_NOT_NEIGHBOR);
}

TEST_CASE("UNSAT: predecessor distance mismatch") {
    auto g = make_graph(3, {{0,1,1.0f},{1,2,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    cert.d[2] = 1.5f;  // d[pi[2]]+w = 1+1=2, mismatch
    auto r = verify(g, 0, cert.d, cert.pi);
    // Could be relaxation or mismatch depending on order; either is UNSAT
    REQUIRE(r.verdict != Verdict::SAT);
}

TEST_CASE("UNSAT: cycle in pi") {
    auto g = make_graph(4, {
        {0,1,1.0f},{1,2,1.0f},{2,3,1.0f}
    });
    auto cert = dijkstra_cpu(g, 0);
    // Force a cycle: pi[1]=2, pi[2]=1
    cert.pi[1] = 2;
    cert.pi[2] = 1;
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.verdict != Verdict::SAT);  // cycle detected as any UNSAT (may be caught as pred mismatch first)
}

TEST_CASE("UNSAT: unreachable vertex has non-INVALID pi") {
    auto g = make_graph(3, {{0,1,1.0f}});
    auto cert = dijkstra_cpu(g, 0);
    // vertex 2 is unreachable
    cert.pi[2] = 0;  // corrupt: unreachable should have INVALID_VID
    auto r = verify(g, 0, cert.d, cert.pi);
    REQUIRE(r.verdict == Verdict::UNSAT_UNREACHABLE_PRED);
}
