#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "sssp/cpu_delta_stepping.h"
#include "sssp/cpu_dijkstra.h"
#include "core/csr_sssp.h"

// Build a small directed graph:
//   0 --(1.0)--> 1 --(1.0)--> 2
//   0 --(3.5)--> 2
// Shortest path 0→2: via 1, cost 2.0
static CSR<float> small_graph() {
    CSR<float> g;
    g.n_vertices = 3;
    g.n_edges    = 3;
    g.row_offsets = {0, 2, 3, 3};
    g.col_indices = {1, 2, 2};
    g.weights     = {1.0f, 3.5f, 1.0f};
    return g;
}

// 4-node graph with heavier edges to test Phase B
static CSR<float> mixed_graph() {
    // 0 --(0.5)--> 1   (light, Δ=1)
    // 0 --(2.0)--> 2   (heavy)
    // 1 --(0.5)--> 3   (light)
    // 2 --(0.5)--> 3   (light after heavy)
    CSR<float> g;
    g.n_vertices = 4;
    g.n_edges    = 4;
    g.row_offsets = {0, 2, 3, 4, 4};
    g.col_indices = {1, 2, 3, 3};
    g.weights     = {0.5f, 2.0f, 0.5f, 0.5f};
    return g;
}

TEST_CASE("cpu_delta_stepping: simple chain", "[delta_stepping_cpu]") {
    auto g = small_graph();
    auto cert = delta_stepping_cpu(g, 0, 1.0f, true);
    REQUIRE(cert.d[0] == Catch::Approx(0.0f));
    REQUIRE(cert.d[1] == Catch::Approx(1.0f));
    REQUIRE(cert.d[2] == Catch::Approx(2.0f));  // via 1, not direct 3.5
}

TEST_CASE("cpu_delta_stepping: matches dijkstra on chain", "[delta_stepping_cpu]") {
    auto g    = small_graph();
    auto dijk = dijkstra_cpu(g, 0);
    auto delt = delta_stepping_cpu(g, 0, 0.5f, false);
    for (vid_t v = 0; v < g.n_vertices; ++v)
        REQUIRE(dijk.d[v] == Catch::Approx(delt.d[v]).margin(1e-5f));
}

TEST_CASE("cpu_delta_stepping: heavy edge Phase B", "[delta_stepping_cpu]") {
    auto g    = mixed_graph();
    auto cert = delta_stepping_cpu(g, 0, 1.0f, true);
    // d[3] = min(d[1]+0.5, d[2]+0.5) = min(1.0, 2.5) = 1.0
    REQUIRE(cert.d[0] == Catch::Approx(0.0f));
    REQUIRE(cert.d[1] == Catch::Approx(0.5f));
    REQUIRE(cert.d[2] == Catch::Approx(2.0f));
    REQUIRE(cert.d[3] == Catch::Approx(1.0f));
}

TEST_CASE("cpu_delta_stepping: auto delta (0 → average)", "[delta_stepping_cpu]") {
    auto g    = small_graph();
    auto cert = delta_stepping_cpu(g, 0, 0.0f, false);  // delta auto
    REQUIRE(cert.d[2] == Catch::Approx(2.0f));
}

TEST_CASE("cpu_delta_stepping: unreachable vertex stays INF", "[delta_stepping_cpu]") {
    // Add isolated vertex 3
    CSR<float> g = small_graph();
    g.n_vertices = 4;
    g.row_offsets.push_back(g.row_offsets.back());  // no outgoing from 3
    auto cert = delta_stepping_cpu(g, 0, 1.0f, false);
    REQUIRE(cert.d[3] >= Sentinel<float>::inf / 2.0f);
}

TEST_CASE("reconstruct_pi: basic correctness", "[reconstruct_pi]") {
    auto g    = small_graph();
    auto cert = delta_stepping_cpu(g, 0, 1.0f, false);
    // emit_cert=false → pi is all INVALID_VID
    std::vector<vid_t> pi;
    reconstruct_pi(g, 0, cert.d, pi);
    // pi[2] should be 1 (path 0→1→2)
    REQUIRE(pi[0] == INVALID_VID);  // source has no predecessor
    REQUIRE(pi[1] == 0);
    REQUIRE(pi[2] == 1);
}
