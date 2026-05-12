#include <catch2/catch_test_macros.hpp>
#include "../src/core/csr_sssp.h"
#include "../src/sssp/cpu_dijkstra.h"

static CSR<float> chain(int n) {
    CSR<float> g;
    g.n_vertices = n;
    g.n_edges    = n - 1;
    g.row_offsets.resize(n + 1, 0);
    g.col_indices.resize(n - 1);
    g.weights.resize(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        g.row_offsets[i + 1] = i + 1;
        g.col_indices[i] = i + 1;
        g.weights[i]     = 1.0f;
    }
    g.row_offsets[n] = n - 1;
    return g;
}

TEST_CASE("dijkstra: chain graph distances") {
    auto g = chain(5);
    auto cert = dijkstra_cpu(g, 0);
    for (int i = 0; i < 5; ++i)
        REQUIRE(cert.d[i] == (float)i);
}

TEST_CASE("dijkstra: shortest path respects weights") {
    // 0→1 cost 10, 0→2 cost 1, 2→1 cost 1 (total 2 < 10)
    CSR<float> g;
    g.n_vertices = 3; g.n_edges = 3;
    g.row_offsets = {0, 2, 2, 3};
    g.col_indices = {1, 2, 1};
    g.weights     = {10.0f, 1.0f, 1.0f};
    g.sort_neighbors();
    auto cert = dijkstra_cpu(g, 0);
    REQUIRE(cert.d[1] == 2.0f);
    REQUIRE(cert.pi[1] == 2);  // via vertex 2
}

TEST_CASE("dijkstra: fp32 precision on equal paths") {
    // Two paths of exact same cost → tiebreak by smaller vid
    CSR<float> g;
    g.n_vertices = 4; g.n_edges = 4;
    g.row_offsets = {0, 2, 3, 3, 4};
    g.col_indices = {1, 2, 3, 3};
    g.weights     = {1.0f, 1.0f, 1.0f, 1.0f};
    g.sort_neighbors();
    auto cert = dijkstra_cpu(g, 0);
    REQUIRE(cert.d[3] == 2.0f);
    REQUIRE(cert.pi[3] == 1);  // tiebreak: 1 < 2
}
