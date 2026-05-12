#include <catch2/catch_test_macros.hpp>
#include "../src/core/csr_sssp.h"

TEST_CASE("CSR: edge_weight finds existing edges") {
    CSR<float> g;
    g.n_vertices = 3; g.n_edges = 3;
    g.row_offsets = {0, 2, 3, 3};
    g.col_indices = {1, 2, 2};
    g.weights     = {1.5f, 2.5f, 3.5f};
    g.sort_neighbors();
    REQUIRE(g.edge_weight(0, 1) == 1.5f);
    REQUIRE(g.edge_weight(0, 2) == 2.5f);
    REQUIRE(g.edge_weight(1, 2) == 3.5f);
}

TEST_CASE("CSR: edge_weight returns inf for missing edges") {
    CSR<float> g;
    g.n_vertices = 3; g.n_edges = 1;
    g.row_offsets = {0, 1, 1, 1};
    g.col_indices = {2};
    g.weights     = {1.0f};
    g.sort_neighbors();
    REQUIRE(g.edge_weight(0, 1) == Sentinel<float>::inf);
    REQUIRE(g.edge_weight(1, 2) == Sentinel<float>::inf);
}

TEST_CASE("CSR: degree") {
    CSR<float> g;
    g.n_vertices = 3; g.n_edges = 3;
    g.row_offsets = {0, 2, 3, 3};
    g.col_indices = {1, 2, 2};
    g.weights     = {1.0f, 2.0f, 3.0f};
    REQUIRE(g.degree(0) == 2);
    REQUIRE(g.degree(1) == 1);
    REQUIRE(g.degree(2) == 0);
}

TEST_CASE("CSR: sort_neighbors produces sorted adjacency") {
    CSR<float> g;
    g.n_vertices = 2; g.n_edges = 3;
    g.row_offsets = {0, 3, 3};
    g.col_indices = {2, 0, 1};  // unsorted
    g.weights     = {3.0f, 1.0f, 2.0f};
    g.sort_neighbors();
    // After sort: 0, 1, 2
    REQUIRE(g.col_indices[0] == 0);
    REQUIRE(g.col_indices[1] == 1);
    REQUIRE(g.col_indices[2] == 2);
    REQUIRE(g.weights[0] == 1.0f);
    REQUIRE(g.weights[1] == 2.0f);
    REQUIRE(g.weights[2] == 3.0f);
}
