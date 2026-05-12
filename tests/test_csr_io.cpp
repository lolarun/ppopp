#include <catch2/catch_test_macros.hpp>
#include "io/csr_io.h"
#include "core/csr_sssp.h"
#include <filesystem>
#include <cstdio>

// Build a small CSR for round-trip tests
static CSR<float> make_test_graph() {
    CSR<float> g;
    g.n_vertices = 4;
    g.n_edges    = 5;
    g.row_offsets = {0, 2, 3, 5, 5};
    g.col_indices = {1, 2, 3, 0, 3};
    g.weights     = {0.5f, 1.5f, 2.0f, 0.1f, 0.9f};
    return g;
}

TEST_CASE("csr_io: float round-trip", "[csr_io]") {
    auto orig = make_test_graph();
    const char* path = "test_round_trip_f32.csr";

    write_csr(orig, path);
    auto loaded = read_csr<float>(path);
    std::remove(path);

    REQUIRE(loaded.n_vertices == orig.n_vertices);
    REQUIRE(loaded.n_edges    == orig.n_edges);
    REQUIRE(loaded.row_offsets == orig.row_offsets);
    REQUIRE(loaded.col_indices == orig.col_indices);
    REQUIRE(loaded.weights     == orig.weights);
}

TEST_CASE("csr_io: double round-trip", "[csr_io]") {
    CSR<double> g;
    g.n_vertices  = 3;
    g.n_edges     = 2;
    g.row_offsets = {0, 1, 2, 2};
    g.col_indices = {1, 2};
    g.weights     = {3.14159265358979, 2.71828182845905};

    const char* path = "test_round_trip_f64.csr";
    write_csr(g, path);
    auto loaded = read_csr<double>(path);
    std::remove(path);

    REQUIRE(loaded.n_vertices == g.n_vertices);
    REQUIRE(loaded.weights[0] == g.weights[0]);
    REQUIRE(loaded.weights[1] == g.weights[1]);
}

TEST_CASE("csr_io: precision tag mismatch throws", "[csr_io]") {
    auto g = make_test_graph();
    const char* path = "test_prec_mismatch.csr";
    write_csr(g, path);
    REQUIRE_THROWS_AS(read_csr<double>(path), std::runtime_error);
    std::remove(path);
}

TEST_CASE("csr_io: corrupted file throws (CRC)", "[csr_io]") {
    auto g = make_test_graph();
    const char* path = "test_crc_fail.csr";
    write_csr(g, path);

    // Corrupt one byte in the middle of the file
    FILE* fp = fopen(path, "r+b");
    fseek(fp, 30, SEEK_SET);
    uint8_t b = 0xFF;
    fwrite(&b, 1, 1, fp);
    fclose(fp);

    REQUIRE_THROWS_AS(read_csr<float>(path), std::runtime_error);
    std::remove(path);
}

TEST_CASE("csr_io: missing file throws", "[csr_io]") {
    REQUIRE_THROWS_AS(read_csr<float>("/nonexistent/path/file.csr"),
                      std::runtime_error);
}
