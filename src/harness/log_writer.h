#pragma once
#include "../core/csr_sssp.h"
#include "../verifier/invariants.h"
#include <string>
#include <fstream>
#include <chrono>

struct RunConfig {
    std::string dataset_path;
    std::string dataset_name;
    std::string algo;           // "dijkstra_cpu" | "delta_stepping_cpu" | "delta_stepping_gpu"
    std::string precision;      // "fp32" | "fp64"
    int         gpu_id{0};
    vid_t       source{0};
    float       delta{-1};      // -1 = heuristic (avg edge weight)
    bool        emit_cert{true};
    bool        verify{true};
    int         seed{42};
    int         reps{3};
    std::string output_jsonl;

    // Weight distribution remapping (E9 stress test)
    std::string weight_dist;    // "uniform" (default) | "gaussian" | "powerlaw" | "adversarial"

    // RMAT in-memory generation (E8 scaling; replaces dataset_path)
    std::string rmat_scale_str; // empty = not used; non-empty = 2^N vertices
    int         rmat_edgefactor{32};
    int         rmat_seed{42};

    // Binary CSR save path (--save-csr=; saves graph then exits)
    std::string save_csr_path;

    // Cert binary save prefix (--save-cert=; writes <prefix>.d.bin + .pi.bin
    // each rep, used for cross-platform drift comparison in E1)
    std::string save_cert_prefix;

    // Verify-only mode (--verify-only --cert-prefix=<path>): skip SSSP,
    // load graph + cert binaries from <path>.d.bin + <path>.pi.bin, run
    // verifier only. Used by E4 error-injection harness to validate
    // perturbed certificates without recomputing SSSP.
    bool        verify_only{false};
    std::string verify_cert_prefix;
    // Batch verify-only mode (--cert-prefix-list=<file>): file contains one
    // cert prefix per line; load graph once, verify each cert, append JSONL
    // line per cert. Avoids per-cert process startup + graph parse overhead
    // (~60x faster than --verify-only loop on web_google).
    std::string verify_cert_list;
};

struct RunMetrics {
    double sssp_ms{0};
    double verifier_ms{0};
    double teps{0};
    VerifyResult vr{Verdict::SAT};
    std::string  d_hash;       // CRC32 hex of distance array
    std::string  pi_hash;      // CRC32 hex of pi array
    uint64_t     n_vertices{0}; // for dataset section in log
    uint64_t     n_edges{0};
    uint32_t     n_unreachable{0};
    std::string  gpu_name;      // hardware.gpu — populated by harness if detectable
};

class LogWriter {
public:
    explicit LogWriter(const std::string& path);
    ~LogWriter();

    void write_run(const RunConfig& cfg,
                   int              rep,
                   uint64_t         n_edges,
                   const RunMetrics& m);
private:
    std::ofstream f_;
    std::string   git_commit_;
    std::string   timestamp_session_;
    std::string   gpu_name_;       // detected once at construction

    static std::string git_head();
    static std::string now_iso8601();
};
