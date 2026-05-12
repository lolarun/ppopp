#include "../core/csr_sssp.h"
#include "../io/dimacs_loader.h"
#include "../io/csr_io.h"
#include "../io/rmat_generator.h"
#include "../sssp/cpu_dijkstra.h"
#include "../sssp/cpu_delta_stepping.h"
#include "../verifier/cpu_verifier.h"
#include "log_writer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <random>
#include <algorithm>

// Forward declarations for GPU path (compiled separately when GPU_BACKEND set)
#if defined(USE_GPU)
#include "../sssp/delta_stepping.h"
#include "../sssp/bellman_ford.h"
#include "../sssp/async_push_sssp.h"
#endif

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --dataset=<path>          .gr/.csr file (or omit for --rmat-scale)\n"
        "  --dataset-name=<name>     label for logging\n"
        "  --algo=<dijkstra_cpu|delta_stepping_cpu|delta_stepping_gpu|bellman_ford_gpu|async_push_sssp_gpu>\n"
        "  --precision=<fp32|fp64>   (default fp32)\n"
        "  --source=<vid>            (default 0)\n"
        "  --delta=<float>           bucket width; -1 = avg edge weight (default)\n"
        "  --emit-cert=<0|1>         emit predecessor certificate (default 1)\n"
        "  --verify=<0|1>            run verifier after SSSP (default 1)\n"
        "  --reps=<n>                repetitions (default 3)\n"
        "  --seed=<n>                random seed (default 42)\n"
        "  --output=<path.jsonl>     log file (default results.jsonl)\n"
        "  --threads=<n>             verifier OMP threads (default: env)\n"
        "  --weight-dist=<uniform|gaussian|powerlaw|adversarial>  (default uniform)\n"
        "  --rmat-scale=<n>          generate in-memory RMAT graph (2^n vertices)\n"
        "  --rmat-edgefactor=<n>     edges per vertex for RMAT (default 32)\n"
        "  --rmat-seed=<n>           RMAT generator seed (default 42)\n"
        "  --save-csr=<path>         after loading/generating, save .csr then exit\n"
        "  --save-cert=<prefix>      dump cert.d/cert.pi binaries each rep (E1 drift)\n"
        "  --verify-only=1           load <cert-prefix>.{d,pi}.bin, run verifier, exit (E4)\n"
        "  --cert-prefix=<path>      cert binary prefix for --verify-only mode\n",
        prog);
}

static std::string get_arg(int argc, char** argv, const char* key, const char* def) {
    std::string prefix = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], prefix.c_str(), prefix.size()) == 0)
            return argv[i] + prefix.size();
    return def ? def : "";
}

// Apply weight distribution remapping to a loaded graph.
// Used by --weight-dist= flag (E9 stress test).
//
// Empty dist string = no remap (use loaded DIMACS/RMAT weights as-is).
// "uniform" = seed-deterministic uniform[1e-4, 1.0] PRNG remap (true PRNG, not no-op).
// "gaussian" = seed-deterministic clipped-normal PRNG remap.
// "powerlaw" = approximate power-law in [0, 1] via u^2 transform.
// "adversarial" = all weights set to 0.5 (maximum FP ties).
template<typename W>
static void remap_weights(CSR<W>& g, const std::string& dist, unsigned seed) {
    if (dist.empty()) return;  // no remap; use loaded weights
    std::mt19937 rng(seed);
    if (dist == "uniform") {
        // Seed-deterministic uniform[1e-4, 1.0] FP remap. Bug fix 2026-05-02:
        // previously this branch returned without modifying weights, making
        // --weight-dist=uniform a silent no-op duplicate of the empty default.
        std::uniform_real_distribution<double> ud(1e-4, 1.0);
        for (auto& w : g.weights)
            w = (W)ud(rng);
    } else if (dist == "gaussian") {
        std::normal_distribution<double> nd(0.5, 0.2);
        for (auto& w : g.weights)
            w = (W)std::max(1e-4, std::min(1.0, nd(rng)));
    } else if (dist == "powerlaw") {
        // power-law: sample u ~ U[0,1], w = u^(-1/(alpha-1)), alpha=2 → w = 1/u
        std::uniform_real_distribution<double> ud(1e-4, 1.0);
        for (auto& w : g.weights) {
            double u = ud(rng);
            w = (W)std::min(1.0, u * u);  // approximate power-law in [0,1]
        }
    } else if (dist == "adversarial") {
        // adversarial: all weights identical → maximum ties → maximises FP sensitivity
        for (auto& w : g.weights) w = (W)0.5;
    }
    // unknown dist: leave weights unchanged
}

template<typename W>
static CSR<W> load_graph(const RunConfig& cfg) {
    // RMAT in-memory generation
    if (!cfg.rmat_scale_str.empty()) {
        RMATConfig rc;
        rc.scale      = std::stoi(cfg.rmat_scale_str);
        rc.edgefactor = cfg.rmat_edgefactor;
        rc.seed       = (unsigned)cfg.rmat_seed;
        return generate_rmat<W>(rc);
    }
    // Binary CSR
    const std::string& p = cfg.dataset_path;
    if (p.size() > 4 && p.substr(p.size() - 4) == ".csr")
        return read_csr<W>(p.c_str());
    // DIMACS .gr
    return load_dimacs<W>(p.c_str());
}

template<typename W>
static void run(const RunConfig& base_cfg, int num_threads) {
    CSR<W> g = load_graph<W>(base_cfg);
    remap_weights(g, base_cfg.weight_dist, (unsigned)base_cfg.seed);
    g.sort_neighbors();   // required for binary-search in verifier

    // If --save-csr was requested, write and exit
    if (!base_cfg.save_csr_path.empty()) {
        write_csr(g, base_cfg.save_csr_path.c_str());
        fprintf(stderr, "Saved CSR to %s\n", base_cfg.save_csr_path.c_str());
        return;
    }

    // Batch verify mode: graph loaded once, verify each cert in list.
    if (!base_cfg.verify_cert_list.empty()) {
        std::ifstream listf(base_cfg.verify_cert_list);
        if (!listf) {
            fprintf(stderr, "Error: cannot open cert list %s\n",
                    base_cfg.verify_cert_list.c_str());
            return;
        }
        LogWriter log(base_cfg.output_jsonl);
        std::string prefix;
        Certificate<W> cert;
        cert.d.resize(g.n_vertices);
        cert.pi.resize(g.n_vertices);
        int n_processed = 0;
        while (std::getline(listf, prefix)) {
            if (prefix.empty()) continue;
            std::ifstream din(prefix + ".d.bin",  std::ios::binary);
            std::ifstream pin(prefix + ".pi.bin", std::ios::binary);
            if (!din || !pin) {
                fprintf(stderr, "Error: cannot open cert %s\n", prefix.c_str());
                continue;
            }
            din.read(reinterpret_cast<char*>(cert.d.data()),
                     cert.d.size() * sizeof(W));
            pin.read(reinterpret_cast<char*>(cert.pi.data()),
                     cert.pi.size() * sizeof(vid_t));
            auto t0 = std::chrono::steady_clock::now();
            VerifyResult vr = verify<W>(g, base_cfg.source,
                                        std::span<const W>(cert.d),
                                        std::span<const vid_t>(cert.pi),
                                        num_threads);
            auto t1 = std::chrono::steady_clock::now();
            double verifier_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            RunConfig cfg_log = base_cfg;
            cfg_log.verify_cert_prefix = prefix;
            RunMetrics m;
            m.sssp_ms = 0;
            m.verifier_ms = verifier_ms;
            m.vr = vr;
            m.n_vertices = g.n_vertices;
            m.n_edges = g.n_edges;
            log.write_run(cfg_log, 0, g.n_edges, m);
            ++n_processed;
            if (n_processed % 50 == 0)
                fprintf(stderr, "[batch-verify] processed %d certs\n", n_processed);
        }
        fprintf(stderr, "[batch-verify] done: %d certs processed\n", n_processed);
        return;
    }

    // Verify-only mode: load cert binaries instead of computing SSSP.
    // Used by E4 to verify perturbed/injected certificates.
    if (base_cfg.verify_only) {
        if (base_cfg.verify_cert_prefix.empty()) {
            fprintf(stderr, "Error: --verify-only requires --cert-prefix=<path>\n");
            return;
        }
        Certificate<W> cert;
        cert.d.resize(g.n_vertices);
        cert.pi.resize(g.n_vertices);
        std::string df = base_cfg.verify_cert_prefix + ".d.bin";
        std::string pf = base_cfg.verify_cert_prefix + ".pi.bin";
        std::ifstream din(df, std::ios::binary);
        std::ifstream pin(pf, std::ios::binary);
        if (!din || !pin) {
            fprintf(stderr, "Error: cannot open %s or %s\n", df.c_str(), pf.c_str());
            return;
        }
        din.read(reinterpret_cast<char*>(cert.d.data()), cert.d.size() * sizeof(W));
        pin.read(reinterpret_cast<char*>(cert.pi.data()), cert.pi.size() * sizeof(vid_t));
        if (din.gcount() != (std::streamsize)(cert.d.size() * sizeof(W))) {
            fprintf(stderr, "Error: %s size mismatch (expected %zu bytes)\n",
                    df.c_str(), cert.d.size() * sizeof(W));
            return;
        }
        auto t0 = std::chrono::steady_clock::now();
        VerifyResult vr = verify<W>(g, base_cfg.source,
                                    std::span<const W>(cert.d),
                                    std::span<const vid_t>(cert.pi),
                                    num_threads);
        auto t1 = std::chrono::steady_clock::now();
        double verifier_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Emit a single JSONL line with verdict + witness
        LogWriter log(base_cfg.output_jsonl);
        RunMetrics m;
        m.sssp_ms = 0;
        m.verifier_ms = verifier_ms;
        m.vr = vr;
        m.n_vertices = g.n_vertices;
        m.n_edges = g.n_edges;
        log.write_run(base_cfg, /*rep=*/0, g.n_edges, m);
        fprintf(stderr, "[verify-only] cert=%s verifier=%.1fms verdict=%s\n",
                base_cfg.verify_cert_prefix.c_str(), verifier_ms,
                verdict_str(vr.verdict));
        return;
    }

    // Compute delta: use CLI value if positive, else avg edge weight
    W delta;
    if (base_cfg.delta > 0) {
        delta = W(base_cfg.delta);
    } else {
        W sum = W{0};
        for (auto w : g.weights) sum += w;
        delta = sum / W(g.n_edges > 0 ? g.n_edges : 1);
        if (delta <= W{0}) delta = W{1};
    }

    LogWriter log(base_cfg.output_jsonl);

    for (int rep = 0; rep < base_cfg.reps; ++rep) {
        Certificate<W> cert;
        double sssp_ms = 0;

        auto t0 = std::chrono::steady_clock::now();

        if (base_cfg.algo == "dijkstra_cpu") {
            cert = dijkstra_cpu<W>(g, base_cfg.source);
        } else if (base_cfg.algo == "delta_stepping_cpu") {
            cert = delta_stepping_cpu<W>(g, base_cfg.source, delta,
                                         base_cfg.emit_cert);
        }
#if defined(USE_GPU)
        else if (base_cfg.algo == "delta_stepping_gpu") {
            cert = delta_stepping_gpu<W>(g, base_cfg.source, delta,
                                         base_cfg.emit_cert);
        }
        else if (base_cfg.algo == "bellman_ford_gpu") {
            cert = bellman_ford_gpu<W>(g, base_cfg.source, base_cfg.emit_cert);
        }
        else if (base_cfg.algo == "async_push_sssp_gpu") {
            cert = async_push_sssp_gpu<W>(g, base_cfg.source, base_cfg.emit_cert);
        }
#endif
        else {
            throw std::runtime_error("Unknown algo: " + base_cfg.algo);
        }

        auto t1 = std::chrono::steady_clock::now();
        sssp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        RunMetrics m;
        m.sssp_ms    = sssp_ms;
        m.teps       = (double)g.n_edges / (sssp_ms * 1e-3);
        m.n_vertices = g.n_vertices;
        m.n_edges    = g.n_edges;

        if (base_cfg.verify) {
            VerifyResult vr = verify<W>(g, base_cfg.source,
                                        std::span<const W>(cert.d),
                                        std::span<const vid_t>(cert.pi),
                                        num_threads);
            m.verifier_ms = vr.wall_time_ms;
            m.vr          = vr;
        }

        // Count unreachable vertices
        {
            uint32_t n_unr = 0;
            for (const W& dv : cert.d)
                if (dv >= Sentinel<W>::inf / W{2}) ++n_unr;
            m.n_unreachable = n_unr;
        }

        // CRC32 hashes for cross-platform byte-diff detection
        {
            uint32_t crc = 0xFFFFFFFFu;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(cert.d.data());
            for (size_t i = 0; i < cert.d.size() * sizeof(W); ++i) {
                crc ^= p[i];
                for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320u : 0);
            }
            crc ^= 0xFFFFFFFFu;
            char buf[9]; snprintf(buf, sizeof(buf), "%08x", crc);
            m.d_hash = buf;
        }
        {
            uint32_t crc = 0xFFFFFFFFu;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(cert.pi.data());
            for (size_t i = 0; i < cert.pi.size() * sizeof(vid_t); ++i) {
                crc ^= p[i];
                for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320u : 0);
            }
            crc ^= 0xFFFFFFFFu;
            char buf[9]; snprintf(buf, sizeof(buf), "%08x", crc);
            m.pi_hash = buf;
        }

        log.write_run(base_cfg, rep, g.n_edges, m);

        // Save raw cert.d and cert.pi as binary for cross-platform drift compare
        if (!base_cfg.save_cert_prefix.empty()) {
            std::string p = base_cfg.save_cert_prefix;
            // append rep index only when reps > 1
            if (base_cfg.reps > 1) p += "_rep" + std::to_string(rep);
            std::ofstream df(p + ".d.bin",  std::ios::binary);
            std::ofstream pf(p + ".pi.bin", std::ios::binary);
            df.write(reinterpret_cast<const char*>(cert.d.data()),
                     cert.d.size() * sizeof(W));
            pf.write(reinterpret_cast<const char*>(cert.pi.data()),
                     cert.pi.size() * sizeof(vid_t));
        }

        fprintf(stderr, "[rep %d] sssp=%.1fms  verifier=%.1fms  verdict=%s  "
                "teps=%.2e  d_hash=%s\n",
                rep, m.sssp_ms, m.verifier_ms,
                verdict_str(m.vr.verdict), m.teps, m.d_hash.c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    RunConfig cfg;
    cfg.dataset_path  = get_arg(argc, argv, "dataset", "");
    cfg.dataset_name  = get_arg(argc, argv, "dataset-name", "unknown");
    cfg.algo          = get_arg(argc, argv, "algo", "dijkstra_cpu");
    cfg.precision     = get_arg(argc, argv, "precision", "fp32");
    cfg.source          = (vid_t)std::stoul(get_arg(argc, argv, "source", "0"));
    cfg.emit_cert       = get_arg(argc, argv, "emit-cert", "1") != "0";
    cfg.verify          = get_arg(argc, argv, "verify", "1") != "0";
    cfg.reps            = std::stoi(get_arg(argc, argv, "reps", "3"));
    cfg.seed            = std::stoi(get_arg(argc, argv, "seed", "42"));
    cfg.output_jsonl    = get_arg(argc, argv, "output", "results.jsonl");
    // Default empty string = no remap (use loaded DIMACS/RMAT weights as-is).
    // BACKWARDS COMPAT after the 2026-05-02 uniform-remap fix: pre-fix
    // `--weight-dist=` defaulted to "uniform" which silently fell through
    // to no-op; post-fix "uniform" applies a genuine PRNG remap. Changing
    // the default to "" preserves the historical "no-op when not specified"
    // behavior so existing scripts keep using the loaded weights.
    cfg.weight_dist     = get_arg(argc, argv, "weight-dist", "");
    cfg.rmat_scale_str  = get_arg(argc, argv, "rmat-scale", "");
    cfg.rmat_edgefactor = std::stoi(get_arg(argc, argv, "rmat-edgefactor", "32"));
    cfg.rmat_seed       = std::stoi(get_arg(argc, argv, "rmat-seed", "42"));
    cfg.save_csr_path     = get_arg(argc, argv, "save-csr", "");
    cfg.save_cert_prefix  = get_arg(argc, argv, "save-cert", "");
    cfg.verify_only       = get_arg(argc, argv, "verify-only", "0") != "0";
    cfg.verify_cert_prefix= get_arg(argc, argv, "cert-prefix", "");
    cfg.verify_cert_list  = get_arg(argc, argv, "cert-prefix-list", "");
    {
        std::string ds = get_arg(argc, argv, "delta", "-1");
        cfg.delta = std::stof(ds);
    }
    int threads = std::stoi(get_arg(argc, argv, "threads", "0"));

    bool have_dataset = !cfg.dataset_path.empty() || !cfg.rmat_scale_str.empty();
    if (!have_dataset) {
        fprintf(stderr, "Error: --dataset or --rmat-scale is required\n");
        usage(argv[0]);
        return 1;
    }

    if (cfg.precision == "fp64")
        run<double>(cfg, threads);
    else
        run<float>(cfg, threads);

    return 0;
}
