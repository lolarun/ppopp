#include "log_writer.h"
#include "../common/crc32.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <ctime>
#include <array>
#include <numeric>
#include <cstring>
#include <limits>

using json = nlohmann::json;

// ── GPU detection (best-effort, no error if unavailable) ─────────────────────
static std::string detect_gpu() {
    // Try nvidia-smi first
    FILE* p = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (p) {
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), p)) {
            pclose(p);
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                s.pop_back();
            if (!s.empty()) return s;
        } else {
            pclose(p);
        }
    }
    // Try rocm-smi
    p = popen("rocm-smi --showproductname 2>/dev/null | grep -v '^$' | head -1", "r");
    if (p) {
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), p)) {
            pclose(p);
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                s.pop_back();
            if (!s.empty()) return s;
        } else {
            pclose(p);
        }
    }
    return "cpu-only";
}

// ── LogWriter ────────────────────────────────────────────────────────────────

std::string LogWriter::git_head() {
    FILE* p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!p) return "unknown";
    char buf[32] = {};
    if (fgets(buf, sizeof(buf), p) == nullptr) buf[0] = '\0';
    pclose(p);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s.empty() ? "unknown" : s;
}

std::string LogWriter::now_iso8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32]; strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    return buf;
}

LogWriter::LogWriter(const std::string& path)
    : f_(path, std::ios::app)
    , git_commit_(git_head())
    , timestamp_session_(now_iso8601())
    , gpu_name_(detect_gpu())
{
    if (!f_.is_open())
        throw std::runtime_error("Cannot open log file: " + path);
}

LogWriter::~LogWriter() { f_.flush(); }

void LogWriter::write_run(const RunConfig& cfg, int rep,
                           uint64_t n_edges, const RunMetrics& m)
{
    json j;

    // ── Unified top-level fields (shared with PageRank schema) ──────────
    j["algorithm"]        = cfg.algo;
    j["gpu"]              = m.gpu_name.empty() ? gpu_name_ : m.gpu_name;
    j["precision"]        = cfg.precision;
    j["wall_ms"]          = m.sssp_ms;
    j["output_crc32"]     = m.d_hash;
    j["converged"]        = m.vr.sat();
    j["iterations"]       = rep;
    j["N"]                = m.n_vertices;
    j["E"]                = m.n_edges > 0 ? m.n_edges : n_edges;

    // ── Legacy fields (backward compat with analysis scripts) ───────────
    j["timestamp"]        = now_iso8601();
    j["git_commit"]       = git_commit_;
    j["rep"]              = rep;
    j["sssp_ms"]          = m.sssp_ms;
    j["verifier_ms"]      = m.verifier_ms;
    j["teps"]             = m.teps;
    j["verifier_verdict"] = verdict_str(m.vr.verdict);

    j["hardware"] = {
        {"gpu", m.gpu_name.empty() ? gpu_name_ : m.gpu_name},
    };

    j["dataset"] = {
        {"name", cfg.dataset_name},
        {"n_v",  m.n_vertices},
        {"n_e",  m.n_edges > 0 ? m.n_edges : n_edges},
    };

    j["config"] = {
        {"algo",            cfg.algo},
        {"precision",       cfg.precision},
        {"source",          cfg.source},
        {"delta",           cfg.delta},
        {"emit_cert",       cfg.emit_cert},
        {"verify",          cfg.verify},
        {"seed",            cfg.seed},
        {"weight_dist",     cfg.weight_dist},
        {"rmat_scale",      cfg.rmat_scale_str},
        {"rmat_edgefactor", cfg.rmat_edgefactor},
        {"rmat_seed",       cfg.rmat_seed},
    };

    j["cert_summary"] = {
        {"d_hash",       m.d_hash},
        {"pi_hash",      m.pi_hash},
        {"n_unreachable", m.n_unreachable},
    };

    if (!m.vr.sat() && m.vr.witness_vertex.has_value()) {
        j["witness"] = {{"vertex", *m.vr.witness_vertex}};
    } else {
        j["witness"] = nullptr;
    }

    f_ << j.dump() << "\n";
    f_.flush();
}
