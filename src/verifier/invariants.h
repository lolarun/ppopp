#pragma once
#include "../core/csr_sssp.h"
#include <optional>
#include <chrono>

enum class Verdict {
    SAT,
    UNSAT_SOURCE_DISTANCE,
    UNSAT_SOURCE_PRED,
    UNSAT_RELAXATION,
    UNSAT_UNREACHABLE_PRED,
    UNSAT_REACHABLE_NO_PRED,
    UNSAT_PRED_NOT_NEIGHBOR,
    UNSAT_PRED_DISTANCE_MISMATCH,
    UNSAT_CYCLE,
    UNSAT_DISCONNECTED_TREE,
};

inline const char* verdict_str(Verdict v) {
    switch (v) {
        case Verdict::SAT:                       return "SAT";
        case Verdict::UNSAT_SOURCE_DISTANCE:     return "UNSAT_SOURCE_DISTANCE";
        case Verdict::UNSAT_SOURCE_PRED:         return "UNSAT_SOURCE_PRED";
        case Verdict::UNSAT_RELAXATION:          return "UNSAT_RELAXATION";
        case Verdict::UNSAT_UNREACHABLE_PRED:    return "UNSAT_UNREACHABLE_PRED";
        case Verdict::UNSAT_REACHABLE_NO_PRED:   return "UNSAT_REACHABLE_NO_PRED";
        case Verdict::UNSAT_PRED_NOT_NEIGHBOR:   return "UNSAT_PRED_NOT_NEIGHBOR";
        case Verdict::UNSAT_PRED_DISTANCE_MISMATCH: return "UNSAT_PRED_DISTANCE_MISMATCH";
        case Verdict::UNSAT_CYCLE:               return "UNSAT_CYCLE";
        case Verdict::UNSAT_DISCONNECTED_TREE:   return "UNSAT_DISCONNECTED_TREE";
    }
    return "UNKNOWN";
}

struct VerifyResult {
    Verdict              verdict;
    std::optional<vid_t> witness_vertex;
    std::optional<eid_t> witness_edge;
    double               wall_time_ms{0.0};

    bool sat() const { return verdict == Verdict::SAT; }
};
