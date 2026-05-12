#!/usr/bin/env bash
# One-click reproduction of ALL paper experiments (PageRank + SSSP).
#
# Usage (run on each GPU host):
#   GPU_BACKEND=CUDA CUDA_ARCH=86  DEVICE_TAG=a10_sm86   ./scripts/run_full_paper.sh
#   GPU_BACKEND=CUDA CUDA_ARCH=75  DEVICE_TAG=t4_sm75    ./scripts/run_full_paper.sh
#   GPU_BACKEND=ROCM HIP_ARCH=gfx942 DEVICE_TAG=mi300x_vf ./scripts/run_full_paper.sh
#
# After running on ALL hosts, collect results/ together and run:
#   ./scripts/compare_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

GPU_BACKEND="${GPU_BACKEND:?Set GPU_BACKEND=CUDA|ROCM}"
DEVICE_TAG="${DEVICE_TAG:?Set DEVICE_TAG (e.g. a10_sm86, t4_sm75, mi300x_vf)}"

echo "================================================================"
echo "  FULL PAPER REPRODUCTION: $DEVICE_TAG ($GPU_BACKEND)"
echo "================================================================"
echo

# ── 1. Verify data integrity ────────────────────────────────────────────────
if [[ -f "$ROOT/data/csr_anchors.txt" ]]; then
    echo "[1/5] Verifying CSR data integrity..."
    (cd "$ROOT/data/cache" && sha256sum -c "$ROOT/data/csr_anchors.txt")
    echo
fi

# ── 2. PageRank experiments ─────────────────────────────────────────────────
echo "[2/5] PageRank — push + pull + pull_v2 (fp32 + fp64)"
for prec in fp32 fp64; do
    echo "  --- precision=$prec ---"
    PRECISION=$prec "$ROOT/scripts/run_re0.sh"
    PRECISION=$prec "$ROOT/scripts/run_re2e_re3.sh"
    PRECISION=$prec "$ROOT/scripts/run_re2e_v2.sh"
done
echo

# ── 3. SSSP experiments ────────────────────────────────────────────────────
echo "[3/5] SSSP — E1-E9 phases"
echo "  (SSSP scripts may need RESULTS_ROOT and data path adjustment for this host)"

RESULTS_ROOT="$ROOT/results/sssp" bash "$ROOT/scripts/sssp/run_phase1.sh" || \
    echo "  [WARN] run_phase1.sh returned non-zero (check output above)"

RESULTS_ROOT="$ROOT/results/sssp" bash "$ROOT/scripts/sssp/run_phases_2_to_5.sh" || \
    echo "  [WARN] run_phases_2_to_5.sh returned non-zero (check output above)"
echo

# ── 4. Evidence metadata ───────────────────────────────────────────────────
echo "[4/5] Capturing build evidence..."
EVIDENCE_DIR="$ROOT/results/pagerank/${DEVICE_TAG}/_evidence"
mkdir -p "$EVIDENCE_DIR"
{
    echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "device_tag: $DEVICE_TAG"
    echo "gpu_backend: $GPU_BACKEND"
    echo "git_commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia_driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
        echo "cuda_version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/gpu_info: /' || true
    fi
    if command -v rocminfo &>/dev/null; then
        echo "rocm_version: $(cat /opt/rocm/.info/version 2>/dev/null || echo n/a)"
        rocminfo 2>/dev/null | grep -E 'Marketing Name|Compute Unit' | head -4 | sed 's/^/gpu_info: /' || true
    fi
} > "$EVIDENCE_DIR/build_metadata.txt"
echo

# ── 5. Summary ─────────────────────────────────────────────────────────────
echo "[5/5] Done on $DEVICE_TAG."
echo
echo "Next steps:"
echo "  1. scp results/ back to local"
echo "  2. After collecting results from ALL hosts, run:"
echo "     ./scripts/compare_all.sh"
echo "  3. Validate against legacy/results/ for reproducibility"
