#!/usr/bin/env bash
# Reproducible official ASVspoof 2021 LA evaluation entrypoint (Phase 0.4).
#
# Pins the data path, decoder, clip length, and metrics so any checkpoint is
# scored identically. Emits the official-eval JSON and caches raw scores so
# minDCF + bootstrap CIs (scripts/bootstrap_ci.py) recompute without re-running
# inference.
#
#   scripts/eval_official.sh <run_name>
#   e.g. scripts/eval_official.sh xlsr_aasist_v9c
#
# Override locations via env: VG_CKPT_ROOT, VG_FLAC_DIR, VG_RUNS_DIR.
set -euo pipefail

RUN="${1:?usage: eval_official.sh <run_name> (e.g. xlsr_aasist_v9c)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_ROOT="${VG_CKPT_ROOT:-/srv/thabet/voiceguard-checkpoints}"
RUNS_DIR="${VG_RUNS_DIR:-$CKPT_ROOT/runs}"
FLAC_DIR="${VG_FLAC_DIR:-$CKPT_ROOT/asvspoof2021_LA_official/ASVspoof2021_LA_eval/flac}"

CKPT="$RUNS_DIR/$RUN/model_best.pt"
CONFIG="$RUNS_DIR/$RUN/config.json"
OUT="$RUNS_DIR/official_${RUN}.json"
SCORES="$RUNS_DIR/scores_${RUN}_official.npz"

[ -f "$CKPT" ]   || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[ -d "$FLAC_DIR" ] || { echo "missing flac dir: $FLAC_DIR" >&2; exit 1; }

# NVML stub (GPU access after Ollama); harmless if already present.
mkdir -p /tmp/nvml_fix
[ -f /tmp/nvml_fix/libnvidia-ml.so.1 ] || \
  gcc -shared -fPIC -o /tmp/nvml_fix/libnvidia-ml.so.1 \
      "$CKPT_ROOT/nvml_stub.c" 2>/dev/null || true

export LD_LIBRARY_PATH="/tmp/nvml_fix:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/src"

echo ">> Evaluating $RUN on the official ASVspoof 2021 LA protocol"
python3 "$CKPT_ROOT/run_official_eval.py" \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --flac-dir "$FLAC_DIR" --out "$OUT" --save-scores "$SCORES"

echo ">> Bootstrap 95% CIs + corrected minDCF"
python3 "$REPO/scripts/bootstrap_ci.py" "$SCORES" --n 1000 \
    --out "$RUNS_DIR/ci_${RUN}_official.json"

echo ">> Done. JSON: $OUT  |  scores: $SCORES"
