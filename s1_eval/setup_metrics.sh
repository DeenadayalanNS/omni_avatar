#!/usr/bin/env bash
# setup_metrics.sh — install the PINNED evaluation stack for score_metrics.py.
#
# Installs:
#   * SyncNet (joonson/syncnet_python) at a fixed commit + its pretrained weights
#     -> Sync-C (LSE-C) and Sync-D (LSE-D)
#   * insightface buffalo_l (ArcFace) + onnxruntime
#     -> CSIM identity cosine
#
# Reproducibility: the exact SyncNet commit used is written to
#   s1_eval/third_party/SYNCNET_COMMIT.txt
# Freeze that value (set SYNCNET_COMMIT below) so the STUDENT is scored with the
# identical metric code as the teacher baseline.
#
# Usage:
#   bash s1_eval/setup_metrics.sh
#   ONNX_CPU=1 bash s1_eval/setup_metrics.sh    # CPU-only scoring box
set -euo pipefail

PY="${PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/s1_eval/third_party"
SYNC_DIR="$TP/syncnet_python"

# Pin here for strict reproducibility. Empty = use current HEAD and record it.
SYNCNET_COMMIT="${SYNCNET_COMMIT:-}"

mkdir -p "$TP"

# --- 1. CSIM stack ------------------------------------------------------------
echo "==> Installing CSIM stack (insightface + onnxruntime)"
if [ "${ONNX_CPU:-0}" = "1" ]; then
    $PY -m pip install onnxruntime
else
    $PY -m pip install onnxruntime-gpu || $PY -m pip install onnxruntime
fi
$PY -m pip install insightface opencv-python

# --- 2. SyncNet ---------------------------------------------------------------
echo "==> Installing SyncNet dependencies"
$PY -m pip install scenedetect python_speech_features

if [ ! -d "$SYNC_DIR/.git" ]; then
    echo "==> Cloning syncnet_python"
    git clone https://github.com/joonson/syncnet_python "$SYNC_DIR"
fi

cd "$SYNC_DIR"
git fetch --all --quiet || true
if [ -n "$SYNCNET_COMMIT" ]; then
    echo "==> Checking out pinned commit $SYNCNET_COMMIT"
    git checkout --quiet "$SYNCNET_COMMIT"
fi
RESOLVED="$(git rev-parse HEAD)"
echo "$RESOLVED" > "$TP/SYNCNET_COMMIT.txt"
echo "==> SyncNet commit: $RESOLVED  (recorded in third_party/SYNCNET_COMMIT.txt)"

echo "==> Downloading SyncNet + S3FD weights"
if [ -f "download_model.sh" ]; then
    bash download_model.sh
fi

if [ ! -f "$SYNC_DIR/data/syncnet_v2.model" ]; then
    echo "!! SyncNet weights missing at data/syncnet_v2.model"
    echo "   Check download_model.sh output above (the VGG host is occasionally down)."
    exit 1
fi

echo ""
echo "==> Metrics stack ready."
echo "    Sync-C/Sync-D : $SYNC_DIR (commit $RESOLVED)"
echo "    CSIM          : insightface buffalo_l"
echo ""
echo "    Test:  $PY s1_eval/score_metrics.py <videos_dir> --pairs baseline_pairs.txt --save-baseline"
