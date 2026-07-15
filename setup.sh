#!/usr/bin/env bash
# setup.sh — one-shot environment + weights setup for OmniAvatar on a RunPod pod.
#
# Usage:
#   bash setup.sh                 # 1.3B model (default)
#   bash setup.sh 14B             # 14B model
#   bash setup.sh both            # both models
#
# Env vars:
#   HF_TOKEN=hf_xxx   optional, faster/authenticated Hugging Face downloads
#   FLASH_ATTN=1      also build flash-attn (optional, slow to compile)
#   PY=python3        interpreter to use (default: python3)
#   PERSIST=1         install deps into a venv at /workspace/omni_venv so that
#                     STOP (not terminate) keeps EVERYTHING — deps, weights, code
#                     — and the next start needs NO reinstall. Run generate.py
#                     with /workspace/omni_venv/bin/python afterwards.
#
# Safe to re-run: pip is idempotent and hf download resumes/skips existing files.
set -euo pipefail

MODEL="${1:-1.3B}"
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Persistent venv on the /workspace volume: survives pod STOP/START (and network
# volumes), so deps are installed once and never again.
if [ "${PERSIST:-0}" = "1" ]; then
    VENV="${VENV:-/workspace/omni_venv}"
    if [ ! -x "$VENV/bin/python" ]; then
        echo "==> Creating persistent venv at $VENV"
        $PY -m venv "$VENV"
    fi
    PY="$VENV/bin/python"
    echo "==> Using persistent venv: $PY"
fi

echo "==> OmniAvatar setup | model=$MODEL | python=$($PY --version 2>&1)"

# --- 0. System deps -----------------------------------------------------------
# ffmpeg is required by the audio preprocessing (called as a system binary via
# subprocess, not the pip imageio-ffmpeg). RunPod base images often lack it.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "==> Installing ffmpeg (system)"
    apt-get update -qq && apt-get install -y -qq ffmpeg || \
        echo "!! could not apt-get ffmpeg — install it manually (apt-get install -y ffmpeg)"
fi

# --- 1. Python deps -----------------------------------------------------------
# Order matters. Install the CUDA torch stack first, then install requirements
# WITH constraints.txt so its transitive deps cannot upgrade/clobber torch
# (otherwise torch jumps to 2.13 from PyPI and breaks torchvision::nms).
echo "==> Installing PyTorch 2.4.0 (cu124)"
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing requirements (torch pinned via constraints.txt)"
# --ignore-installed blinker: RunPod/Ubuntu ships blinker via distutils, which pip
# cannot uninstall when flask (an xfuser dep) needs a newer blinker. Skip it.
$PY -m pip install -r requirements.txt -c constraints.txt --ignore-installed blinker
$PY -m pip install "huggingface_hub[cli]"

# Safety net: re-pin the exact CUDA torch build in case anything slipped past.
$PY -m pip install --no-deps --force-reinstall \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Verifying the generation stack imports"
# CUDA availability is checked but NOT fatal: a transient "CUDA unknown error"
# (common on fresh RunPod pods — fixed by restarting the pod) should not abort the
# weight download, which needs no GPU. It's re-checked at generate time.
$PY - <<'PYEOF' || echo "!! verify import step reported an issue (see above) — continuing"
import torch
from torchvision.ops import nms
from transformers import Wav2Vec2FeatureExtractor
from peft import LoraConfig, inject_adapter_in_model
from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
import diffusers
cuda = torch.cuda.is_available()
print(f"OK: torch {torch.__version__} | diffusers {diffusers.__version__} | cuda {cuda}")
if not cuda:
    print("!! WARNING: CUDA not visible right now. If this is a fresh pod, RESTART the pod "
          "(RunPod: pod menu -> Restart), then re-run setup.sh. Weight download continues.")
PYEOF

if [ "${FLASH_ATTN:-0}" = "1" ]; then
    echo "==> Building flash-attn (optional, this can take a while)"
    $PY -m pip install flash_attn || echo "!! flash_attn build failed — continuing without it"
fi

# --- 2. Model weights ---------------------------------------------------------
mkdir -p pretrained_models
# Use the venv's hf when PERSIST, else the system hf. Reads HF_TOKEN from env.
HF="hf"; [ "${PERSIST:-0}" = "1" ] && HF="$(dirname "$PY")/hf"
dl() { "$HF" download "$1" --local-dir "./pretrained_models/$2"; }

if [[ "$MODEL" != "1.3B" && "$MODEL" != "14B" && "$MODEL" != "both" ]]; then
    echo "!! Unknown model '$MODEL' (use: 1.3B | 14B | both)"; exit 1
fi

echo "==> Downloading wav2vec2 (audio encoder)"
dl facebook/wav2vec2-base-960h wav2vec2-base-960h

if [[ "$MODEL" == "1.3B" || "$MODEL" == "both" ]]; then
    echo "==> Downloading Wan2.1-T2V-1.3B base + OmniAvatar-1.3B"
    dl Wan-AI/Wan2.1-T2V-1.3B Wan2.1-T2V-1.3B
    dl OmniAvatar/OmniAvatar-1.3B OmniAvatar-1.3B
fi

if [[ "$MODEL" == "14B" || "$MODEL" == "both" ]]; then
    echo "==> Downloading Wan2.1-T2V-14B base + OmniAvatar-14B (large!)"
    dl Wan-AI/Wan2.1-T2V-14B Wan2.1-T2V-14B
    dl OmniAvatar/OmniAvatar-14B OmniAvatar-14B
fi

echo ""
TEST_MODEL="$MODEL"; [ "$TEST_MODEL" = "both" ] && TEST_MODEL="1.3B"
echo "==> Done. Quick test:"
echo "    $PY generate.py --model $TEST_MODEL \\"
echo "        --prompt \"A realistic video of a man speaking to the camera.\" \\"
echo "        --image examples/images/0000.jpeg \\"
echo "        --audio examples/audios/0000.MP3"
if [ "${PERSIST:-0}" = "1" ]; then
    echo ""
    echo "==> PERSIST mode: deps live in $VENV, weights in $ROOT/pretrained_models"
    echo "    Both are on /workspace -> STOP the pod (do NOT terminate) and next start"
    echo "    needs NO reinstall. Always run with:  $PY generate.py ..."
fi
