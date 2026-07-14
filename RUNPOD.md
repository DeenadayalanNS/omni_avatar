# Running OmniAvatar on RunPod

This repo is set up so you can push it to GitHub, clone it on a RunPod pod, run one
setup script, and then generate avatar videos with a single Python command — no API,
no server.

## TL;DR — which GPU / how much VRAM?

**Short answer: a 24 GB GPU (RTX 4090) is the sweet spot.** It runs the big 14B
model with light CPU offload, and runs the small 1.3B model very fast.

| Your goal | GPU (RunPod) | VRAM | Model | Notes |
|---|---|---|---|---|
| **Best value (recommended)** | RTX 4090 | 24 GB | 14B or 1.3B | 14B fits at ~21 GB with offload; 1.3B flies |
| **Max quality + speed** | A100 40/80 GB, H100 | 40–80 GB | 14B | Whole model resident (~36 GB), fastest per step |
| **Cheapest / fastest** | RTX 4090 / A10 / L4 / A5000 | 16–24 GB | 1.3B | Lowest cost, good quality |
| **Bare minimum for 14B** | RTX 4090 / 3090 | 12–16 GB | 14B | Works via heavy streaming, but slow |

### VRAM detail (from the upstream benchmark, tested on A800)

**14B model** — VRAM depends on how much of the DiT stays resident on the GPU
(`--num-persistent`, which `generate.py` auto-picks for you):

| num_persistent | VRAM | Speed |
|---|---|---|
| none (full) | ~36 GB | 16.0 s/it |
| 7e9 | ~21 GB | 19.4 s/it |
| 0 (stream all) | ~8 GB | 22.1 s/it |

**1.3B model** — the DiT is only ~2.6 GB, so it stays fully resident on any
≥12 GB GPU and runs much faster. Great for iterating.

### Also check (not just VRAM)

- **System RAM:** offloaded weights live in CPU RAM. Use a pod with **≥32 GB RAM
  for 1.3B**, **≥64 GB RAM for 14B**.
- **Disk:** weights are big. Give the pod/volume **≥50 GB for 1.3B**, **≥150 GB for 14B**.
- The umt5-xxl text encoder (~11 GB) is streamed from CPU, so it does **not** add
  much to the resident VRAM requirement above.

---

## 1. Launch a pod

Pick a RunPod **PyTorch / CUDA** template (CUDA 12.1+), on one of the GPUs above.
Make sure the container/volume disk is large enough (see table).

## 2. Clone + set up

```bash
git clone https://github.com/<you>/OmniAvatar
cd OmniAvatar

# Installs pinned deps and downloads weights for the chosen model.
bash setup.sh 1.3B          # or: bash setup.sh 14B   /   bash setup.sh both
```

Optional:
```bash
export HF_TOKEN=hf_xxx      # faster, authenticated Hugging Face downloads
FLASH_ATTN=1 bash setup.sh 1.3B   # also build flash-attn (optional, slow)
```

## 3. Generate a video

Single clip:
```bash
python generate.py \
    --prompt "A realistic video of a man speaking directly to the camera." \
    --image examples/images/0000.jpeg \
    --audio examples/audios/0000.MP3
```

The image and audio can be **local paths or http(s) URLs**:
```bash
python generate.py --model 14B \
    --prompt "A woman narrating calmly to the camera." \
    --image https://example.com/face.jpg \
    --audio https://example.com/voice.wav \
    --guidance-scale 4.5 --audio-scale 3 --num-steps 30
```

Batch (one `prompt@@image@@audio` per line):
```bash
python generate.py --input-file examples/infer_samples.txt
```

Multi-GPU (e.g. 4 GPUs):
```bash
python generate.py --sp-size 4 --input-file examples/infer_samples.txt
```

Output videos are written under `demo_out/OmniAvatar-<model>/...`.

## 4. Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `1.3B` | `1.3B` or `14B` |
| `--resolution` | `480p` | `480p` or `720p` |
| `--guidance-scale` | `4.5` | Prompt CFG (4–6 recommended) |
| `--audio-scale` | = guidance | Higher → tighter lip-sync (try `3`) |
| `--num-steps` | `25` | 20–50; more = better/slower |
| `--tea-cache` | `0` | `0.05–0.15` to speed up (slight quality cost) |
| `--num-persistent` | `auto` | VRAM knob; `auto` picks from detected VRAM. `none`/`0`/a number to override |
| `--sp-size` | `1` | Number of GPUs |
| `--dry-run` | off | Print the command and exit |

### Prompt tip
Structure prompts as: `[first-frame description] - [human behavior] - [background]`.
Recommended guidance/audio CFG range is **4–6**.
