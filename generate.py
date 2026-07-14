#!/usr/bin/env python
"""
generate.py — OmniAvatar video generation entrypoint for RunPod (or any GPU box).

Runs the audio-driven avatar pipeline with a simple CLI. No server / API — just
run this Python file on the pod. It wraps the repo's tested inference path
(scripts/inference.py via torchrun) so behaviour matches upstream exactly, while
adding:

  * single-shot generation from CLI args (or batch from a file)
  * local paths OR http(s) URLs for the image and audio
  * automatic VRAM tuning (picks DiT CPU-offload level from the detected GPU)
  * sensible, documented defaults

Examples
--------
Single generation (1.3B model):
    python generate.py \
        --prompt "A realistic video of a man speaking to the camera." \
        --image examples/images/0000.jpeg \
        --audio examples/audios/0000.MP3

From URLs, 14B model, custom guidance:
    python generate.py --model 14B \
        --prompt "A woman narrating calmly." \
        --image https://example.com/face.jpg \
        --audio https://example.com/voice.wav \
        --guidance-scale 4.5 --audio-scale 3 --num-steps 30

Batch (one 'prompt@@image@@audio' per line, same format as upstream):
    python generate.py --input-file examples/infer_samples.txt

Multi-GPU (e.g. 4 GPUs, sequence parallel):
    python generate.py --sp-size 4 --input-file examples/infer_samples.txt
"""

import argparse
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# model name -> (config file, human label)
MODELS = {
    "1.3B": REPO_ROOT / "configs" / "inference_1.3B.yaml",
    "14B": REPO_ROOT / "configs" / "inference.yaml",
}

# DiT parameter count per model (used for VRAM auto-tuning).
DIT_PARAMS = {"1.3B": 1_300_000_000, "14B": 14_000_000_000}


def detect_gpu_mem_gb():
    """Return VRAM (GiB) of GPU 0, or None if it can't be determined."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3), props.name
    except Exception:
        pass
    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,name",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        mem_mib, name = out.split(",")
        return float(mem_mib) / 1024.0, name.strip()
    except Exception:
        return None, None


def auto_num_persistent(model, gpu_gb):
    """
    Choose num_persistent_param_in_dit for the given model + VRAM.

    This is the number of DiT parameters kept resident on the GPU; the rest are
    streamed from CPU on demand. Higher = faster but more VRAM. None = keep the
    whole DiT resident (fastest). 0 = stream everything (lowest VRAM, slowest).
    """
    if gpu_gb is None:
        return None  # unknown -> let the config default decide

    total = DIT_PARAMS[model]
    if model == "1.3B":
        # 1.3B DiT is ~2.6 GB in bf16; keep it fully resident whenever we can.
        if gpu_gb >= 12:
            return None
        if gpu_gb >= 8:
            return 0.5e9
        return 0
    else:  # 14B
        if gpu_gb >= 40:
            return None          # ~36 GB, full speed
        if gpu_gb >= 24:
            return 7e9           # ~21 GB
        if gpu_gb >= 16:
            return 3e9           # ~14 GB
        return 0                 # ~8 GB minimum, slow (heavy streaming)


def is_url(s):
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))


def fetch_if_url(value, dest_dir, tag):
    """Download `value` to dest_dir if it's a URL; otherwise return it unchanged."""
    if not is_url(value):
        return value
    suffix = os.path.splitext(value.split("?")[0])[1] or ""
    dest = os.path.join(dest_dir, f"{tag}{suffix}")
    print(f"[generate] downloading {tag}: {value}")
    urllib.request.urlretrieve(value, dest)
    return dest


def build_input_file(prompt, image, audio, tmp_dir):
    """Create a one-line upstream-format input file: prompt@@image@@audio."""
    parts = [prompt]
    if image:
        parts.append(image)
    if audio:
        if not image:
            raise SystemExit("--audio requires --image (format is prompt@@image@@audio)")
        parts.append(audio)
    line = "@@".join(parts)
    path = os.path.join(tmp_dir, "infer_input.txt")
    with open(path, "w") as f:
        f.write(line + "\n")
    return path


def main():
    p = argparse.ArgumentParser(
        description="Generate OmniAvatar videos (audio-driven avatar).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs — either a batch file, or prompt/image/audio for a single clip.
    p.add_argument("--input-file", help="Batch file: 'prompt@@image@@audio' per line.")
    p.add_argument("--prompt", help="Text prompt (single-clip mode).")
    p.add_argument("--image", help="Reference image: local path or http(s) URL.")
    p.add_argument("--audio", help="Driving audio: local path or http(s) URL.")

    p.add_argument("--model", choices=list(MODELS), default="1.3B",
                   help="Model size to run.")
    p.add_argument("--resolution", choices=["480p", "720p"], default="480p",
                   help="Output resolution (maps to max_hw 720/1280).")

    # Quality / sampling knobs (passed through to the pipeline).
    p.add_argument("--guidance-scale", type=float, default=4.5,
                   help="Prompt CFG. Recommended 4-6.")
    p.add_argument("--audio-scale", type=float, default=None,
                   help="Audio CFG. Higher = tighter lip-sync (e.g. 3). "
                        "Default: same as guidance-scale.")
    p.add_argument("--num-steps", type=int, default=25,
                   help="Diffusion steps. 20-50; more = higher quality/slower.")
    p.add_argument("--max-tokens", type=int, default=30000,
                   help="Tokens per clip (controls latent length).")
    p.add_argument("--overlap-frame", type=int, default=13,
                   help="Overlap between chunks (1 or 13). Must be 1+4n.")
    p.add_argument("--tea-cache", type=float, default=0.0,
                   help="TeaCache L1 threshold (0=off). 0.05-0.15 speeds up.")
    p.add_argument("--seed", type=int, default=42)

    # Hardware.
    p.add_argument("--sp-size", type=int, default=1,
                   help="Sequence-parallel size = number of GPUs.")
    p.add_argument("--num-persistent", type=str, default="auto",
                   help="DiT params kept on GPU (VRAM knob). 'auto' picks from "
                        "detected VRAM; a number overrides; 'none' = whole DiT resident.")
    p.add_argument("--use-fsdp", action="store_true",
                   help="Shard DiT with FSDP (multi-GPU VRAM saving).")

    p.add_argument("--dry-run", action="store_true",
                   help="Print the command that would run and exit.")

    args = p.parse_args()

    config = MODELS[args.model]
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    # Sanity check weights are present before spinning up torchrun.
    exp = REPO_ROOT / "pretrained_models" / f"OmniAvatar-{args.model}" / "pytorch_model.pt"
    if not exp.exists():
        raise SystemExit(
            f"Weights missing: {exp}\n"
            f"Run ./setup.sh (or download the {args.model} weights) first."
        )

    # --- resolve VRAM offload level ---
    gpu_gb, gpu_name = detect_gpu_mem_gb()
    if isinstance(args.num_persistent, str) and args.num_persistent.lower() == "auto":
        num_persistent = auto_num_persistent(args.model, gpu_gb)
    elif isinstance(args.num_persistent, str) and args.num_persistent.lower() == "none":
        num_persistent = None
    else:
        num_persistent = int(float(args.num_persistent))

    if gpu_gb:
        print(f"[generate] GPU: {gpu_name} ({gpu_gb:.1f} GiB) | model={args.model} | "
              f"num_persistent_param_in_dit={num_persistent}")

    tmp_dir = tempfile.mkdtemp(prefix="omniavatar_")

    # --- assemble the input file ---
    if args.input_file:
        input_file = args.input_file
    else:
        if not (args.prompt and args.image):
            raise SystemExit("Provide --input-file, or --prompt and --image (and usually --audio).")
        image = fetch_if_url(args.image, tmp_dir, "image")
        audio = fetch_if_url(args.audio, tmp_dir, "audio") if args.audio else None
        input_file = build_input_file(args.prompt, image, audio, tmp_dir)

    # --- build -hp override string ---
    max_hw = 1280 if args.resolution == "720p" else 720
    hp = {
        "sp_size": args.sp_size,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "max_tokens": args.max_tokens,
        "overlap_frame": args.overlap_frame,
        "tea_cache_l1_thresh": args.tea_cache,
        "seed": args.seed,
        "max_hw": max_hw,
        "use_fsdp": "True" if args.use_fsdp else "False",
    }
    if args.audio_scale is not None:
        hp["audio_scale"] = args.audio_scale
    if num_persistent is not None:
        hp["num_persistent_param_in_dit"] = int(num_persistent)
    hp_str = ",".join(f"{k}={v}" for k, v in hp.items())

    # --- launch via torchrun, using THIS interpreter (keeps the venv consistent) ---
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={args.sp_size}",
        str(REPO_ROOT / "scripts" / "inference.py"),
        "--config", str(config),
        "--input_file", input_file,
        f"--hp={hp_str}",
    ]

    print("[generate] running:\n  " + " ".join(cmd))
    if args.dry_run:
        return

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    out_root = REPO_ROOT / "demo_out" / f"OmniAvatar-{args.model}"
    print(f"\n[generate] done. Output videos are under: {out_root}")


if __name__ == "__main__":
    main()
