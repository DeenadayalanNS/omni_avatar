#!/usr/bin/env python
"""
score_metrics.py — teacher/student evaluation for OmniAvatar avatar videos.

Measures, per generated video:
  * Sync-C  (LSE-C) : SyncNet lip-sync confidence   (higher = better lip-sync)
  * Sync-D  (LSE-D) : SyncNet lip-sync distance       (lower  = better lip-sync)
  * CSIM            : ArcFace identity cosine vs the reference image (0..1, higher = better)

The SAME script + SAME pinned metric stack is used to score the teacher baseline
AND every future student checkpoint, so results are apples-to-apples. The metric
versions are recorded inside baseline.json so drift is detectable.

Sync-C backend : joonson/syncnet_python (pinned in setup_metrics.sh), the canonical
                 LSE-C/LSE-D implementation used across talking-head papers.
CSIM backend   : insightface buffalo_l (ArcFace r100), cosine similarity.

Usage
-----
# Score the 12 teacher videos and freeze the baseline:
python s1_eval/score_metrics.py demo_out/OmniAvatar-14B/<run_dir> \
    --pairs baseline_pairs.txt \
    --save-baseline

# Score a student checkpoint's videos against the same refs (no --save-baseline):
python s1_eval/score_metrics.py student_out/<run_dir> \
    --pairs baseline_pairs.txt \
    --out eval/student_step5000.json --compare eval/baseline.json

Reference mapping (how each video is paired with its reference image):
  1. --pairs baseline_pairs.txt  (AUTHORITATIVE): line i -> image i -> video i (sorted).
  2. else --refs DIR : match by the zero-padded index in the video filename
                       (result_000.mp4 <-> the 1st ref, sorted).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNCNET_DIR = REPO_ROOT / "s1_eval" / "third_party" / "syncnet_python"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg):
    print(f"[score] {msg}", flush=True)


def list_videos(spec):
    """spec may be a directory, a glob, or a single file. Returns sorted paths."""
    p = Path(spec)
    if p.is_dir():
        vids = sorted(glob.glob(str(p / "**" / "*.mp4"), recursive=True))
    elif any(ch in spec for ch in "*?["):
        vids = sorted(glob.glob(spec))
    else:
        vids = [spec]
    # Drop the audio-only / grid helper files if present; keep result_*.mp4 first.
    vids = [v for v in vids if v.lower().endswith(".mp4")]
    return vids


def parse_pairs_file(path):
    """Parse 'prompt@@image@@audio' lines -> list of dicts (image, audio, prompt)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("@@")
            rows.append({
                "prompt": parts[0] if parts else "",
                "image": parts[1] if len(parts) > 1 else None,
                "audio": parts[2] if len(parts) > 2 else None,
            })
    return rows


def index_in_name(name):
    """Extract the trailing zero-padded index from e.g. result_007.mp4 -> 7."""
    m = re.search(r"(\d+)(?=\D*$)", Path(name).stem)
    return int(m.group(1)) if m else None


def map_videos_to_refs(videos, pairs, refs_dir):
    """Return list of (video, ref_image) pairs."""
    if pairs:
        # Authoritative: sort both, align by index.
        vids = sorted(videos, key=lambda v: (index_in_name(v) if index_in_name(v) is not None else 1e9, v))
        mapping = []
        for i, row in enumerate(pairs):
            vid = vids[i] if i < len(vids) else None
            mapping.append((vid, row["image"], row))
        return mapping
    if refs_dir:
        ref_imgs = sorted(
            glob.glob(str(Path(refs_dir) / "*"))
        )
        ref_imgs = [r for r in ref_imgs if r.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        mapping = []
        for v in sorted(videos, key=lambda v: (index_in_name(v) or 0, v)):
            idx = index_in_name(v) or 0
            ref = ref_imgs[idx] if idx < len(ref_imgs) else (ref_imgs[0] if ref_imgs else None)
            mapping.append((v, ref, None))
        return mapping
    raise SystemExit("Provide either --pairs or --refs to map videos to reference images.")


# --------------------------------------------------------------------------- #
# CSIM — ArcFace identity cosine similarity
# --------------------------------------------------------------------------- #
class CSIMScorer:
    def __init__(self, det_size=640, sample_frames=25):
        from insightface.app import FaceAnalysis
        self.sample_frames = sample_frames
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def _largest_face_embedding(self, img_bgr):
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding
        return emb / (np.linalg.norm(emb) + 1e-8)

    def ref_embedding(self, image_path):
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"reference image not readable: {image_path}")
        return self._largest_face_embedding(img)

    def video_embedding(self, video_path):
        """Average embedding over evenly-sampled frames of the video."""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            # Stream-read fallback.
            idxs = None
        else:
            n = min(self.sample_frames, total)
            idxs = set(np.linspace(0, total - 1, n).astype(int).tolist())
        embs = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idxs is None or i in idxs:
                e = self._largest_face_embedding(frame)
                if e is not None:
                    embs.append(e)
            i += 1
        cap.release()
        if not embs:
            return None
        m = np.mean(embs, axis=0)
        return m / (np.linalg.norm(m) + 1e-8)

    def score(self, video_path, ref_image):
        ref = self.ref_embedding(ref_image)
        vid = self.video_embedding(video_path)
        if ref is None or vid is None:
            return None
        return float(np.dot(ref, vid))  # both L2-normalised -> cosine


# --------------------------------------------------------------------------- #
# Sync-C — SyncNet LSE-C / LSE-D via the pinned syncnet_python
# --------------------------------------------------------------------------- #
class SyncScorer:
    def __init__(self, syncnet_dir=SYNCNET_DIR):
        self.dir = Path(syncnet_dir)
        self.model = self.dir / "data" / "syncnet_v2.model"
        if not self.model.exists():
            raise SystemExit(
                f"SyncNet weights not found at {self.model}\n"
                f"Run: bash s1_eval/setup_metrics.sh"
            )

    def score(self, video_path):
        """Run the canonical two-step SyncNet pipeline; parse LSE-C / LSE-D / offset."""
        work = tempfile.mkdtemp(prefix="syncnet_")
        ref = "clip"
        py = sys.executable
        env = dict(os.environ)
        try:
            # 1) face detect + track + crop
            r1 = subprocess.run(
                [py, "run_pipeline.py", "--videofile", str(video_path),
                 "--reference", ref, "--data_dir", work],
                cwd=str(self.dir), env=env, capture_output=True, text=True)
            if r1.returncode != 0:
                return {"sync_c": None, "sync_d": None, "offset": None,
                        "error": "run_pipeline failed", "stderr": r1.stderr[-800:]}
            # 2) SyncNet scoring
            r2 = subprocess.run(
                [py, "run_syncnet.py", "--videofile", str(video_path),
                 "--reference", ref, "--data_dir", work],
                cwd=str(self.dir), env=env, capture_output=True, text=True)
            out = r2.stdout + "\n" + r2.stderr
            conf = self._grab(out, r"Confidence[:\s]+([-\d.]+)")
            dist = self._grab(out, r"Min dist[:\s]+([-\d.]+)")
            off = self._grab(out, r"AV offset[:\s]+([-\d.]+)")
            if conf is None:
                return {"sync_c": None, "sync_d": None, "offset": None,
                        "error": "no face track / no score", "stdout": out[-800:]}
            return {"sync_c": conf, "sync_d": dist, "offset": off}
        finally:
            import shutil
            shutil.rmtree(work, ignore_errors=True)

    @staticmethod
    def _grab(text, pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Score OmniAvatar videos (Sync-C, Sync-D, CSIM).")
    ap.add_argument("videos", help="Directory, glob, or single .mp4 of generated videos.")
    ap.add_argument("--pairs", help="baseline_pairs.txt (authoritative video<->image mapping).")
    ap.add_argument("--refs", help="Directory of reference images (if not using --pairs).")
    ap.add_argument("--out", default="eval/baseline.json", help="Where to write the JSON report.")
    ap.add_argument("--save-baseline", action="store_true",
                    help="Mark this report as the frozen baseline (writes eval/baseline.json).")
    ap.add_argument("--compare", help="Path to a baseline.json to diff against (student runs).")
    ap.add_argument("--model-name", default="OmniAvatar-14B", help="Label recorded in the report.")
    ap.add_argument("--no-sync", action="store_true", help="Skip Sync-C/Sync-D (CSIM only).")
    ap.add_argument("--no-csim", action="store_true", help="Skip CSIM (sync only).")
    args = ap.parse_args()

    videos = list_videos(args.videos)
    if not videos:
        raise SystemExit(f"No .mp4 videos found under: {args.videos}")
    pairs = parse_pairs_file(args.pairs) if args.pairs else None
    mapping = map_videos_to_refs(videos, pairs, args.refs)
    log(f"scoring {len(mapping)} videos")

    csim_scorer = None if args.no_csim else CSIMScorer()
    sync_scorer = None if args.no_sync else SyncScorer()

    results = []
    for i, (video, ref_img, row) in enumerate(mapping, 1):
        entry = {"id": i, "video": os.path.basename(video) if video else None,
                 "ref": os.path.basename(ref_img) if ref_img else None}
        if video is None:
            entry["error"] = "no video for this pair"
            results.append(entry)
            log(f"[{i}/{len(mapping)}] MISSING video")
            continue
        if sync_scorer:
            entry.update(sync_scorer.score(video))
        if csim_scorer and ref_img:
            entry["csim"] = csim_scorer.score(video, ref_img)
        results.append(entry)
        log(f"[{i}/{len(mapping)}] {entry.get('video')}  "
            f"Sync-C={entry.get('sync_c')}  Sync-D={entry.get('sync_d')}  CSIM={entry.get('csim')}")

    def stat(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        a = np.array(vals, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()), "n": len(vals)}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "is_baseline": bool(args.save_baseline),
        "metric_versions": metric_versions(),
        "n_pairs": len(results),
        "summary": {"sync_c": stat("sync_c"), "sync_d": stat("sync_d"), "csim": stat("csim")},
        "pairs": results,
    }

    out_path = Path("eval/baseline.json") if args.save_baseline else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out_path}")

    s = report["summary"]
    log(f"SUMMARY  Sync-C mean={fmt(s['sync_c']['mean'])} (n={s['sync_c']['n']})  "
        f"CSIM mean={fmt(s['csim']['mean'])} (n={s['csim']['n']})")
    if args.save_baseline:
        sc = s["sync_c"]["mean"]
        if sc is not None and not (6.0 <= sc <= 8.0):
            log(f"WARNING: baseline Sync-C mean {sc:.2f} is outside the expected ~6-8 band. Inspect before freezing.")

    if args.compare:
        diff_against(report, args.compare)


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def diff_against(report, baseline_path):
    with open(baseline_path) as f:
        base = json.load(f)
    log(f"--- vs baseline {baseline_path} ({base.get('model')}) ---")
    for key in ("sync_c", "csim"):
        b = base["summary"][key]["mean"]
        c = report["summary"][key]["mean"]
        if b is None or c is None:
            continue
        d = c - b
        pct = 100 * d / b if b else 0.0
        arrow = "▲" if d >= 0 else "▼"
        log(f"{key:7s} baseline={b:.3f}  now={c:.3f}  {arrow}{d:+.3f} ({pct:+.1f}%)")


def metric_versions():
    versions = {"syncnet": "joonson/syncnet_python@pinned (see setup_metrics.sh)"}
    try:
        import insightface
        versions["insightface"] = insightface.__version__
        versions["csim_model"] = "buffalo_l (ArcFace r100)"
    except Exception:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    return versions


if __name__ == "__main__":
    main()
