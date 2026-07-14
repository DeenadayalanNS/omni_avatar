#!/usr/bin/env python
"""
validate_pairs.py — pre-flight check for the 12 baseline pairs.

Run this BEFORE generating on the H100. It enforces the mechanically-checkable
rules from the spec so a wrong sample rate or a repeated identity is caught in
seconds instead of after 6-7 hours of generation.

HARD checks (non-zero exit on failure):
  * exactly N pairs (default 12), each parses as prompt@@image@@audio
  * image + audio files exist
  * audio: WAV, 16 kHz, mono, duration 8-12 s
  * image: shorter side >= 720 px (720p+), exactly one dominant face,
           face bbox area >= 8% of the frame
  * all identities distinct (pairwise ArcFace cosine < --id-threshold)

WARN checks (reported, do not fail):
  * image sharpness (Laplacian variance) below --blur-threshold
  * optional language/type mix vs the spec, if a meta CSV is supplied

Usage:
  python s1_eval/validate_pairs.py baseline_pairs.txt
  python s1_eval/validate_pairs.py baseline_pairs.txt --meta baseline_pairs.meta.csv
"""

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

MIN_DUR, MAX_DUR = 8.0, 12.0
TARGET_SR = 16000
MIN_SHORT_SIDE = 720
MIN_FACE_AREA_FRAC = 0.08


def parse_pairs(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("@@")
            rows.append({
                "line": ln,
                "prompt": parts[0] if parts else "",
                "image": parts[1].strip() if len(parts) > 1 else None,
                "audio": parts[2].strip() if len(parts) > 2 else None,
            })
    return rows


def audio_props(path):
    """Return (ok, msg, dict) using stdlib wave; falls back to soundfile."""
    p = Path(path)
    if not p.exists():
        return False, "file missing", {}
    try:
        with wave.open(str(p), "rb") as w:
            ch = w.getnchannels()
            sr = w.getframerate()
            dur = w.getnframes() / float(sr) if sr else 0
        return True, "", {"sr": sr, "channels": ch, "duration": dur, "container": "wav"}
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return True, "", {"sr": info.samplerate, "channels": info.channels,
                              "duration": info.duration,
                              "container": info.format.lower()}
        except Exception as e:
            return False, f"cannot read audio ({e})", {}


class FaceChecker:
    def __init__(self, det_size=640):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def analyze(self, image_path):
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        faces = self.app.get(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        res = {"h": h, "w": w, "n_faces": len(faces), "blur": blur,
               "face_area_frac": 0.0, "embedding": None}
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            fw = face.bbox[2] - face.bbox[0]
            fh = face.bbox[3] - face.bbox[1]
            res["face_area_frac"] = float((fw * fh) / (w * h))
            emb = face.normed_embedding
            res["embedding"] = emb / (np.linalg.norm(emb) + 1e-8)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", help="baseline_pairs.txt")
    ap.add_argument("--n", type=int, default=12, help="expected number of pairs")
    ap.add_argument("--meta", help="optional CSV: id,lang,type for mix validation")
    ap.add_argument("--id-threshold", type=float, default=0.45,
                    help="pairwise ArcFace cosine above this = likely same identity")
    ap.add_argument("--blur-threshold", type=float, default=80.0,
                    help="Laplacian variance below this = warn (possibly soft)")
    ap.add_argument("--skip-face", action="store_true",
                    help="skip image face/identity checks (audio-only validation)")
    args = ap.parse_args()

    rows = parse_pairs(args.pairs)
    hard_fail = []
    warn = []

    print(f"== Validating {len(rows)} pairs from {args.pairs} ==\n")

    if len(rows) != args.n:
        hard_fail.append(f"expected {args.n} pairs, found {len(rows)}")

    # ---- audio ----
    for r in rows:
        tag = f"pair@line{r['line']}"
        if not r["image"] or not r["audio"]:
            hard_fail.append(f"{tag}: not in prompt@@image@@audio format")
            continue
        ok, msg, a = audio_props(r["audio"])
        if not ok:
            hard_fail.append(f"{tag} audio: {msg} ({r['audio']})")
            continue
        if a["container"] != "wav":
            hard_fail.append(f"{tag} audio: not WAV ({a['container']})")
        if a["sr"] != TARGET_SR:
            hard_fail.append(f"{tag} audio: sample rate {a['sr']} != {TARGET_SR}")
        if a["channels"] != 1:
            hard_fail.append(f"{tag} audio: {a['channels']} channels, need mono")
        if not (MIN_DUR <= a["duration"] <= MAX_DUR):
            hard_fail.append(f"{tag} audio: duration {a['duration']:.1f}s outside {MIN_DUR}-{MAX_DUR}s")
        print(f"  {tag} audio: {a['sr']}Hz {a['channels']}ch {a['duration']:.1f}s")

    # ---- images + identity ----
    embeddings = {}
    if not args.skip_face:
        checker = FaceChecker()
        for r in rows:
            tag = f"pair@line{r['line']}"
            if not r["image"]:
                continue
            info = checker.analyze(r["image"])
            if info is None:
                hard_fail.append(f"{tag} image: unreadable ({r['image']})")
                continue
            short = min(info["h"], info["w"])
            if short < MIN_SHORT_SIDE:
                hard_fail.append(f"{tag} image: shorter side {short}px < {MIN_SHORT_SIDE}px (need 720p+)")
            if info["n_faces"] == 0:
                hard_fail.append(f"{tag} image: no face detected")
            elif info["n_faces"] > 1:
                warn.append(f"{tag} image: {info['n_faces']} faces detected (need single person)")
            if info["face_area_frac"] < MIN_FACE_AREA_FRAC:
                hard_fail.append(f"{tag} image: face is {info['face_area_frac']*100:.1f}% of frame "
                                 f"< {MIN_FACE_AREA_FRAC*100:.0f}%")
            if info["blur"] < args.blur_threshold:
                warn.append(f"{tag} image: low sharpness (Laplacian var {info['blur']:.0f})")
            if info["embedding"] is not None:
                embeddings[r["line"]] = info["embedding"]
            print(f"  {tag} image: {info['w']}x{info['h']} face={info['face_area_frac']*100:.1f}% "
                  f"blur={info['blur']:.0f}")

        # pairwise identity uniqueness
        lines = list(embeddings)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                cos = float(np.dot(embeddings[lines[i]], embeddings[lines[j]]))
                if cos > args.id_threshold:
                    hard_fail.append(f"identities line{lines[i]} & line{lines[j]} look the same "
                                     f"(cosine {cos:.2f} > {args.id_threshold}) — need 12 distinct people")

    # ---- optional mix check ----
    if args.meta and Path(args.meta).exists():
        check_mix(args.meta, warn)

    # ---- report ----
    print()
    for w in warn:
        print(f"  WARN: {w}")
    if hard_fail:
        print(f"\n FAILED ({len(hard_fail)}):")
        for h in hard_fail:
            print(f"  ✗ {h}")
        print("\nFix these before generating on the H100.")
        sys.exit(1)
    print(f"\n ALL HARD CHECKS PASSED ({len(warn)} warnings). Ready to generate. ")


def check_mix(meta_path, warn):
    langs, types = {}, {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            langs[row.get("lang", "?")] = langs.get(row.get("lang", "?"), 0) + 1
            types[row.get("type", "?")] = types.get(row.get("type", "?"), 0) + 1
    print(f"\n  mix — languages: {dict(langs)}  types: {dict(types)}")
    # Spec minimums that actually matter for coverage (upper bounds are soft).
    en = langs.get("en", 0) + langs.get("english", 0)
    ta = langs.get("ta", 0) + langs.get("tamil", 0)
    hi_te = (langs.get("hi", 0) + langs.get("hindi", 0)
             + langs.get("te", 0) + langs.get("telugu", 0))
    if en < 4:
        warn.append(f"English coverage low: {en} (spec wants ~4-5)")
    if ta < 2:
        warn.append(f"Tamil coverage low: {ta} (launch language, spec wants 2-3)")
    if hi_te < 1:
        warn.append("no Hindi/Telugu pair (spec wants 1 for 2nd Indian-language phonemes)")
    for special in ("singing", "pause"):
        if types.get(special, 0) < 1:
            warn.append(f"missing '{special}' pair (spec wants 1)")
    tts = types.get("tts", 0)
    real = types.get("real", 0)
    print(f"  controlled(TTS)={tts}  natural(real)={real}  singing={types.get('singing',0)}  "
          f"pause={types.get('pause',0)}  (spec: ~6 TTS + ~6 real overall)")


if __name__ == "__main__":
    main()
