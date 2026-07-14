#!/usr/bin/env python
"""
prep_audio.py — normalise audio to the OmniAvatar pipeline standard.

Converts any input (mp3/m4a/wav/…) into 16 kHz mono WAV, trimmed to a clip in
the 8-12 s window, optionally loudness-normalised. Use it to prepare the 12
baseline audios so they pass validate_pairs.py.

Requires ffmpeg on PATH.

Single file:
  python s1_eval/prep_audio.py in.mp3 -o out/aud01.wav --start 2.0 --duration 10

Whole folder (each file -> <name>.wav in --outdir, first 10 s):
  python s1_eval/prep_audio.py raw_audio/ --outdir prepared_audio --duration 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4"}


def ffprobe_duration(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True).strip()
        return float(out)
    except Exception:
        return None


def convert(src, dst, start, duration, loudnorm):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration and duration > 0:
        cmd += ["-t", str(duration)]
    filt = "aresample=16000"
    if loudnorm:
        filt = "loudnorm=I=-16:TP=-1.5:LRA=11," + filt
    cmd += ["-ac", "1", "-ar", "16000", "-af", filt,
            "-c:a", "pcm_s16le", "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input audio file OR a directory of audio files")
    ap.add_argument("-o", "--out", help="output wav path (single-file mode)")
    ap.add_argument("--outdir", help="output directory (folder mode)")
    ap.add_argument("--start", type=float, default=0.0, help="clip start (seconds)")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="clip length (seconds); target 8-12 for the baseline")
    ap.add_argument("--loudnorm", action="store_true",
                    help="EBU R128 loudness normalisation (recommended for TTS/real mix)")
    args = ap.parse_args()

    if not (8.0 <= args.duration <= 12.0):
        print(f"[prep] note: duration {args.duration}s is outside the 8-12s baseline window.",
              file=sys.stderr)

    src = Path(args.input)
    if src.is_dir():
        outdir = Path(args.outdir or "prepared_audio")
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if not files:
            raise SystemExit(f"no audio files in {src}")
        for f in files:
            dst = outdir / (f.stem + ".wav")
            convert(f, dst, args.start, args.duration, args.loudnorm)
            report(f, dst, args.duration)
    else:
        dst = Path(args.out) if args.out else src.with_suffix(".16k.wav")
        convert(src, dst, args.start, args.duration, args.loudnorm)
        report(src, dst, args.duration)


def report(src, dst, requested):
    """Print the ACTUAL output duration and warn if it fell short of the 8-12s window."""
    actual = ffprobe_duration(dst)
    shown = f"{actual:.1f}s" if actual is not None else "?"
    print(f"[prep] {Path(src).name} -> {dst}  (16kHz mono, {shown})")
    if actual is not None and not (8.0 <= actual <= 12.0):
        print(f"[prep] WARN: {Path(dst).name} is {actual:.1f}s — outside the 8-12s "
              f"baseline window (source likely shorter than {requested}s). "
              f"validate_pairs.py will reject it.")


if __name__ == "__main__":
    main()
