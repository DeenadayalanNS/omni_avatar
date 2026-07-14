# s1_eval — Teacher Baseline Validation (12 pairs)

This is the **Stage-1 exit** evaluation. The teacher (OmniAvatar-14B) generates 12
canonical videos = the "exam." We measure **Sync-C**, **Sync-D**, and **CSIM** and
freeze them into `eval/baseline.json`. Every future *student* checkpoint is scored
on the **same 12 pairs with the same code**, so results are apples-to-apples.

## Metrics

| Metric | Meaning | Backend | Good |
|---|---|---|---|
| **Sync-C** (LSE-C) | Lip-sync confidence | SyncNet (`joonson/syncnet_python`, pinned) | higher (~6-8 expected) |
| **Sync-D** (LSE-D) | Lip-sync feature distance | SyncNet | lower |
| **CSIM** | Identity cosine vs reference image | insightface `buffalo_l` (ArcFace r100) | higher (0..1) |

**Reproducibility is the whole point.** The SyncNet commit and insightface version
are recorded inside `eval/baseline.json` (`metric_versions`) and in
`third_party/SYNCNET_COMMIT.txt`. Freeze the SyncNet commit (`SYNCNET_COMMIT` in
`setup_metrics.sh`) so the student is scored with byte-identical metric code.

## Files

| File | Purpose |
|---|---|
| `setup_metrics.sh` | Install pinned SyncNet + weights, insightface, onnxruntime |
| `score_metrics.py` | Score videos → `eval/baseline.json` (`--save-baseline`) |
| `validate_pairs.py` | Pre-flight: enforce all 12-pair rules **before** generating |
| `prep_audio.py` | Normalise audio → 16 kHz mono WAV, 8–12 s |
| `baseline_pairs.template.txt` | 12-pair scaffold with per-concept prompts |
| `baseline_pairs.meta.csv` | id, lang, type per pair (drives the mix check) |
| `sources.csv` | Provenance + license log for every image/audio asset |

## The workflow (Day 13–14)

> Do this **only after** the eval freeze (STEP 7). The 12 images come **from** the
> frozen 50 unseen faces — never from training pairs.

### 0. One-time: install the metric stack (on the scoring PC)
```bash
bash s1_eval/setup_metrics.sh          # add ONNX_CPU=1 on a CPU-only box
```

### 1. Select 12 images from the frozen set
One per row of the concept table, **12 distinct identities**, 720p+, face ≥8% of
frame, waist/chest-up, hands visible for #6 (also nice for #1, #4). Copy them to
`eval_frozen/img01.jpg … img12.jpg` (or any paths you like).

### 2. Prepare 12 audios
Get each clip to **16 kHz mono WAV, 8–12 s**:
```bash
# single clip, take 10 s starting at 2.0 s, loudness-normalise
python s1_eval/prep_audio.py raw/podium.m4a -o prepared_audio/aud06.wav --start 2 --duration 10 --loudnorm
# or a whole folder at once
python s1_eval/prep_audio.py raw_audio/ --outdir prepared_audio --duration 10 --loudnorm
```
Log every asset (source + license) in `s1_eval/sources.csv`.

### 3. Build `baseline_pairs.txt`
```bash
cp s1_eval/baseline_pairs.template.txt baseline_pairs.txt
# edit paths (eval_frozen/imgNN, prepared_audio/audNN); tune prompts if needed
```

### 4. Validate BEFORE generating (saves 6–7 H100 hrs)
```bash
python s1_eval/validate_pairs.py baseline_pairs.txt --meta s1_eval/baseline_pairs.meta.csv
```
Fix every ✗ (hard fail) before continuing. Warnings are advisory.

### 5. Generate the 12 teacher videos (H100)
```bash
nohup python generate.py --model 14B --input-file baseline_pairs.txt \
      --num-steps 30 --guidance-scale 4.5 > gen.log 2>&1 &
```
Then pull the results and **terminate the pod** (cost). Outputs land under
`demo_out/OmniAvatar-14B/<run_dir>/`.

### 6. Score + freeze the baseline (scoring PC)
```bash
python s1_eval/score_metrics.py demo_out/OmniAvatar-14B/<run_dir> \
       --pairs baseline_pairs.txt --save-baseline
```
Check `eval/baseline.json` (Sync-C mean ~6–8). The script warns if it's outside
that band. Then `rclone` it to B2 → **Stage 1 exit ✅**.

### Later: score a student the same way
```bash
python s1_eval/score_metrics.py student_out/<run_dir> \
       --pairs baseline_pairs.txt --out eval/student_stepNNNN.json \
       --compare eval/baseline.json --model-name student-stepNNNN
```

## Asset-prep guidance (since assets aren't ready)

- **English TTS (#1,3,4)** — any high-quality TTS (e.g. a neutral studio voice for
  #1/#3, an expressive/energetic voice for #4). #3 is the deliberate mismatch:
  **female voice on a male face** — keep it that way, it's the robustness probe.
- **Tamil TTS (#2 female, #12 male)** and **Hindi/Telugu (#9)** — use a TTS that
  actually supports the language; verify pronunciation before locking.
- **Real speech (#5 serious, #6 podium, #7 fast, #8 beard casual)** — natural
  recordings. #6 must keep its **room ambience** (that's the point); no crowd noise
  on the others.
- **Singing (#10)** — **acapella / vocal-dominant only**; instrument-heavy tracks
  confuse Sync scoring. Pixabay/Mixkit license or CC0 vocal, 8–10 s with clear
  sustained notes. Match the singer image's gender. Log the URL + license.
- **Pause-heavy (#11)** — long silences between sentences; this tests that the
  mouth **closes** during silence. Fine to make with TTS + inserted gaps.
- **Pairing rule** — image and audio must come from **different sources**;
  scenario-match them (podium image ↔ podium audio, singer ↔ singing).

## Notes
- `eval_frozen/`, `prepared_audio/`, `raw_audio/`, `demo_out/`, `eval/` are
  git-ignored — assets and outputs are not committed.
- `validate_pairs.py` and CSIM need `insightface`; install via `setup_metrics.sh`.
