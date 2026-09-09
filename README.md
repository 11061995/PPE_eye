# PPE-EYE reproduction & training guide

Training pipeline for PPE compliance detection, built to reproduce
*PPE-EYE: A Deep Learning Approach to Personal Protective Equipment Compliance
Detection* (Rahman et al., **Computers** 2026, 15, 45, doi:10.3390/computers15010045)
and then measure what actually survives on a Jetson Orin Nano driving 8 RTSP cameras.

Target deployment: **Amsar** — RTSP → YOLO person detection → danger-zone intrusion
+ PPE check → GPIO relay interlock.

---

## The question this repo answers

The paper claims **mAP50 = 96.9%** at **7.3 ms** inference. Two things to establish,
in this order:

1. Is 96.9% real on a clean split? → `audit_split.py`, then `train.py`
2. Does anything like it survive 8 concurrent streams on 15 W? → `bench_jetson.py`

Do **not** treat 96.9% as a target. Treat it as a hypothesis.

---

## Files

| File | Role |
|---|---|
| `audit_split.py` | **Run this first.** Inspects an existing Roboflow export for leakage, class balance, label scheme, integrity. No GPU, no torch. |
| `split_no_leak.py` | Re-splits grouped by source image. Only needed if the audit finds leakage. |
| `train.py` | Reproduces the paper's Table 4 grid, plus deployable model sizes. |
| `eval_honest.py` | Person-level compliance verdicts: violation recall, false-alarm rate. The metric the relay actually needs. |
| `live_check.py` | Live RTSP/webcam inference with per-person verdicts + CSV logging. |
| `bench_jetson.py` | TensorRT export and real batch-8 latency on the Orin Nano. |
| `setup.sh` / `requirements-train.txt` | Training box environment (x86 + NVIDIA GPU). |
| `requirements-jetson.txt` | Orin Nano environment. **Different, and not optional.** |

---

## Step 0 — Environment

### Training box (x86_64 + NVIDIA GPU)

```bash
bash setup.sh                    # detects CUDA, installs matching torch, verifies
```

Manual equivalent:

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # Linux/macOS
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

Match the torch index to your driver (`nvidia-smi`, top right). Confirm before
going further — a silently CPU-only torch is the most common way to "discover"
that YOLO11x trains slowly:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Jetson Orin Nano — **do not reuse the file above**

`requirements-train.txt` will install cleanly on the Jetson and then not work:

- PyPI's aarch64 `torch` has **no Tegra CUDA**. Clean import, `cuda.is_available() == False`, ~1 FPS.
- pip's `opencv-python` is built **without GStreamer**. It shadows JetPack's system
  OpenCV, so `cv2.VideoCapture` drops to FFmpeg software decode and your 8-stream
  `nvv4l2decode` hardware path silently disappears.

Follow `requirements-jetson.txt`, which has the full sequence and a four-line
verification block. Run that block before trusting any benchmark number.

---

## Step 1 — Audit your dataset **before** training

Expects the standard Roboflow YOLO export:

```
<root>/
  data.yaml
  train/images   train/labels
  valid/images   valid/labels
  test/images    test/labels
```

```powershell
pip install pillow numpy
python audit_split.py --root "C:/data/ppe" --json audit.json
```

> **Windows:** use forward slashes. `"C:\data\train"` contains `\t`, which some
> shells pass through as a tab.

Read three things off the output:

### 1a. Label scheme

| Reported | Meaning |
|---|---|
| **CHVG** (`person`/`vest`/`glass`/`head`/`white`/`yellow`/`blue`/`red`) | Comparable to the paper. Compliance must be **inferred** by helmet→person association — the step the paper never evaluates. `live_check.py` → MODE B. |
| **Explicit-negative** (`Hardhat`/`NO-Hardhat`/`NO-Safety Vest`/…) | Model outputs compliance directly. Easier, but **not comparable to the paper's 96.9%**. `live_check.py` → MODE A. |

These are different experiments. Know which one you are running before you
interpret any number.

### 1b. Leakage verdict

| Result | Action |
|---|---|
| `cross_split_stem_collisions == 0` and `cross_split_near_duplicates == 0` | Split is honest. **Skip step 2.** Train directly on your existing tree. |
| Either > 0 | The split overstates accuracy. Go to step 2. |

**Epistemic status, stated plainly:** I originally argued the paper leaked, based
on §3.1.2's sentence ordering — augment to 28,000 instances, *then* split 80:20.
But Roboflow's default generate flow splits **first** and augments **train only**.
If the authors used the default, their eval sets were clean and that criticism
collapses. Sentence ordering is not evidence. Your audit is the actual test, and
a zero result means I was wrong.

### 1c. Thin test classes

Any class with <30 test objects has an AP that is noise. The paper's weakest class
(`glass`) had 51 source instances in total and it reports no confidence intervals.
Note which of your classes are in that regime — you will need to caveat them.

---

## Step 2 — Re-split (**only if step 1b found leakage**)

```powershell
# leakage-free: augmented siblings and near-duplicates kept together
python split_no_leak.py --src "C:/data/ppe" --dst "C:/data/ppe_clean" --val 0.15 --test 0.15 --copy

# naive random split of the same pool, for the A/B
python split_no_leak.py --src "C:/data/ppe" --dst "C:/data/ppe_naive" --hamming 999 --copy
```

> `--copy` is **required on Windows** — the default is symlinks, which need admin
> or Developer Mode.

Then train both and report `mAP50(naive) − mAP50(clean)`. That delta is the
leakage premium, and it is a publishable result on its own.

---

## Step 3 — Train

### 3a. Smoke model first (~40 min, one GPU)

Do this before the paper's grid. If it looks wrong on your own footage, no
architecture search will save it.

```bash
yolo detect train model=yolo11s.pt data="C:/data/ppe/data.yaml" \
     epochs=50 imgsz=640 batch=16 seed=0
```

### 3b. The paper's grid

```bash
python train.py --data "C:/data/ppe/data.yaml" --tag clean
python train.py --data "C:/data/ppe/data.yaml" --tag clean --only yolo11x_21
python train.py --data "C:/data/ppe/data.yaml" --tag clean --deploy   # + n/s/m sizes
```

Reproduced verbatim from Table 4:

| config | model | epochs | batch | optimizer | lr0 | momentum | cos_lr | claimed mAP50 |
|---|---|---|---|---|---|---|---|---|
| `yolo10n_25` | yolov10n | 25 | 32 | auto | 0.01 | 0.937 | False | 0.770 |
| `yolo10s_75` | yolov10s | 75 | 16 | auto | 0.01 | 0.937 | False | 0.873 |
| `yolo11l_100` | yolo11l | 100 | 28 | NAdam | 1e-5 | 0.937 | True | 0.907 |
| `yolo11x_87` | yolo11x | 87 | 20 | NAdam | 1e-5 | 0.5 | True | 0.910 |
| `yolo11x_21` | yolo11x | 21 | 16 | NAdam | 1e-5 | 0.937 | True | **0.969** |

Two things to watch:

- **`yolo11x_21` beating `yolo11x_87` at identical lr** is the most suspicious cell
  in the paper. Reproducing both under one split is the cheapest way to find out
  whether it is real.
- **lr0 = 1e-5 with a pretrained backbone is extremely low.** Expect the 21-epoch
  run to barely move off the COCO initialisation. If it scores well anyway, the
  benchmark is easy — which is itself the finding.

Add `--no-aug` to disable Ultralytics' online augmentation. Your Roboflow export is
already augmented; stacking both is a confound the paper never controls for.

**Known failure:** `yolov10n.pt` / `yolov10s.pt` may 404. The v10 *configs* ship in
the ultralytics wheel, but the pretrained checkpoints are hosted separately and have
moved. If they fail, drop those two rows — they are not the interesting cells.

---

## Step 4 — Evaluate the decision, not the boxes

The paper reports object-level mAP over 8 classes. But an interlock does not fire on
"a helmet was detected somewhere in frame." It fires on a per-person verdict:

```
for each person P:   compliant(P) = helmet_on(P) AND vest_on(P)
```

That needs helmet→head→person association. The paper never describes or evaluates
it, yet its Figure 10 UI clearly draws one verdict box per person. A model can hit
0.97 object mAP and still be wrong in a frame with three overlapping workers.

```bash
python eval_honest.py --weights runs/ppe_eye/clean_yolo11x_21/weights/best.pt \
    --data "C:/data/ppe/data.yaml" --split test

# and on your own labelled site footage
python eval_honest.py --weights ... --data "C:/data/amsar_site/data.yaml" --split val
```

Reports across a confidence sweep:

- **violation recall** — non-compliant workers caught. Missing one is the safety
  failure. This is the number that makes Amsar defensible.
- **false-alarm rate** — compliant workers flagged. Two nuisance trips a shift and
  the operator bypasses the interlock, at which point recall is irrelevant.
- **undetected persons** — never detected, therefore never checked. mAP counts none
  of this the way you need it counted.

Pick your operating point from this sweep. Do not inherit the paper's 0.424.

---

## Step 5 — Live look

```bash
python live_check.py --weights best.pt --source 0 --debug-assoc
python live_check.py --weights best.pt --source "rtsp://user:pass@192.168.1.50:554/stream1"
python live_check.py --weights best.pt --source clip.mp4 --no-show --csv run1.csv
```

Green = compliant, red = violation, **amber = unresolved** (detected but not
adjudicable). Record two numbers from the first run:

- **UNRESOLVED %** — the honest headline
- **p95 latency**

Everything later gets compared against those. Note this run has **no ground truth**:
it measures behaviour, not accuracy. A very low violation rate on a site where people
*do* skip vests means silent misses, not compliance.

If the scheme is MODE B, the verdict depends partly on `live_check.py`'s association
heuristic rather than the weights. Watch with `--debug-assoc` before you blame the model.

---

## Step 6 — Jetson reality check

```bash
python bench_jetson.py --weights best.pt --export --int8 --calib "C:/data/ppe/data.yaml"
python bench_jetson.py --engine best.engine --batches 1,2,4,8 --cameras 8 --target-fps 5
```

Run it with FFmpeg/NVENC **live**, on a warm device (`sudo nvpmodel -m 0 && sudo jetson_clocks`).
An idle-Jetson number is one you will never see in production.

Then re-run step 4 **on the INT8 engine**. Quantisation costs accuracy on exactly the
small classes that already struggle — the paper's own confusion matrix sends 20% of
`glass` and 14% of `head` to background.

### Sizing

| Model | Params | GFLOPs @640 | Relative |
|---|---|---|---|
| yolo11n | ~2.6 M | ~6.5 | 1× |
| yolo11s | ~9.4 M | ~21.5 | ~3.3× |
| yolo11m | ~20.1 M | ~68 | ~10× |
| yolo11x | ~56.9 M | ~195 | ~30× |

8 cameras × 5 FPS = 40 inferences/s, on a part simultaneously doing 8 H.264 decodes
and HLS packaging. Expectation: **yolo11s is the largest thing that fits**, meaning
the paper's headline architecture is not the one you deploy.

On the paper's 7.3 ms: it was measured on a Dell XPS 9320 with Intel Iris Xe
integrated graphics. That implies ~137 FPS for YOLO11x on an iGPU. Table 3 also lists
exactly 7.3 ms for a *different* model from ref [20]. Measure your own.

---

## Open design decisions

1. **Collapse the four helmet colours into one `helmet` class?** White/yellow/blue/red
   splits your data four ways for zero safety value. Collapsing roughly 4×s the
   per-class sample count and may beat 96.9% on its own. Argument against: you lose
   role identification if the site uses colour-coded helmets by trade.
2. **Helmet-only or helmet+vest compliance?** CHVG has no gloves, boots, or goggles
   beyond `glass`. More data of the wrong ontology does not help. Ask the site safety
   officer what actually needs enforcing before scaling the dataset.
3. **Should PPE be on the interlock path at all?** Amsar already does person-in-zone
   → relay. PPE is a *reporting* function — you do not stop a machine because someone's
   vest is missing 30 m away. A missed intrusion at 2 FPS kills someone; a missed vest
   does not. Consider one batched yolo11s pass serving both, spending reclaimed
   compute on frame rate.

---

## Where the publishable result is

The reproduction is less interesting than the gap. Three angles, ascending effort:

1. **Split audit** — quantified naive-vs-grouped delta on CHVG. Short, clean,
   immediately legible to reviewers. (Only if your audit finds leakage.)
2. **Object mAP vs person-level verdict** — a model with high mAP50 and poor violation
   recall in multi-worker frames. Everyone in this literature reports mAP; nobody
   reports the decision.
3. **Edge accuracy–latency Pareto for 8-stream interlock** — FP32 → FP16 → INT8 across
   model sizes, with **violation recall** on the y-axis instead of mAP. Amsar's
   differentiator, and unpublished for Saudi construction conditions.

The authors are at IAU Dammam (corresponding address in the paper). Asking for the
split manifest and trained weights costs one email and could settle the reproduction
in an afternoon. If they cannot produce the split, that is itself a finding.

---

## Enhancement experiments (`experiments/`)

A separate, self-driving experiment pipeline that takes a 17-item enhancement
list (learning rate, resolution, augmentation, label quality, robustness,
distillation, calibration, architecture) and turns each item into a runnable
experiment that emits a result row and a figure. Built and run by **Claude Code
(Sonnet 5)** — see *Contributions* below.

### Layout

| path | role |
|---|---|
| `SCHEDULE.md` | the full 6-week plan; every item mapped to an experiment id |
| `experiments/registry.yaml` | the experiment matrix — `defaults` + per-experiment `params`, `deps`, `week`, `kind` |
| `experiments/run.py` | run one experiment (`--next` for loop mode, or an id): train → test-split eval → latency bench → `results/<id>.json` |
| `experiments/report.py` | aggregate `results/*.json` → `results/REPORT.md`, `STATUS.md`, `figs/*.png` |
| `experiments/lib.py` | registry loading, dependency/next-experiment logic, latency bench |
| `experiments/results/` | one `<id>.json` per experiment, `results.jsonl`, `REPORT.md`, `WEEK1_FINDINGS.md`, `figs/` |

Runs in the `PPE/` venv as shipped (no extra installs for Week 1). Trained
checkpoints land in `runs/ppe_enh/<id>/weights/` (git-ignored).

### Running it

```powershell
PPE/Scripts/python.exe experiments/run.py --next        # next pending, deps-ready
PPE/Scripts/python.exe experiments/run.py w1_s_adamw_1e4 # a specific experiment
PPE/Scripts/python.exe experiments/report.py             # rebuild tables + figures
```

As a self-paced loop (Claude Code): `/loop PPE/Scripts/python.exe experiments/run.py --next`.
`--next` exits 3 at the first experiment whose `kind` has no handler yet — a
deliberate review gate between weeks. Only `kind: train` is implemented; Weeks
2–6 (`audit`, `eval`, `corrupt`, `crossdata`, `track`, `distill`, `calib`,
`arch`) are declared but not yet coded.

### Week 1 result — Tier 1 training hygiene (11/11 runs)

**The Tier-1 knobs did not beat a properly-trained baseline.** The earlier
"prior best mAP50-95 ≈ 0.14" was interrupted training. A clean 100-epoch
fine-tune sits at **mAP50 ≈ 0.49 / mAP50-95 ≈ 0.29**, and nothing in the sweep
moves it up. Full writeup: `experiments/results/WEEK1_FINDINGS.md`.

| item | hypothesis | result | verdict |
|---|---|---|---|
| #2 learning rate | `lr0=1e-3` beats the paper's `1e-5` | `1e-3` → 0.117 (worst); `1e-4` ≈ `1e-5` ≈ 0.29; default `1e-2` (SGD) → 0.219 | **falsified** — low LR wins |
| #3 resolution | 640 → 960 → 1280 helps small objects | monotonic decline 0.282 → 0.248 → 0.144 | **falsified** — export is pre-resized; upscaling adds overfit, not detail |
| #5 augmentation | domain-matched aug helps | none 0.196 · default 0.282 · heavy 0.238 | **partial** — aug helps, but Ultralytics' default is already the sweet spot |
| #1\* thin classes | `copy_paste`+`mixup` lift WHV/WV | WHV **fell** 0.13 → 0.045 | **falsified** — WHV is a data/label problem |
| model size | bigger = better | s 0.282 > m 0.270 > n 0.230 | yolo11s is the sweet spot *and* the deployable size |

Working model: `runs/ppe_enh/w1_s_adamw_1e4/weights/best.pt`
(AdamW `lr0=1e-4`, 640, default aug — best precision/recall, 7-min train).

**Two open issues that gate everything after Week 1:**

1. **mAP50-95 has a ~0.30 ceiling** no optimiser setting touches → the
   constraint is the data, not training.
2. **WV (worker+vest) scores 0.995 in almost every run** on 36 training
   instances — near-perfect and flat. Likely trivially separable in this export
   or a within-class near-duplicate in the split. Until a label audit resolves
   it, the 0.49 headline is measured against a questionable ruler.

### Week 2 result — data quality & honest evaluation

Full writeup: `experiments/results/WEEK2_FINDINGS.md`. Model under test:
`w1_s_adamw_1e4` (mAP50 ≈ 0.49).

**1. The WV=0.995 question — answered by counting.** Real test-set object counts:
WV = 6, WHV = 20 (both < 30 → "AP is noise"). Not leakage; just tiny classes
times the ~3× within-split augmentation. The compliant class WHV has 53 training
instances total.

**2. Train ≠ test distribution — the real ceiling.** Train: 1.46 workers/image,
6.5 % median box area (close-up singletons). Test: 3.16 workers/image, 1.6 %
median box area (crowded, distant). The model learns easy close-ups and is scored
on small clustered workers — this, not the optimiser, is why Week 1 stalled.

**3. Person-level honest eval (`experiments/honest_eval.py`)** — the mAP → decision gap:

| rule | best violation recall | false-alarm rate | undetected persons |
|---|---|---|---|
| helmet (WH or WHV compliant) | 0.34 | 0.20 | 354–576 / 999 |
| strict (WHV compliant) | 0.65 | 0.88–1.00 | 354–576 / 999 |

mAP50 ≈ 0.49 delivers **person-level violation recall 0.34** and **27–58 % of
workers never detected**. Strict-rule false-alarm hits 1.0 — the model barely
emits WHV, so every compliant worker is flagged.

**4. Label noise is a minor contributor.** Conservative auto-clean (drop GT boxes
that are both < 0.3 % area and invisible to the model) removes 88 / 999 test
boxes → +6 pts strict verdict accuracy. The other 266 undetected persons are
real misses.

**5. SAHI sliced inference (item #4) does not help.** 320 px and 480 px slices
both raise undetected persons (354 → 437) — the model is scale-brittle from the
narrow training distribution, so tiling hurts. A null result, not an upper bound.

`w2_hard_negatives` (item #7) is **deferred** — it targets false positives; the
measured failure mode is false negatives.

**Status: paused after Week 2.** The project's finding is now firm: on this
dataset a properly trained YOLO11s reaches mAP50 ≈ 0.49 but only ~0.34
person-level violation recall and misses a quarter to a half of all workers,
because the training data (sparse, close-up) does not resemble deployment
(crowded, distant) and the compliant class is data-starved. No training-recipe,
augmentation, resolution, or architecture change addresses this — it needs
deployment-representative labelled data.

## Contributions

- **Reproduction pipeline** (`audit_split.py`, `split_no_leak.py`, `train.py`,
  `eval_honest.py`, `live_check.py`, `bench_jetson.py`, this README's
  reproduction sections): project author.
- **Enhancement pipeline** (`SCHEDULE.md`, `experiments/` — incl.
  `experiments/honest_eval.py`, `experiments/audit.py`, `experiments/sahi_eval.py`
  — the Week 1–2 experiments and findings, and the enhancement sections of this
  README): **Claude Code (Sonnet 5)**, run as a self-paced experiment loop under
  the author's direction. Design decisions (dataset ontology, base-recipe
  re-pointing after the LR sweep, deferring hard-negatives, pausing after each
  week) were made by the author at review gates.

## Reference

Rahman, A.; Ahmed, M.S.; AlBugami, K.N.; Alabbad, A.Y.; AlFantoukh, A.A.;
Alshaikhahmed, Y.H.; Alzahrani, Z.S.; Khan, M.A.A.; Youldash, M.; Alshahrani, S.M.
*PPE-EYE: A Deep Learning Approach to Personal Protective Equipment Compliance
Detection.* Computers **2026**, 15, 45. https://doi.org/10.3390/computers15010045
(CC BY)

Dataset: PPE Object Detection (V9, Pictor-V3-Revised), Roboflow —
https://universe.roboflow.com/ppe-orxtt/ppe-u7jtr/dataset/9