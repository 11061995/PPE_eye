# PPE-EYE enhancement schedule

Six-week plan turning the 17-item enhancement list into runnable experiments,
each producing a row in `experiments/results/REPORT.md` and a figure in
`experiments/results/figs/`. Every week ends with a **go / no-go** note: did the
change move `mAP50-95` and (where relevant) `violation_recall`, or not.

## Ground truth about this repo (read once)

- **Dataset is 4-class person-level compliance**, not the paper's 8-class CHVG:
  `data/data.yaml` → `['W', 'WH', 'WHV', 'WV']` (worker / +helmet / +helmet+vest
  / +vest). There are **no helmet-colour classes** in this export, so list
  item #1 ("collapse white/yellow/blue/red") has nothing to collapse. It is
  re-scoped to **thin-class handling** for `WHV` (48 train inst.) and `WV`
  (36 train inst.).
- **Baseline so far:** best prior run `runs/detect/runs/ppe_eye/lightaug_yolo11s`
  reached `mAP50 ≈ 0.35` (and was cut short at epoch 79/150). Everything here is
  measured against that, not against the paper's 0.969.
- **Data is small:** 727 train / 154 val / 316 test images, group-aware split,
  `cross_split_dupes == 0` (see `data/split_audit.json`).
- **Hardware:** RTX 4060 Ti 16 GB, venv at `PPE/` (torch 2.13+cu126,
  ultralytics 8.4.127). ~18–30 min for a 150-epoch `yolo11s` run at 640.

## How the loop runs

```
/loop  python experiments/run.py --next
```

Each tick: the driver finds the next `pending` experiment whose `deps` are
`done`, trains it in the background, then regenerates the report. Self-paced —
no fixed interval. Details in `experiments/README.md`.

---

## Week 1 — Tier 1: training hygiene (items #2, #3, #5, #1-rescoped)

The cheapest gains. All on `yolo11s` unless noted; base recipe for the
resolution/aug arms is fixed to **AdamW lr0=1e-3, cosine, warmup 3,
close_mosaic 10, 150 ep, patience 30**.

| id | item | what it tests |
|---|---|---|
| `w1_s_baseline_repro` | #2 | paper's `lr0=1e-5` NAdam cell on our split — expected to barely move |
| `w1_s_sgd_1e2` | #2 | SGD `lr0=1e-2` cosine |
| `w1_s_adamw_1e3` | #2 | AdamW `lr0=1e-3` cosine — expected LR winner, becomes the base recipe |
| `w1_s_adamw_1e4` | #2 | AdamW `lr0=1e-4` cosine |
| `w1_s_res_960` | #3 | base recipe @ `imgsz=960` |
| `w1_s_res_1280` | #3 | base recipe @ `imgsz=1280` |
| `w1_s_noaug` | #5 | base recipe, **all online aug off** — the control arm |
| `w1_s_domainaug` | #5 | base recipe + strong HSV-V (glare), scale (distance), erasing (occlusion) |
| `w1_s_imbalance` | #1* | base recipe + `copy_paste` + `mixup` to lift `WHV`/`WV` |
| `w1_n_adamw_1e3` | #3 | `yolo11n` base recipe — deployable size point |
| `w1_m_adamw_1e3` | #3 | `yolo11m` base recipe — upper size point / KD teacher candidate |

**Deliverables:** LR-sweep bar chart, resolution mAP-vs-latency curve, per-class
AP grouped bar, aug-ablation table, size/latency table.
**Go/no-go:** does any cell clear `mAP50-95` of the 0.14 prior best by a clear
margin?

## Week 2 — Tier 1: data quality (items #6, #7, #4)

| id | item | what it does |
|---|---|---|
| `w2_label_audit` | #6 | model-assisted flag pass over 316 test images: tiny boxes, high model/GT disagreement, edge-of-frame helmets → 200-image review queue + cleaned test set |
| `w2_eval_clean_vs_noisy` | #6 | re-eval Week-1 winner on cleaned vs original test — reports the label-noise premium |
| `w2_hard_negatives` | #7 | auto-fetch distractor images (hard hats on surfaces, hi-vis on railings, buckets) from public sets, add as background-only images, retrain winner |
| `w2_sahi_upperbound` | #4 | sliced inference (`sahi`) on full-res test — upper-bound row for distant small objects |

**Deliverables:** label-noise report, clean-vs-noisy delta, hard-negative
false-alarm delta, SAHI upper-bound row.

## Week 3 — Tier 2: corruption benchmark (item #8)

`w3_corruptions`: synthetic suite over the test set — motion blur, defocus,
JPEG q30–50, low light, overexposure, dust/haze — **5 severities each**. Report
`mAP50-95` and `violation_recall` as a function of severity, per corruption.
No new data; ~1 week of compute-light eval.

**Deliverables:** robustness curves (one per corruption, mAP + violation recall
vs severity), summary degradation table.

## Week 4 — Tier 2: generalisation + temporal (items #9, #10)

| id | item | what it does |
|---|---|---|
| `w4_crossdata_sh17` | #9 | train on our set, test on SH17 after ontology mapping — expect a large drop; the drop is the finding |
| `w4_crossdata_chv` | #9 | same, CHV dataset |
| `w4_track_vote` | #10 | ByteTrack + BoT-SORT on persons, N-frame majority / confidence-weighted vote, sweep N∈{1,3,5,7,9}, report recall vs false-alarm |

**Deliverables:** cross-dataset drop table, temporal N-sweep trade-off curve.

## Week 5 — Tier 2: deployment contributions (items #11, #12)

| id | item | what it does |
|---|---|---|
| `w5_kd_x_to_s` | #11 | `yolo11x` (or best `m`) teacher → `yolo11s` student, logit KD + neck feature KD; compare to `w1_s` winner |
| `w5_kd_x_to_n` | #11 | same, `yolo11n` student |
| `w5_calibration` | #12 | reliability diagrams + temperature scaling on the winner; report ECE before/after |

**Deliverables:** KD student-vs-baseline table, reliability diagrams, ECE numbers.

## Week 6 — Tier 3: architecture (items #13–#17, only if Weeks 1–2 show headroom)

| id | item |
|---|---|
| `w6_p2_head` | #13 P2 detection head, measure latency cost |
| `w6_attention` | #14 CBAM / ECA / SE / CoordAtt in the neck — **ablation rows only** |
| `w6_two_stage` | #15 person detector → crop → PPE classifier, worst-case latency vs worker count |
| `w6_baselines` | #16 RT-DETR / D-FINE / RTMDet baseline table |
| `w6_qat` | #17 QAT vs PTQ on the deploy student — recovery on thin classes |

**Deliverables:** architecture ablation table, final Pareto — `violation_recall`
vs latency across FP32 → FP16 → INT8 → QAT and model sizes.

---

## Week 7 — the final checkpoint (added after Week 6)

Weeks 1-3 swept the recipe and Week 6 swept the architecture; both came back
flat. Week 4 found the only lever that moved the deployment metric was the
**data**. Week 7 applies that finding and produces the `.pt` that ships.

| id | what it does |
|---|---|
| `final_s_base` | yolo11s on Pictor + SH17-v2, Week-1 winning recipe, no extra aug |
| `final_s_thin` | + `copy_paste`/`mixup`, now that WHV has 180 train instances not 53 |
| `final_s_scale` | + `scale`/`translate` only — isolates the distance-matching aug that `w1_s_domainaug` bundled with brightness and erasing |
| `final_n_deploy` | yolo11n on the same data — the size point Amsar actually deploys |

New data: `experiments/sh17_map_v2.py` uses SH17's `head` class (bare head) as a
**confirmed** no-helmet signal instead of discarding images with no PPE
annotation, then filters the resulting head-only images to deployment geometry
(median person-box area < 4%). Assembled by `experiments/build_finest.py`.
`val`/`test` stay pure Pictor and unchanged since Week 1.

New gate: `experiments/gate.py` implements the acceptance test HEAD_TO_HEAD.md
asked for — four person-level test beds, each reporting its own trivial
baselines, with a pass/fail verdict. `experiments/deploy.py` refuses to install a
checkpoint that has not passed it.

**Deliverables:** `experiments/results/GAINS.md` (the whole-project accounting),
`GATE.json`, `figs/gains_recall_vs_falsealarm.png`, and `ppe_eye.pt` in the
Amsar tree.

---

## Master outputs (regenerated every tick)

- `experiments/results/REPORT.md` — all tables, grouped by week
- `experiments/results/results.jsonl` — one line per completed experiment
- `experiments/results/figs/*.png` — all figures
- `experiments/results/STATUS.md` — what's done / running / pending / blocked
