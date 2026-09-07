# Week 1 findings — Tier 1 training hygiene

11/11 runs complete. **Headline: the Tier-1 knobs (items #2 LR, #3 resolution,
#5 aug) do not beat a properly-trained baseline.** The "prior best 0.14" was an
artifact of interrupted runs, not a real baseline. A clean 100-epoch fine-tune
sits at **mAP50 ≈ 0.49 / mAP50-95 ≈ 0.29** and nothing in the sweep moves it up.

## Full Week 1 table

| run | recipe | mAP50 | mAP50-95 | P | R | epochs | wall |
|---|---|---|---|---|---|---|---|
| w1_s_baseline_repro | NAdam 1e-5, 100ep (paper's cell) | 0.494 | **0.295** | 0.587 | 0.475 | 100 | 14.8 m |
| w1_s_adamw_1e4 | AdamW 1e-4 | 0.494 | 0.282 | **0.599** | 0.498 | 45* | 6.8 m |
| w1_s_imbalance | AdamW 1e-4 + copy_paste + mixup | 0.453 | 0.292 | 0.467 | 0.482 | 45* | 6.8 m |
| w1_m_adamw_1e3 | yolo11m, AdamW 1e-4 | 0.461 | 0.270 | 0.452 | **0.519** | 53* | 17.2 m |
| w1_s_res_960 | AdamW 1e-4 @ 960 | 0.461 | 0.248 | 0.398 | 0.509 | 39* | 13.0 m |
| w1_s_domainaug | AdamW 1e-4 + heavy HSV/scale/erasing | 0.452 | 0.238 | 0.518 | 0.478 | 74* | 12.5 m |
| w1_n_adamw_1e3 | yolo11n, AdamW 1e-4 | 0.417 | 0.230 | 0.477 | 0.460 | 65* | 5.7 m |
| w1_s_sgd_1e2 | SGD 1e-2 | 0.371 | 0.219 | 0.376 | 0.414 | 82* | 11.8 m |
| w1_s_noaug | AdamW 1e-4, all online aug OFF | 0.404 | 0.196 | 0.471 | 0.460 | 42* | 7.0 m |
| w1_s_res_1280 | AdamW 1e-4 @ 1280 | 0.424 | 0.144 | 0.479 | 0.454 | 47* | 28.4 m |
| w1_s_adamw_1e3 | AdamW 1e-3 | 0.269 | 0.117 | 0.317 | 0.393 | 110* | 15.8 m |

`*` = early-stopped (patience 30). Latency: consistent runs ~6 ms on the 4060 Ti
(yolo11s @640); `noaug`/`domainaug`/`res_1280` bench numbers look inflated and
should be re-measured isolated — latency is not a Week-1 deliverable anyway
(`bench_jetson.py` is the instrument).

## Per-item verdict

| item | hypothesis | result | verdict |
|---|---|---|---|
| #2 learning rate | lr0=1e-3 beats the paper's 1e-5 | 1e-3 → 0.117 (worst); 1e-4 ≈ 1e-5 ≈ 0.29 | **falsified** — low LR wins; raising it destabilises on 727 imgs |
| #3 resolution | 640 → 960 → 1280 helps small objects | monotonic decline 0.282 → 0.248 → 0.144 | **falsified** — export is pre-resized; upscaling adds overfit, not detail |
| #5 augmentation | domain-matched aug helps | default > heavy (0.238) > none (0.196) | **partly** — aug helps (+0.086 vs none) but Ultralytics' default is already the sweet spot; heavy hurts |
| #1* thin classes | copy_paste + mixup lift WHV/WV | WHV **fell** to 0.045 (from 0.13) | **falsified** — WHV is a data/label problem, not an aug problem |
| size | bigger = better | s (0.282) > m (0.270) > n (0.230) | yolo11s is the sweet spot — and it's the deployable one |

## What's actually going on (for the writeup)

1. **The benchmark has a low ceiling (~0.30 mAP50-95) that recipe tuning cannot
   raise.** Every reasonable config lands in 0.24–0.30. That points at the data,
   not the optimiser.
2. **WHV (worker+helmet+vest) is broken everywhere: 0.04–0.13 AP50.** 48 train /
   33 test instances. No recipe fixes it.
3. **WV (worker+vest) is 0.995 in almost every run** — near-perfect and flat.
   That is not a healthy signal on 36 train instances; it suggests the class is
   trivially separable in this export or the split shares near-duplicates within
   the class. **This is the single most important thing for Week 2's label
   audit to resolve** — if WV is leaking, the whole 0.49 number is soft.
4. **yolo11s is both the accuracy sweet spot and the deployable size.** Convenient
   for the Amsar story; removes the "the model you want isn't the one you ship"
   tension for this dataset (KD in Week 5 becomes less critical, still worth a row).

## Recommended pick going into Week 2

**`w1_s_adamw_1e4`** as the working model: mAP50 tied with the best, highest
precision (0.599) and recall (0.498), trains in 7 min. Weights at
`runs/ppe_enh/w1_s_adamw_1e4/weights/best.pt`.

## Go / no-go

**No-go on Tier 1 as a source of mAP gain.** The gain was already banked by
training properly (0.14 → 0.29). The remaining leverage is entirely in
**data quality (Week 2)** and **honest evaluation** (person-level verdicts,
cross-dataset, corruption). Recommend proceeding straight to Week 2 with the
label audit as the priority — especially the WV=0.995 question.
