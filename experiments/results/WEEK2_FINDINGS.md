# Week 2 findings — data quality & honest evaluation

Model under test: `runs/ppe_enh/w1_s_adamw_1e4/weights/best.pt` (Week-1 pick,
mAP50 ≈ 0.49 / mAP50-95 ≈ 0.28).

## 1. The Week-1 "WV = 0.995" mystery: solved by counting, not auditing

Real per-split object counts (`w2_label_audit_composition.json`):

| split | images | objects | obj/img | W | WH | WHV | WV |
|---|---|---|---|---|---|---|---|
| train | 727 | 1059 | 1.46 | 422 | 520 | **53** | **64** |
| val | 154 | 999 | 6.49 | 495 | 459 | **27** | **18** |
| test | 316 | 999 | 3.16 | 456 | 517 | **20** | **6** |

WV has **6 test objects**, WHV has **20** — both under the "< 30 → AP is noise"
threshold. The 0.995 was never leakage; it is two or three easy boxes (times the
~3× within-split augmentation) landing right. **No manual audit needed.** The
consequence: the 0.49 mAP50 is really just W + WH; WHV and WV contribute noise,
and WHV — the only *compliant* class — has 53 training instances total.

## 2. Train ≠ test distribution (the real ceiling)

| | train | test |
|---|---|---|
| workers per image | **1.46** | 3.16 |
| median box area (frac of image) | **6.5 %** | 1.6 % |
| p10 box area | 0.46 % | 0.16 % |

Train images are sparse, close-up singletons. Test images are crowded and
distant. The model learns easy close-ups and is scored on small clustered
workers. This — not the optimiser — is why Week 1 hit a wall, why higher
resolution overfit instead of helping, and why 27–35 % of test workers are
never detected.

## 3. Person-level honest eval — the mAP → decision gap, quantified

`experiments/honest_eval.py`, confidence sweep, IoU 0.5. Positive class =
violation (non-compliant worker).

**Strict rule (compliant ⇔ WHV: helmet AND vest)**

| conf | violation recall | false-alarm rate | undetected persons |
|---|---|---|---|
| 0.10 | 0.649 | 0.875 | 354 / 999 |
| 0.30 | 0.535 | 1.000 | 466 |
| 0.50 | 0.426 | 1.000 | 576 |

**Helmet rule (compliant ⇔ WH or WHV: helmet only)**

| conf | violation recall | false-alarm rate | undetected persons |
|---|---|---|---|
| 0.10 | 0.344 | 0.204 | 354 |
| 0.30 | 0.294 | 0.166 | 466 |
| 0.50 | 0.221 | 0.149 | 576 |

Read-out:
- **mAP50 ≈ 0.49 → violation recall 0.34–0.65** depending on rule and threshold.
- **35 % of workers undetected at the most permissive threshold, 58 % at conf 0.5.**
  mAP hides this entirely — it only scores boxes that exist.
- **Strict-rule false-alarm hits 1.0**: the model almost never emits WHV (53
  train boxes), so every genuinely compliant worker is flagged. Under the strict
  rule this system would nuisance-trip on every compliant worker.
- Helmet rule is the only defensible operating mode on this data, and even there
  peak recall is 0.34.

Figure: `figs/w2_honest_eval.png`.

## 4. Label noise is a minor contributor

Conservative auto-clean (drop GT boxes that are BOTH `tiny` < 0.3 % area AND
invisible to the model at conf 0.05) removed **88 of 999** test boxes.

| rule | violation recall (noisy → clean) | verdict acc (noisy → clean) | undetected (noisy → clean) |
|---|---|---|---|
| strict | 0.649 → 0.711 (+0.063) | 0.644 → 0.706 | 354 → 266 |
| helmet | 0.344 → 0.362 (+0.018) | 0.534 → 0.550 | 354 → 266 |

So ~6 points of the strict-rule verdict accuracy is annotation dust. The
remaining **266 undetected persons (27 %) are real misses of visible workers** —
distribution mismatch, not label noise.

Flag tally over 999 test boxes (`w2_label_audit_flags.csv`, 200-image review
queue): edge 277, class_disagree 241 (mostly W↔WH), no_model_support 172,
tiny 163, aspect 130, unlabeled_pred 19.

## 5. SAHI sliced inference (item #4) — does NOT help here

`experiments/sahi_eval.py`, sliced prediction reusing the honest-eval scorer.

| rule | undetected: full-frame → SAHI | violation recall: full → SAHI |
|---|---|---|
| strict | 354 → **437** | 0.649 → 0.559 |
| helmet | 354 → **437** | 0.344 → 0.320 |

Tried 320 px and 480 px slices — both worse than full-frame. Slicing is the
standard small-object recovery, but it assumes the detector generalises across
scale. This model was trained on 640 px frames with large subjects (6.5 % median
box area); on tiles it sees people at scales and crop-states it never trained on,
and tile boundaries cut workers the model wants whole. **Not an upper bound — a
null result**, and it reinforces §2: tricks that don't touch the training
distribution don't move the number.

## Go / no-go

| item | verdict |
|---|---|
| #6 label audit | **done** — composition explains Week 1; label noise worth ~6 pts; distribution mismatch is the real issue |
| #6 honest eval | **done** — mAP 0.49 ⇒ recall 0.34 (helmet) / undetected 27–58 %. This is the headline result of the whole project so far |
| #7 hard negatives | **low value now** — addresses false *positives*; our failure mode is false *negatives* + unlearned WHV. Defer. |
| #4 SAHI upper bound | **worth doing** — directly targets the small-distant-worker misses that dominate. Run as the upper-bound row. |

**The project's real finding is now clear:** on this dataset a properly trained
YOLO11s reaches mAP50 ≈ 0.49 but only ~0.34 person-level violation recall and
misses a quarter to a half of all workers, because the training data (sparse,
close-up) does not resemble deployment (crowded, distant) and the compliant
class is data-starved. No training-recipe or architecture change addresses this;
it needs deployment-representative labelled data.
