# Amsar vs PPE-EYE — full head-to-head (detection + efficiency)

Date: 2026-09-10 · GPU: RTX 4060 Ti · imgsz 640 · IoU 0.5 · rule = `helmet`

Extends `AMSAR_COMPARISON.md`, which measured accuracy on Pictor only and closed
by saying *"the genuinely fair architectural test has not been run."* This
document runs it, and adds the cost side that was missing entirely.

Three things are new here:

1. **A fair test bed (SH17).** Neither system has trained on it — `ppe.pt` never
   saw it, and the PPE-EYE model used here is Pictor-only. No home-field advantage.
2. **Trivial baselines.** What a zero-intelligence classifier scores on each set.
3. **Efficiency.** ms/frame, passes, params, GFLOPs, VRAM, 8-camera capacity.

---

## The two systems

| | Amsar | PPE-EYE |
|---|---|---|
| shape | 2-model cascade | 1 model |
| stage 1 | stock COCO `yolov8n.pt` → persons (conf 0.4) | one pass → person box **already carries** the verdict |
| stage 2 | `ppe.pt` → 6 PPE classes (conf 0.40) | — |
| stage 3 | `_associate_ppe()` — geometry, first match wins | — |
| ontology | compliance **inferred** from presence/absence of a PPE box | compliance **is** the class (W/WH/WHV/WV) |
| negative class | **none** — "no helmet" and "helmet missed" are the same signal | explicit — the model is trained to say *bare head* |

That missing negative class is the structural crux, and it drives most of what follows.

---

## Test bed 1 — Pictor test (PPE-EYE's home turf)

316 images · 999 persons · violation base rate **0.462**

| system | conf | viol. recall | false alarm | verdict acc |
|---|---|---|---|---|
| NULL — persons only, all=violation | 0.1 | 0.463 | 1.000 | 0.339 |
| AMSAR as deployed (800×448) | 0.4 | 0.277 | 0.823 | 0.263 |
| AMSAR native res (generous) | 0.1 | 0.442 | 0.812 | 0.373 |
| PPE-EYE w1 baseline | 0.1 | 0.344 | **0.204** | 0.534 |
| **PPE-EYE w4 mixed** | 0.1 | **0.457** | **0.253** | **0.575** |

PPE-EYE wins decisively — same recall, **3.2× lower false-alarm rate**.
But it trained on these images and `ppe.pt` did not, so this table alone
proves nothing about architecture.

---

## Test bed 2 — SH17, 498 images (FAIR: both out-of-domain)

498 images · 1392 persons · violation base rate **0.413** · classes well balanced
(390 W / 549 WH / 268 WHV / 185 WV)

Neither system has seen a single one of these images.

| system | viol. recall | false alarm | verdict acc | helmet boxes found |
|---|---|---|---|---|
| NULL — persons only, all=violation | 0.584 | 1.000 | 0.270 | 0 (by construction) |
| **AMSAR cascade** | **0.570** | **0.886** | **0.325** | 90 |
| **PPE-EYE (Pictor-only)** | **0.151** | **0.160** | **0.523** | n/a |
| *trivial: always say "violation"* | 1.000 | 1.000 | 0.413 | — |
| *trivial: always say "compliant"* | 0.000 | 0.000 | **0.587** | — |

### This is the most important table in the document

**Neither system beats a constant.** A classifier that ignores the image and
says *"everyone is compliant"* scores 0.587 verdict accuracy on SH17. PPE-EYE
scores 0.523. Amsar scores 0.325. Both are **below** the dumbest possible
baseline on data they weren't trained on.

They fail in opposite directions, which is why raw recall numbers mislead here:

- **Amsar cries wolf.** 0.886 false-alarm rate — it flags almost every compliant
  worker as a violation. Its 0.570 recall looks respectable only because
  flagging everyone catches most violations by default.
- **PPE-EYE stays silent.** 0.160 false-alarm rate, but 0.151 recall — it clears
  almost everyone, missing 85% of real violations.

Amsar's high recall on this table is **not detection skill**. Compare it to the
null control: person-detector alone, everyone labelled a violation, scores
recall 0.584 / FA 1.000. Amsar's full cascade scores 0.570 / 0.886.

> **Amsar's entire PPE stage buys an 11-point false-alarm reduction and costs
> 1.4 points of recall.** It is close to a no-op — the same finding as on Pictor,
> now confirmed on a second, balanced, fair dataset.

The diagnostics say why: **90 helmet boxes across 498 images**, against 817
helmet-wearing workers in the ground truth. `ppe.pt` essentially does not fire.
Because it has no negative class, a missed helmet is indistinguishable from an
absent helmet, so every miss becomes a false alarm. The 0.886 FA rate is that
mechanism, measured.

---

## Efficiency (150 real frames, warmed up, same GPU)

| system | ms/frame | p95 | FPS | passes | params | GFLOPs | file | VRAM | fps/cam @8 cams |
|---|---|---|---|---|---|---|---|---|---|
| AMSAR native | 11.64 | 13.13 | 85.9 | **2** | 5.85M | 15.81 | 12.2 MB | 46 MB | 10.7 |
| AMSAR as deployed (800×448) | 10.12 | 11.38 | 98.9 | **2** | 5.85M | 15.81 | 12.2 MB | 60 MB | 12.4 |
| PPE-EYE yolo11**n** | **6.95** | 8.19 | 143.9 | **1** | **2.59M** | **6.50** | **5.5 MB** | 153 MB | **18.0** |
| PPE-EYE yolo11**s** | 6.98 | 8.42 | 143.2 | 1 | 9.43M | 21.67 | 19.2 MB | 162 MB | 17.9 |
| PPE-EYE yolo11s (w4 mixed) | 6.78 | 7.62 | 147.6 | 1 | 9.43M | 21.67 | 19.2 MB | 210 MB | 18.4 |

Amsar's per-stage split — `person 5.90ms + ppe 5.63ms + assoc 0.06ms`. The two
forward passes cost almost exactly the same, so the second model **doubles**
inference for the near-zero accuracy gain measured above.

**PPE-EYE yolo11n is the standout on cost:** fewer params than Amsar's *person
detector alone* (2.59M vs 3.16M), 2.4× less compute than Amsar's two models
combined (6.50 vs 15.81 GFLOPs), half the disk, and **1.7× faster end-to-end**.

Two honest caveats:
- **VRAM is Amsar's one win** (46–60 MB vs 153–210 MB). Peak activation memory
  at 640, not weight size. Irrelevant on any Jetson with headroom; worth checking
  if you're near the ceiling with 8 concurrent streams.
- The association loop (0.06ms) is free — it is **not** the bottleneck. The
  second forward pass is.

---

## Verdict

**On architecture and cost, PPE-EYE is the better chassis — but do not ship
either set of weights as-is.**

Where PPE-EYE clearly wins:

1. **Half the inference.** One pass, not two. 1.7× faster; yolo11n does it with
   2.4× less compute than Amsar's cascade.
2. **Specificity.** 3.2× lower false alarms in-domain, 5.5× lower out-of-domain.
   A PPE system that alarms on 89% of compliant workers gets switched off by the
   site in a week — this is the difference between a product and a nuisance.
3. **The ontology is right.** Learning the verdict directly gives you a real
   negative class. Amsar's inferred-from-parts design *cannot* distinguish
   "no helmet" from "helmet missed" — that is a design limit, not a tuning issue.
4. **No geometry heuristics to break.** No top-45%/25–75% band assumptions that
   fail on crouching, occluded, or overhead-angle workers.

What the fair test also proves, and must not be glossed over:

5. **PPE-EYE's current weights don't generalise either.** 0.151 recall on SH17 —
   below the always-say-compliant baseline. Week 4 already measured this
   (SH17→Pictor transfer, mAP50 0.103, an 82% relative drop). Swapping Amsar's
   cascade for today's PPE-EYE checkpoint would trade a false-alarm problem for a
   missed-violation problem, which in a safety product is the more dangerous one.

**So the bottleneck is data, not architecture.** Both architectures collapse
out-of-domain; only one of them is cheap, specific, and structurally capable of
saying "that worker has no helmet."

### Recommended plan

1. **Adopt the PPE-EYE single-model architecture** into Amsar, on **yolo11n** —
   it is smaller and faster than what Amsar ships today *and* removes the
   second forward pass.
2. **Do not adopt the weights.** Retrain on Pictor + SH17 + CHV combined. The
   `w4_mixed` run is the template; it already leads on Pictor (0.575 verdict acc).
3. **Gate the swap on an OOD acceptance number.** Pick a target on a held-out
   dataset neither model trained on — e.g. verdict accuracy ≥ 0.70, comfortably
   above the 0.587 trivial baseline — and don't ship until a checkpoint clears it.
   Today nothing does.
4. **Keep Amsar's `PPESmoother`.** Temporal voting is a genuine strength of the
   product and is orthogonal to the detector. It is untested here (stills only)
   and should cut false alarms further on video.
5. **Skip the `_associate_ppe` 1-to-1 fix.** Measured at +0.002 recall / −0.023 FA
   — noise. It also becomes moot the moment the cascade is removed.

---

## Reproduce

```bash
# Fair out-of-domain head-to-head (SH17, 498 imgs, neither system trained on it)
PPE/Scripts/python.exe experiments/amsar_eval.py  --data data/sh17_compliance/data_ood.yaml \
    --split test --rule helmet --out experiments/results/ood_sh17_amsar_helmet.json
PPE/Scripts/python.exe experiments/honest_eval.py --weights runs/ppe_enh/w1_s_baseline_repro/weights/best.pt \
    --data data/sh17_compliance/data_ood.yaml --split test --rule helmet \
    --out experiments/results/ood_sh17_ppeeye_helmet.json
# Null control: person detector only, everyone = violation
PPE/Scripts/python.exe experiments/amsar_eval.py  --data data/sh17_compliance/data_ood.yaml \
    --split test --rule helmet --ppe-conf 0.99 --out experiments/results/ood_sh17_null_helmet.json
# Efficiency
PPE/Scripts/python.exe experiments/bench_efficiency.py --n 150 --imgsz 640
```

`data_ood.yaml` deliberately points at all 498 SH17 images. It is only valid with
Pictor-only weights — **never** use it with `w4_sh17_train` / `w4_mixed_train`,
which trained on part of that set.

## Limits of this comparison

- One rule (`helmet`). The `strict` rule's false-alarm denominator on Pictor is
  4–9 persons — too thin to quote. SH17 is balanced enough to support `strict`;
  that run has not been done.
- Stills, not video. Amsar's `PPESmoother` is unmeasured (see plan item 4).
- Desktop RTX 4060 Ti, not Jetson. Ratios should hold; absolute ms will not.
  `bench_jetson.py` exists for the real number, and TensorRT FP16 (which Amsar
  supports via `export_trt.py`) will shift both systems.
- Amsar's PPE stage is held at its shipped `PPE_HAT_CONF = 0.40`, as deployed.
  A sweep to 0.01 was run previously and tops out at 411 helmet boxes, mostly noise.
