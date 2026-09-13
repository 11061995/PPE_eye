# Amsar detection layer vs PPE-EYE — head-to-head on the Pictor test split

> **SUPERSEDED IN PART — see `HEAD_TO_HEAD.md`.** The fair architectural test
> this document calls for (below, "Fairness caveat") has now been run on SH17,
> where neither system has trained, together with the efficiency comparison that
> was missing here. Headline change: on fair out-of-domain data **neither system
> beats a constant classifier** — Amsar over-alarms (FA 0.886), PPE-EYE
> under-alarms (recall 0.151). The Pictor numbers below remain valid as the
> in-domain picture.

Date: 2026-09-10
Harness: `experiments/amsar_eval.py` → `experiments/honest_eval.py`
Split: `data/test` (316 images, ~1168 GT persons), IoU 0.5, rule = `helmet`

## What was run

Amsar's detection layer was pulled out of `Amsar-AI-/` and wrapped so it emits
the same W/WH/WHV/WV person-level verdicts PPE-EYE emits. Same images, same IoU
matching, same rule, same metrics — only the algorithm differs.

Amsar's algorithm, as implemented in `cameras.py` / `ppe_models.py`:

1. stock COCO `yolov8n.pt` → person boxes (`CONF = 0.4`)
2. `ppe.pt` → PPE object boxes (`PPE_HAT_CONF = 0.40`)
   classes: `{0:Gloves, 1:Vest, 2:goggles, 3:helmet, 4:mask, 5:safety_shoe}`
   → derived roles: `HAT={3} NO_HAT=∅ VEST={1} GLOVES={0}`
3. `_associate_ppe()` — geometry: hat centre in person's top 45%, vest centre in
   the 25–75% band, glove centre anywhere in box. First match wins, box not consumed.

`NO_HAT` is empty: **ppe.pt has no negative class**, so "no helmet" and "helmet
missed" are the same signal.

`_associate_ppe` and `_derive_ppe_classes` are copied verbatim into the harness.

## Results (helmet rule)

| system | conf | viol. recall | false alarm | verdict acc | undetected | TP | FP | TN |
|---|---|---|---|---|---|---|---|---|
| NULL — person only, all=violation | 0.1 | 0.463 | **1.000** | 0.339 | 615 | 214 | 170 | 0 |
| NULL — person only, all=violation | 0.4 | 0.346 | 1.000 | 0.280 | 730 | 160 | 109 | 0 |
| AMSAR as deployed (800×448) | 0.1 | 0.387 | 0.881 | 0.330 | 690 | 179 | 111 | 15 |
| AMSAR as deployed (800×448) | **0.4** | **0.277** | **0.823** | 0.263 | 788 | 128 | 65 | 14 |
| AMSAR native-res (generous) | 0.1 | 0.442 | 0.812 | 0.373 | 615 | 204 | 138 | 32 |
| AMSAR + 1-to-1 assoc fix | 0.1 | 0.444 | 0.835 | 0.369 | 615 | 205 | 142 | 28 |
| AMSAR + ppe_conf 0.10 | 0.1 | 0.405 | 0.706 | 0.375 | 615 | 187 | 120 | 50 |
| PPE-EYE w1 baseline | 0.1 | 0.344 | 0.204 | 0.534 | 354 | 159 | 68 | 266 |
| **PPE-EYE w4 mixed (best)** | **0.1** | **0.457** | **0.253** | **0.575** | 386 | 211 | 81 | 239 |

## Findings

### 1. Amsar's PPE stage barely fires on this footage
`ppe.pt` produced **50 helmet boxes and 9 vest boxes across 316 images**, against
~537 GT helmet-wearing workers. Verified not to be a broken integration — a conf
sweep down to 0.01 tops out at 411 helmet boxes, mostly noise:

| ppe conf | 0.40 | 0.25 | 0.10 | 0.05 | 0.01 |
|---|---|---|---|---|---|
| helmet boxes | 50 | 76 | 134 | 187 | 411 |
| vest boxes | 9 | 9 | 16 | 21 | 82 |

### 2. The null control is the headline
A person detector alone, labelling **everyone** a violation, scores recall 0.463 /
FA 1.000. Amsar's full cascade scores 0.442 / 0.812 (native) and 0.387 / 0.881
(as deployed).

**Amsar's entire PPE stage buys a false-alarm reduction of 1.000 → 0.812, and
costs 2 points of recall.** On this footage it is close to a no-op.

### 3. The gap is specificity, not sensitivity
At essentially identical violation recall (0.457 vs 0.442), PPE-EYE's false-alarm
rate is **3.2× lower** (0.253 vs 0.812) and verdict accuracy is 0.575 vs 0.373.

Look at TN: of ~170 genuinely compliant workers detected at conf 0.1, PPE-EYE
correctly clears 239; Amsar clears 32. Amsar can barely tell a compliant worker
from a non-compliant one — it mostly says "violation" and is right at the base rate.

### 4. The 800×448 capture downscale is a real, measured cost
`FrameReader` decodes to 800×448 before inference. That alone costs:
undetected 615 → 690, recall 0.442 → 0.387, FA 0.812 → 0.881.

### 5. CORRECTION — the first-match association bug is real but immaterial
Fixing `_associate_ppe` to greedy 1-to-1 moved recall 0.442 → 0.444 and FA
0.812 → 0.835 (i.e. nothing, slightly worse). The bug is genuine in code, but with
only 50 helmet boxes over 316 images there is almost no contention for them to
be double-counted. **Deprioritise this fix** — it only matters once the PPE model
actually detects things.

### 6. Person stage: yolov8n is worse than the fine-tuned model, as expected
Undetected persons at conf 0.1: 615 (Amsar, stock yolov8n) vs 386 (PPE-EYE w4).
At Amsar's shipped conf 0.4: **730 of ~1168 GT persons (63%) are never detected**,
rising to 788 (67%) at the deployed 800×448 capture size.

## Fairness caveat — read this before quoting any of the above

This is **in-domain vs out-of-domain**. PPE-EYE's model was trained on Pictor
train; `ppe.pt` has never seen Pictor. That is a large home-field advantage and
this experiment does **not** prove PPE-EYE's architecture is superior.

What it does establish:

- Amsar's **currently shipped weights** are unfit for crowded/distant site
  footage of this kind, by a wide margin.
- Amsar ships one fixed `ppe.pt` to every client with no per-site training, so
  every new site is an out-of-domain deployment. The out-of-domain number is
  therefore the deployment-relevant one for Amsar **as currently shipped**.
- Nobody is immune to this: Week 4 measured SH17→Pictor transfer for the PPE-EYE
  architecture at mAP50 0.103, an 82% relative drop.

**The genuinely fair architectural test has not been run.** It would train a
helmet/vest *object* detector on Pictor train, run Amsar's cascade on it, and
compare against PPE-EYE's person-level model trained on the same images. That
isolates ontology (inferred-from-parts vs learned-verdict) from weights quality.
Recommended as the next experiment.

## Reproduce

```bash
PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet --capture amsar   # as deployed
PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet                   # native res
PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet --assoc greedy    # 1-to-1 fix
PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet --ppe-conf 0.99   # null control
```

Strict rule not reported: its FA denominator on this split is 4–9 persons (see
WEEK2_FINDINGS.md). The helmet rule is the only well-supported one here.
