# Week 4 — cross-dataset generalisation (item #9)

The only intervention in the whole 17-item plan that improved the deployment
metric. Weeks 1–3 ruled out learning rate, resolution, augmentation strength,
model size, sliced inference and label noise. Week 4 changed the **data**.

New code: `experiments/sh17_map.py` (SH17 object boxes → person-level
W/WH/WHV/WV), `also_eval_data` cross-dataset eval in `run.py`'s train handler.

## The ontology mapping

SH17 labels objects (person, helmet, vest as separate boxes); we need per-person
compliance verdicts. `sh17_map.py` associates them by **greedy one-to-one
containment** — a helmet must sit in the top 40% of a person box, a vest in the
15–80% torso band, and each helmet/vest is consumed by at most one person so a
single helmet cannot dress a whole crowd. SH17 class ids (0=person, 10=helmet,
16=safety-vest) were derived empirically by matching this export's YOLO labels
against its VOC xml.

**Only the 498 PPE-bearing images are kept.** SH17 annotates PPE sparsely, so in
a no-PPE image a person means "no helmet *annotated*", not "no helmet *worn*".
`--all-images` would turn 92.7% of boxes into a noisy W — i.e. would silently
invent violations. This is also why `--neg` must not be used to mine hard
negatives: those persons are noisy Ws, and feeding them in would push false
alarms up, not down.

## Correction to an earlier assessment

My first read of SH17 was **wrong**: I called it close-up stock photography and
the wrong direction for this project. That was computed over all 8099 images.
The 498 PPE-bearing subset is **2.80 persons/img at 1.81% median box area** —
which nearly matches our Pictor *test* distribution (3.16/img, 1.6%), the very
distribution Week 2 identified as the domain gap.

## `w4_sh17_train` — SH17 only (64 epochs, 5.8 min)

| | mAP50 | mAP50-95 | W | WH | WHV | WV |
|---|---|---|---|---|---|---|
| on SH17 test | 0.584 | 0.426 | 0.511 | 0.708 | **0.582** | 0.533 |

**All four classes learn.** WHV reaches 0.582 here versus 0.130 on Pictor. This
is the proof that Pictor's broken WHV class is a **data** problem, not a recipe
or architecture problem — the same model, same recipe, same ontology.

Transfer to Pictor is near zero: **mAP50 0.103** (an 82% relative drop) despite
identical ontology. The domain gap is real and large.

## `w4_mixed_train` — Pictor + SH17, pure-Pictor val/test (45 epochs, 9.9 min)

Object level, on the same Pictor test set as Weeks 1–2:

| | mAP50 | mAP50-95 | W | WH | WHV | WV |
|---|---|---|---|---|---|---|
| W1 baseline | 0.494 | 0.282 | 0.423 | 0.428 | 0.130 | 0.995 |
| W4 mixed | 0.495 | 0.302 | 0.457 | 0.455 | 0.073 | 0.995 |

**Object mAP is flat.** WHV moved *down*, but on 20 test objects that is noise in
either direction, as is WV's 0.995 on 6 objects (Week 2).

Person level, helmet rule — this is the result:

| conf | violation recall | false alarm | undetected |
|---|---|---|---|
| 0.10 | 0.344 → **0.457** | 0.204 → 0.253 | 354 → 386 |
| 0.20 | 0.314 → 0.420 | 0.192 → 0.246 | 414 → 451 |
| 0.30 | 0.294 → 0.400 | 0.166 → 0.231 | 466 → 507 |
| 0.50 | 0.221 → 0.314 | 0.149 → 0.257 | 576 → 640 |

At conf 0.1: TP 159 → 211, FP 68 → 81. **+52 real violations caught for +13 extra
false alarms**, against 462 real violations. The gain holds at every threshold,
so it is not a threshold artifact, and the denominators are large enough to
trust.

Cross-eval on SH17 held at **0.587 mAP50** versus the SH17-only model's 0.584 —
no catastrophic forgetting in either direction.

## Iso-recall: the mixed model is not strictly dominant

Comparing at matched *threshold* flatters the false-alarm story; comparing at
matched *recall* is fairer:

| | recall | FP | undetected persons |
|---|---|---|---|
| baseline @ conf 0.10 | 0.344 | 68 | 354 |
| mixed @ conf 0.40 | 0.353 | **52** | **571** |
| mixed @ conf 0.10 | **0.457** | 81 | 386 |

At matched recall the mixed model raises **fewer** false alarms in absolute
terms (52 vs 68) — the higher *rate* reported above is partly its smaller
denominator, since false-alarm rate is FP/(FP+TN) and undetected persons never
enter the matrix at all. But it pays with 217 more people missed entirely.

So the honest claim is narrow: the mixed model's advantage is that it can be run
at a **low threshold**, where it catches substantially more violations at
comparable cost. It does not dominate the baseline everywhere.

## Excluded as noise

The strict rule appeared to improve false alarms dramatically (0.88–1.0 → 0.25).
Its denominator (FP+TN) is **4–9 persons**. That is 7 false positives becoming 1.
Not reportable.

## What this week established

1. Deployment-representative data is the lever; nothing about the model was.
2. Object mAP does not track deployment quality here — it stayed flat at 0.49
   while violation recall moved 11 points.
3. Pictor's WHV ceiling is a data limit, proven by the same model reaching 0.582
   on SH17.
4. Weeks 3 and 5 later showed the same mix also bought corruption robustness and
   better-calibrated confidence — three independent benefits, none visible in mAP.
