# Week 3 — corruption robustness (item #8)

`w3_corruptions`: 6 RTSP-realistic corruptions × 5 severities, applied in-memory
to the 316-image Pictor test set, scored **person-level** (violation recall,
undetected persons) under the `helmet` rule. Two checkpoints: the Week-1
Pictor-only baseline and the Week-4 Pictor+SH17 mixed model.

Figure: `figs/w3_corruption_recall.png`

## Why no mAP column

Weeks 1–4 established that object mAP does not track deployment quality on this
dataset (`w4_mixed_train` moved violation recall +0.11 while mAP50 stayed flat at
0.49). A corruption table in mAP would contradict the project's own headline.

## Why only the `helmet` rule

The `strict` rule's false-alarm denominator (FP+TN) on this test set is 4–9
persons (Week 4). It cannot resolve a corruption effect.

## Violation recall vs severity

Clean baselines: W1 = 0.344, W4 mixed = 0.457 (conf 0.1).

| corruption | model | sev1 | sev2 | sev3 | sev4 | sev5 |
|---|---|---|---|---|---|---|
| motion blur   | W1 | 0.255 | 0.177 | 0.091 | 0.056 | 0.026 |
|               | W4 | 0.435 | 0.329 | 0.191 | 0.132 | 0.102 |
| defocus       | W1 | 0.251 | 0.151 | 0.108 | 0.078 | 0.033 |
|               | W4 | 0.446 | 0.351 | 0.268 | 0.247 | 0.149 |
| JPEG          | W1 | 0.316 | 0.277 | 0.221 | 0.119 | 0.022 |
|               | W4 | 0.420 | 0.390 | 0.351 | 0.147 | 0.013 |
| low light     | W1 | 0.294 | 0.193 | 0.067 | 0.002 | 0.000 |
|               | W4 | 0.353 | 0.203 | 0.022 | 0.000 | 0.000 |
| overexposure  | W1 | 0.351 | 0.333 | 0.314 | 0.247 | 0.197 |
|               | W4 | 0.459 | 0.431 | 0.435 | 0.398 | 0.303 |
| dust / haze   | W1 | 0.348 | 0.336 | 0.325 | 0.305 | 0.238 |
|               | W4 | 0.455 | 0.450 | 0.431 | 0.418 | 0.377 |

## Fragility ranking (retention of clean recall at severity 3)

| corruption | W1 baseline | W4 mixed |
|---|---|---|
| dust / haze   | 94% | 94% |
| overexposure  | 91% | 95% |
| JPEG          | 64% | 77% |
| defocus       | 31% | 59% |
| motion blur   | 26% | 42% |
| **low light** | **20%** | **5%** |

## Findings

1. **Sharpness is the binding constraint, not colour.** Photometric corruptions
   (haze, overexposure) cost almost nothing — 94–95% retention at severity 3.
   Optical corruptions (motion blur, defocus) halve or worse. For the 8-camera
   Amsar install this makes **lens focus and shutter speed the highest-leverage
   physical variable**, ahead of lighting colour or camera placement.

2. **Low light is catastrophic for both models.** At severity 4 (12–20% luminance
   with matching sensor noise) violation recall is **0.002 / 0.000** — 987–994 of
   999 persons are never detected at all. Night operation on these checkpoints is
   not degraded, it is non-functional. Caveat: this corruption compounds
   darkening *and* shot noise; a real IR-illuminated camera would land nearer
   severity 1–2, where recall is 0.29–0.35.

3. **Mixing SH17 bought robustness, not just clean recall.** W4 mixed retains
   more of its clean recall than the baseline on every optical and compression
   corruption (defocus 59% vs 31%, motion blur 42% vs 26%, JPEG 77% vs 64%). The
   Week-4 gain was not a clean-image artifact.

4. **...except in low light, where mixing made it relatively worse** (5% vs 20%
   retention at sev3). SH17's PPE-bearing subset is bright industrial imagery;
   adding it appears to have biased the model toward well-lit scenes. This is the
   first cost identified for the SH17 mix, and it argues for dark/IR
   augmentation or dark training frames before night deployment.

## Excluded as unsound: false-alarm rate under corruption

False-alarm rate **must not be read from this benchmark**. Its denominator
(FP+TN) counts only persons the model actually detected, so it collapses exactly
when the model degrades: for the W1 baseline the denominator falls 334 → 291 →
238 → 124 → 78 → 49 across motion-blur severities, and low-light severity 5
leaves it at **0**, printing a meaningless FA of 0.000. The apparent "false
alarms improve under blur" (0.204 → 0.11) is the model detecting fewer people,
not judging them better. Violation recall and undetected-person counts are the
only sound metrics here because their denominator (999 ground-truth persons) is
fixed.

## Deployment implications

- Focus/shutter discipline > lighting colour. Verify lens focus per camera.
- Night is an open problem; do not ship these weights for after-dark monitoring.
- Compression to ~q25 is survivable (77% retention); below q15 it is not.

---

# Week 3b — `w3_darkaug`: the low-light fix that did not work

**Hypothesis:** the low-light collapse above is an augmentation gap, so retrain
the mixed recipe with aggressive brightness jitter (`hsv_v: 0.9`, i.e. value gain
0.1×–1.9×, which spans our low-light severities) and the night failure recovers.

**Result: falsified.** 50 epochs, 10.9 min. Verified by re-running the same
corruption benchmark (`w3_corrupt_dark`).

## It did not fix the night

| low-light severity | W4 mixed | W3 darkaug |
|---|---|---|
| 1 | 0.353 | 0.403 |
| 2 | 0.203 | 0.221 |
| 3 | 0.022 | 0.037 |
| 4 | 0.000 | 0.002 |
| 5 | 0.000 | 0.000 |

Severities 4–5 — actual darkness — remain **total failure**. The only movement is
at severity 1, which is dusk, not night. The target of the experiment was not hit.

Likely cause: `hsv_v` rescales brightness but adds no **sensor noise**, while the
benchmark's low-light model compounds darkening with noise that grows as the
signal shrinks. The augmentation trained for the wrong half of the corruption.

## It made everything else worse

Violation recall, W4 mixed → W3 darkaug, at severity 3:

| corruption | mixed | darkaug |
|---|---|---|
| motion blur | 0.191 | 0.110 |
| defocus | 0.268 | 0.217 |
| JPEG | 0.351 | 0.271 |
| overexposure | 0.435 | 0.372 |
| dust / haze | 0.431 | 0.367 |

And on clean images: object mAP50 0.495 → **0.432**, precision 0.651 → **0.484**,
WV AP50 0.995 → 0.774, SH17 cross-eval 0.587 → 0.557.

**Verdict: do not ship this checkpoint.** Aggressive brightness jitter bought a
marginal dusk improvement and paid for it everywhere else.

## The one genuine second-order finding

The dark model is markedly **more precise** per violation. Full 9-point sweep
(`w3_darkaug_honest_helmet.json`) against the mixed model, at matched recall:

| | recall | violation precision | FP | undetected |
|---|---|---|---|---|
| darkaug @ conf 0.10 | 0.407 | **0.803** | **46** | **410** |
| mixed @ conf 0.30 | 0.400 | 0.761 | 58 | 507 |
| mixed @ conf 0.10 | **0.457** | 0.723 | 81 | 386 |

At ~0.40 recall the dark model beats the mixed model on *both* false positives
and undetected persons. But its recall **caps at 0.407** — even at the lowest
threshold swept it never reaches the mixed model's 0.457.

So the tradeoff is real but narrow: darkaug is the better choice only if ~0.40
violation recall is acceptable and false alarms are expensive. For maximum
violation catch, W4 mixed remains the model. This does not rescue the
experiment — it failed its stated hypothesis.

## What would actually be needed for night operation

1. Augmentation that models **darkening plus sensor noise together**, not
   brightness alone.
2. Failing that, real night/IR frames in training — the Week-4 lesson (data, not
   recipe) applying again.
3. Until then, the honest deployment statement is unchanged: **these checkpoints
   do not work after dark.**
