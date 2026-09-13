# Week 5 — confidence calibration (item #12)

`w5_calibration`. A detection counts as correct if it matches a ground-truth box
of the **same class** at IoU ≥ 0.5, assigned greedily in descending score order
(Küppers et al. 2020). Calibration asks whether the score predicts that: among
boxes scored 0.7, are 70% right? Temperature is fitted on **val**, reported on
**test**. Both checkpoints evaluated.

Figure: `figs/w5_calibration.png`. Raw detection arrays cached in
`results/calib_cache/*.npz`, so re-analysis needs no GPU.

## What temperature scaling cannot do

It is a monotone map. It cannot change detection ranking, so it cannot move
violation recall, false-alarm rate, or the tradeoff between them. It only
changes what a threshold *means*. Nothing in this week improves accuracy.

## Global ECE is an artifact — do not quote it

At the 0.01 collection floor, **65–67% of all detections score below 0.067**.
They are nearly all wrong and they say so, i.e. trivially calibrated. An
n-weighted ECE over all detections is dominated by boxes no operator ever sees:

| | all detections | conf ≥ 0.2 (operating regime) |
|---|---|---|
| W1 baseline ECE | 0.0226 | **0.0773** |
| W4 mixed ECE | 0.0233 | **0.0478** |

Globally the two models look identically calibrated. In the regime that is
actually thresholded they differ by 60%. Every number below is operating-regime.

## The baseline is overconfident; the mixed model is not

Signed gap = observed precision − claimed confidence, per bin (test):

| conf bin | W1 baseline | W4 mixed |
|---|---|---|
| 0.20–0.27 | −0.090 | +0.024 |
| 0.27–0.33 | −0.071 | +0.023 |
| 0.53–0.60 | −0.057 | −0.054 |
| 0.67–0.73 | −0.111 | +0.009 |
| 0.73–0.80 | **−0.185** | +0.102 |
| 0.87–0.93 | −0.066 | −0.101 |

| model | regime ECE | regime MCE | signed bias |
|---|---|---|---|
| W1 baseline | 0.0773 | 0.185 | **−0.073** (overconfident) |
| W4 mixed | 0.0478 | 0.102 | **+0.017** (mildly conservative) |

The W1 baseline is overconfident in **every** bin from 0.2 up; at 0.73–0.80 it
claims 0.77 and delivers 0.58. The W4 mixed model is near-calibrated and errs
conservative — the correct direction for a safety system.

**This is the third independent benefit of the SH17 mix**, after clean-image
violation recall (Week 4) and corruption robustness (Week 3). None of the three
is visible in object mAP, which stayed flat at 0.49.

## Temperature scaling: null result

Fitted on val, fixed test population (selected on raw conf ≥ 0.2):

| model | T | ECE raw → scaled | MCE raw → scaled |
|---|---|---|---|
| W1 baseline | 1.259 | 0.0773 → 0.0735 | 0.185 → **0.223** (worse) |
| W4 mixed | 1.199 | 0.0478 → 0.0398 | 0.102 → 0.075 |

A scalar temperature removes a small uniform bias and little else. Panel (b)
shows why: the miscalibration is **not monotone** in confidence — W4 mixed runs
+0.10 at bin 0.73–0.80 then −0.10 at 0.87–0.93. One global parameter cannot
correct a sign change. Per-bin or per-class isotonic regression would be the
next step if this mattered; on this evidence it does not.

Fitting T over *all* detections is actively harmful — it optimises the sub-0.067
mass and pushes global test ECE **up** (0.0226 → 0.0329 baseline, 0.0233 →
0.0272 mixed) while improving val ECE. Val is drawn from the same easy
distribution as train (Week 2: 1.46 workers/img @ 6.5% box area) while test is
crowded and distant (3.16/img @ 1.6%), so a val-fitted temperature does not
transfer.

## The useful output: threshold → actual precision

Deployment can now pick a threshold by policy instead of by sweeping.

| threshold | W1 baseline (n / precision) | W4 mixed (n / precision) |
|---|---|---|
| 0.10 | 1357 / 0.358 | 1179 / 0.417 |
| 0.20 | 976 / 0.445 | 831 / 0.523 |
| 0.30 | 738 / 0.541 | 647 / 0.595 |
| 0.40 | 611 / 0.578 | 514 / 0.648 |
| 0.50 | 486 / 0.619 | 403 / 0.692 |
| 0.60 | 381 / 0.651 | 281 / 0.758 |
| 0.70 | 266 / 0.688 | 177 / 0.791 |

W4 mixed is more precise at **every** threshold. To hold ~60% box precision the
baseline needs conf ≈ 0.47; the mixed model reaches it at ≈ 0.31, and the lower
threshold is what buys the extra violation recall reported in Week 4.

## Per-class

WHV is badly overconfident on the mixed model — mean confidence 0.164 against
0.061 precision, class ECE 0.138 (n = 49 detections). Consistent with every
earlier week: WHV is the broken class and the cause is data, not recipe. **WV is
not reported** — 17–20 detections, below the n ≥ 30 guard.
