# PPE-EYE enhancement — results

Prior best (lightaug_yolo11s): **mAP50-95 ≈ 0.14**. `Δ` below is vs that.

## Week 1

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w1_s_baseline_repro | 2 | done | 0.4941 | 0.2954 | 0.5869 | 0.4748 | 6.7000 | 149.2 | 14.8000 | +0.1554 | Paper's lr0=1e-5 NAdam cell on our split - expected to barely move off COCO init |
| w1_s_sgd_1e2 | 2 | done | 0.3707 | 0.2189 | 0.3755 | 0.4142 | 6.1500 | 162.7 | 11.8000 | +0.0789 | SGD lr0=1e-2, cosine, warmup 3, close_mosaic 10 |
| w1_s_adamw_1e3 | 2 | done | 0.2685 | 0.1167 | 0.3174 | 0.3930 | 5.9800 | 167.2 | 15.8000 | -0.0233 | AdamW lr0=1e-3, cosine - the expected LR winner and base recipe for later arms |
| w1_s_adamw_1e4 | 2 | done | 0.4938 | 0.2822 | 0.5993 | 0.4976 | 6.0800 | 164.5 | 6.8000 | +0.1422 | AdamW lr0=1e-4, cosine |
| w1_s_res_960 | 3 | done | 0.4614 | 0.2480 | 0.3976 | 0.5086 | 9.8600 | 101.4 | 13.0000 | +0.1080 | Base recipe at imgsz=960 - small-object resolution sweep |
| w1_s_res_1280 | 3 | done | 0.4242 | 0.1437 | 0.4794 | 0.4538 | 21.5500 | 46.4000 | 28.4000 | +0.0037 | Base recipe at imgsz=1280 - small-object resolution sweep |
| w1_s_noaug | 5 | done | 0.4038 | 0.1961 | 0.4707 | 0.4602 | 14.4000 | 69.4000 | 7.0000 | +0.0561 | Base recipe, all online augmentation OFF - the control arm for item #5 |
| w1_s_domainaug | 5 | done | 0.4515 | 0.2382 | 0.5179 | 0.4778 | 13.4200 | 74.5000 | 12.5000 | +0.0982 | Base recipe + strong HSV-V (glare), scale (distance), erasing (occlusion) |
| w1_s_imbalance | * | done | 0.4528 | 0.2923 | 0.4670 | 0.4816 | 5.9400 | 168.5 | 6.8000 | +0.1523 | Base recipe + copy_paste + mixup to lift thin classes WHV (48) / WV (36) |
| w1_n_adamw_1e3 | 3 | done | 0.4169 | 0.2302 | 0.4766 | 0.4599 | 5.8800 | 170.0 | 5.7000 | +0.0902 | yolo11n base recipe - deployable size point for the 8-camera budget |
| w1_m_adamw_1e3 | 3 | done | 0.4610 | 0.2697 | 0.4524 | 0.5193 | 15.7900 | 63.3000 | 17.2000 | +0.1297 | yolo11m base recipe - upper size point, KD teacher candidate |

### Week 1 per-class AP50

| id | W | WH | WHV | WV |
|---|---|---|---|---|
| w1_s_baseline_repro | 0.4618 | 0.4443 | 0.0754 | 0.9950 |
| w1_s_sgd_1e2 | 0.2704 | 0.3241 | 0.0646 | 0.8238 |
| w1_s_adamw_1e3 | 0.2709 | 0.3243 | 0.0193 | 0.4597 |
| w1_s_adamw_1e4 | 0.4227 | 0.4275 | 0.1301 | 0.9950 |
| w1_s_res_960 | 0.3801 | 0.4149 | 0.0956 | 0.9550 |
| w1_s_res_1280 | 0.3670 | 0.3700 | 0.1070 | 0.8529 |
| w1_s_noaug | 0.3257 | 0.2978 | 0.0985 | 0.8931 |
| w1_s_domainaug | 0.4499 | 0.4334 | 0.0887 | 0.8339 |
| w1_s_imbalance | 0.3751 | 0.3959 | 0.0451 | 0.9950 |
| w1_n_adamw_1e3 | 0.3727 | 0.3871 | 0.0923 | 0.8157 |
| w1_m_adamw_1e3 | 0.4177 | 0.4427 | 0.0903 | 0.8931 |

## Week 2

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w2_label_audit | 6 | pending | - | - | - | - | - | - | - | - | Model-assisted flag pass over test set -> 200-image human review queue + cleaned set |
| w2_eval_clean_vs_noisy | 6 | pending | - | - | - | - | - | - | - | - | Re-eval Week-1 winner on cleaned vs original test - the label-noise premium |
| w2_hard_negatives | 7 | pending | - | - | - | - | - | - | - | - | Auto-fetched distractor images as background-only, retrain winner - false-positive mode |
| w2_sahi_upperbound | 4 | pending | - | - | - | - | - | - | - | - | Sliced inference (sahi) on full-res test - distant small-object upper bound |

## Week 3

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w3_corruptions | 8 | pending | - | - | - | - | - | - | - | - | Motion blur / defocus / JPEG / low light / overexposure / dust-haze x 5 severities |

## Week 4

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w4_crossdata_sh17 | 9 | pending | - | - | - | - | - | - | - | - | Train on our set, test on SH17 after ontology mapping - expect a large drop |
| w4_crossdata_chv | 9 | pending | - | - | - | - | - | - | - | - | Cross-dataset test on CHV after ontology mapping |
| w4_track_vote | 10 | pending | - | - | - | - | - | - | - | - | ByteTrack + BoT-SORT, N-frame vote, sweep N in {1,3,5,7,9}, recall vs false-alarm |

## Week 5

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_kd_x_to_s | 11 | pending | - | - | - | - | - | - | - | - | Teacher (best m/x) -> yolo11s student, logit KD + neck feature KD |
| w5_kd_x_to_n | 11 | pending | - | - | - | - | - | - | - | - | Teacher -> yolo11n student, logit + feature KD |
| w5_calibration | 12 | pending | - | - | - | - | - | - | - | - | Reliability diagrams + temperature scaling on the winner; ECE before/after |

## Week 6

| id | item | status | mAP50 | mAP50-95 | P | R | lat ms | fps | train min | Δ mAP50-95 | desc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w6_p2_head | 13 | pending | - | - | - | - | - | - | - | - | P2 detection head for small PPE items, measure latency cost |
| w6_attention | 14 | pending | - | - | - | - | - | - | - | - | CBAM / ECA / SE / CoordAtt in the neck - ablation rows only |
| w6_two_stage | 15 | pending | - | - | - | - | - | - | - | - | person detector -> crop -> PPE classifier, worst-case latency vs worker count |
| w6_baselines | 16 | pending | - | - | - | - | - | - | - | - | RT-DETR / D-FINE / RTMDet baseline table |
| w6_qat | 17 | pending | - | - | - | - | - | - | - | - | QAT vs PTQ on the deploy student - recovery on thin classes |

## Figures

![w1_aug_ablation](figs/w1_aug_ablation.png)

![w1_lr_sweep](figs/w1_lr_sweep.png)

![w1_resolution](figs/w1_resolution.png)

![w1_size_latency](figs/w1_size_latency.png)
