# Shipping a PPE-EYE checkpoint into Amsar

What to run, what to check, and what to tell the site. Numbers live in
`experiments/results/GAINS.md`; this file is the procedure.

## The contract

Amsar's single-pass engine loads `ppe_models.PPE_EYE_MODEL_PATH` — `ppe_eye.pt`,
sitting beside `ppe_models.py`. Nothing else needs to change: `ppe_eye.py`,
`ZoneAnalytics`, `PPESmoother`, the per-zone relay debounce and both zone kinds
all consume the same `(person_boxes, ppe_results)` pair the cascade produced, so
swapping weights is the whole integration.

Two guards already stand between a bad file and a live site:

1. `ppe_eye.verdict_map()` returns `{}` for anything whose labels are not the
   W/WH/WHV/WV ontology, and `load_ppe_eye()` refuses to run on an empty map.
   This exists because `ppe.pt` loads perfectly well as a YOLO model and would
   silently clear every worker on site — a safety system failing *silent* is the
   worst case.
2. `cameras.PPE_ENGINE` is the rollback switch. Setting `ppe_engine: cascade` in
   config puts the old two-model path back without a code change.

## Procedure

```bash
cd research/YOLO-PPEeye

# 1. measure it on all four beds and get a pass/fail
PPE/Scripts/python.exe experiments/gate.py \
    --weights runs/ppe_enh/<run>/weights/best.pt --tag <run>

# 2. dry-run the install: checks the ontology and the gate record, copies nothing
PPE/Scripts/python.exe experiments/deploy.py \
    --weights runs/ppe_enh/<run>/weights/best.pt --dry-run

# 3. install (backs up the previous ppe_eye.pt and writes a provenance file)
PPE/Scripts/python.exe experiments/deploy.py \
    --weights runs/ppe_enh/<run>/weights/best.pt

# 4. the engine's own self-check - no GPU, no weights, no camera needed
cd ../Amsar-AI--main/Amsar-AI--main && python test_ppe_eye.py
```

`deploy.py` refuses a checkpoint that has no gate record or a failing one. If you
override with `--force`, write down in the provenance note why that was right.

## After installing

- **Rebuild or delete `ppe_eye.engine`.** `ppe_models.USE_TENSORRT` prefers a
  prebuilt TensorRT engine sitting next to the `.pt`, so a stale engine silently
  keeps the old weights live. `python3 export_trt.py` rebuilds it.
- **Re-check the threshold.** `cameras.PPE_EYE_CONF` is 0.10. A checkpoint with a
  lower false-alarm rate can be run *lower* for the same nuisance budget, which
  is where its extra violation recall comes from (`WEEK5_FINDINGS.md`). The gate
  sweeps below 0.10 and its `indomain_row` names the operating point the
  reported numbers came from.
- **Keep `PPESmoother` on.** Temporal voting is orthogonal to the detector and
  untested here (this repo has stills only); `w4_track_vote` measures an upper
  bound on what it can buy.

## What to tell the site

- **Not for night operation.** At low-light severity 4 every checkpoint measured
  here misses 987–994 of 999 persons. This is unresolved: Week 3b tried
  brightness augmentation and falsified it. Needs augmentation that models
  darkening *and* sensor noise together, or real night/IR frames.
- **Lens focus is the highest-leverage physical variable.** Photometric
  corruptions (haze, overexposure) cost 5–6% of violation recall at severity 3;
  optical ones (defocus, motion blur) cost 40–70%. Check focus per camera before
  blaming the model.
- **Vest compliance is weaker than helmet compliance.** Every number in this repo
  uses the `helmet` rule. SH17 has no bare-torso class, so "no vest annotated"
  and "no vest worn" are still the same signal — the `strict` rule is not
  supported by the data.

## Rolling back

```yaml
# config
ppe_engine: cascade
```

or restore the backup `deploy.py` left beside the weights:
`ppe_eye.prev-<timestamp>.pt` → `ppe_eye.pt`.
