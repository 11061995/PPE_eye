# experiments/ — the enhancement loop

Turns `../SCHEDULE.md` into runnable experiments, one result row + figures per
run. Everything runs in the `PPE/` venv as shipped (no extra installs for Week 1).

## Files

| file | role |
|---|---|
| `registry.yaml` | the full experiment matrix. `defaults` + per-experiment `params`, `deps`, `week`, `kind`. |
| `lib.py` | registry loading, result IO, dependency/next-experiment logic, latency bench. |
| `run.py` | run one experiment (`--next` for loop mode, or an id). Only `kind: train` implemented. |
| `report.py` | aggregate `results/*.json` → `results/REPORT.md`, `results/STATUS.md`, `results/figs/*.png`. |
| `results/` | `<id>.json` per experiment, `results.jsonl` append log, `REPORT.md`, `STATUS.md`, `figs/`. |

## Running it as a loop

```
/loop  PPE/Scripts/python.exe experiments/run.py --next
```

Each tick:
1. `run.py --next` finds the first experiment with no terminal result whose
   `deps` are all `done`.
2. It marks it `running`, trains it (foreground within the tick), evaluates on
   the **test** split, benchmarks single-image latency, writes
   `results/<id>.json` with `status: done` (or `failed` + traceback).
3. It regenerates the report.
4. Next tick picks the next one.

A `yolo11s` run at 640 is ~18–30 min on the 4060 Ti; 960/1280 and `yolo11m`
longer. Let ticks be self-paced.

### Review gates

When `--next` reaches an experiment whose `kind` has no handler yet
(`audit`, `eval`, `corrupt`, `crossdata`, `track`, `distill`, `calib`, `arch`)
it prints which week to implement and exits **3**. That is the deliberate stop
between weeks — implement that week's handler, then resume the loop.

## Manual use

```
PPE/Scripts/python.exe experiments/run.py w1_s_adamw_1e3   # one experiment
PPE/Scripts/python.exe experiments/run.py --status         # rebuild + print STATUS.md
PPE/Scripts/python.exe experiments/report.py               # rebuild report only
```

## Re-running an experiment

Delete its `results/<id>.json` (and its line in `results.jsonl`) and the loop
will pick it up again. Training run dirs are `runs/ppe_enh/<id>/` with
`exist_ok=True`, so they overwrite.

## Status values

`pending` → `running` → `done` | `failed`. `skipped` / `blocked` are reserved
for later handlers (e.g. cross-dataset when the target set can't be fetched).
Only `done` satisfies a dependency.
