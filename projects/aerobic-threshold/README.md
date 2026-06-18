# Aerobic Threshold

Finds the **metabolic lactate threshold** in a simplified skeletal-muscle FBA model by sweeping ATP maintenance demand and tracking when lactate export becomes sustained.

## Run

```bash
pip install -r requirements.txt
python lactate_threshold_sweep.py
```

## Output

- Console: estimated threshold step, RQ at threshold, peak lactate export
- Plot: `outputs/lactate_threshold_sweep.png` (created on run)

## Model (toy)

Host-only network: glycolysis → LDH → lactate export, with mitochondrial OXPHOS (RQ = 1.0 for aerobic glucose). No synthetic organelle — this is the **baseline physiology** project.
