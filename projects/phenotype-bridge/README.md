# Phenotype Bridge

Maps **metabolic flux summaries** to **observable endurance phenotypes** — VO₂ proxy, lactate handling, fatigue resistance, and oxidative stress risk.

This project does not run FBA itself; it consumes flux dictionaries produced by sibling simulations (`xeno-organelle`, `aerobic-threshold`).

## Run

```bash
pip install -r requirements.txt
python phenotype_mapper.py
```

## Output

- Console: phenotype scores for example native vs. xeno-organelle flux profiles
- Plot: `outputs/phenotype_radar.png`
