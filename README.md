# CellFunctionSimulation

Exploring the power of human cell biochemistry, its physical function, and how structure leads to phenotype.

This repository is a **monorepo of cell-level simulation projects**. Each project lives under `projects/` with its own README, dependencies, and tests.

## Projects

| Project | Question | Stack |
|---------|----------|-------|
| [**xeno-organelle**](projects/xeno-organelle/) | Can a synthetic endosymbiont extend the ATP–ROS Pareto frontier under VO₂ max load? | COBRApy, NumPy, SciPy, Matplotlib |
| [**aerobic-threshold**](projects/aerobic-threshold/) | At what work rate does lactate export exceed clearance — the metabolic lactate threshold? | COBRApy, NumPy, Matplotlib |
| [**phenotype-bridge**](projects/phenotype-bridge/) | How do flux vectors translate into tissue-level endurance phenotypes? | NumPy, Matplotlib |

## Quick start

```bash
git clone https://github.com/proshav8-ops/CellFunctionSimulation.git
cd CellFunctionSimulation

# Project 1 — synthetic organelle + ML controller
cd projects/xeno-organelle
pip install -r requirements.txt
python test_synthetic_organelle_sim.py
python synthetic_organelle_muscle_simulation.py

# Project 2 — lactate threshold sweep
cd ../aerobic-threshold
pip install -r requirements.txt
python lactate_threshold_sweep.py

# Project 3 — flux → phenotype mapping
cd ../phenotype-bridge
pip install -r requirements.txt
python phenotype_mapper.py
```

## Philosophy

**Structure leads to function.** Each simulation starts from stoichiometric closure (mass-balanced reactions), adds physiological constraints (exchange bounds, demand), and reads out phenotypes only after the math is consistent.

## License

MIT — see [LICENSE](LICENSE).
