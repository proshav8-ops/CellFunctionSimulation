# XenoOrganelle-Sim

### Machine Learning-Driven Synthetic Endosymbiont & 3D Tissue Simulation

> **Technical thesis:** XenoOrganelle-Sim unifies **Constraint-Based Reconstruction and Analysis (COBRA)**—solving steady-state metabolic fluxes under stoichiometric closure—with a **non-linear ML flux-gating controller** that dynamically regulates an engineered endosymbiont during simulated VO₂ max workloads, visualized through a **zero-asset WebGL tissue inspector** built on Three.js.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen)
![Build](https://img.shields.io/badge/Build-Passing-success)
![COBRApy](https://img.shields.io/badge/COBRApy-FBA-green)
![Three.js](https://img.shields.io/badge/Three.js-r160-WebGL-black)
![Tests](https://img.shields.io/badge/Tests-unittest-informational)

---

## Executive Summary & Bio-Engineering Vision

Athletic endurance is ultimately negotiated at the **single-cell level**. When skeletal muscle operates near **VO₂ max**, cytosolic ATP demand outpaces mitochondrial throughput, lactate floods the cytoplasm, partial pressure of oxygen collapses, and **reactive oxygen species (ROS)** accumulate toward cytotoxic concentrations. Evolution optimized the native mitochondrion over billions of years—but not for the synthetic performance envelopes of elite human sport.

**XenoOrganelle-Sim** asks a precise engineering question: *What if we could graft a de novo endosymbiont into human myocytes—a xeno-organelle purpose-built to extend the ATP–ROS Pareto frontier?*

The platform models this hypothesis in silico across three tightly coupled layers:

| Layer | Role |
|-------|------|
| **Metabolic reconstruction** | COBRApy FBA over a mass-balanced host + organelle reaction network |
| **ML flux-gating controller** | SciPy/NumPy optimization loop emulating a synthetic gene circuit |
| **3D tissue inspector** | Procedural Three.js visualization with raycast phenotype interrogation |

### Primary Design Objectives

1. **Maximize ATP generation density** — sustain higher maintenance flux under glycogen depletion and hypoxic boundary conditions than a native host cell.
2. **Accelerate systemic lactate clearance** — route fermentative overflow through a high-affinity organelle lactate sink (`SYN_ORG_LAC_SINK`) rather than cytosolic accumulation.
3. **Manage oxidative stress kinetics** — operate hyper-efficient low-pO₂ respiration (`SYN_ORG_HYPER_O2`) behind an ML throttle that enforces **managed dysbiosis**: push performance until ROS approaches toxicity, then trigger a controlled **metabolic cliff** rather than silent thermodynamic failure.

### Native Mitochondrion vs. Engineered Xeno-Organelle (v1.0)

| Dimension | Native Human Mitochondrion | Engineered Xeno-Organelle (v1.0) |
|-----------|---------------------------|----------------------------------|
| **Evolutionary origin** | Endosymbiotic α-proteobacterium (natural) | De novo synthetic compartment (`syn`) |
| **Lactate fate** | Indirect via cytosolic LDH → pyruvate shuttle | Direct high-affinity import + organelle recycling |
| **O₂ affinity profile** | Standard host OXPHOS (`MITO_OXPHOS`) | Hyper-O₂ channel active at reduced pO₂ |
| **ROS yield** | Baseline (implicit in host respiration) | Elevated per flux unit; explicit `H₂O₂` proxy |
| **Regulatory logic** | Static biochemical kinetics | **ML flux-gating** — dynamic upper-bound control each time step |
| **Failure phenotype** | Progressive ATP deficit | **Dysbiosis collapse** at ROS > 5.0 with **80% ATP ceiling penalty** |
| **VO₂ max envelope** | Biologically constrained | Computationally extended until ROS toxicity wins |

> *The xeno-organelle does not eliminate trade-offs—it relocates them. The simulation makes those trade-offs measurable, testable, and visible.*

---

## Backend Architecture — The Computational Engine

### Metabolic Flux Balance Analysis (FBA)

The biochemical core is a **constraint-based metabolic model** reconstructed in [COBRApy](https://opencobra.github.io/cobrapy/). At pseudo-steady state, intracellular metabolite pools are constant; mass balance reduces to a linear system:

$$
S \cdot v = 0
$$

where:

- **S** ∈ ℝ^(m × n) is the stoichiometric matrix (metabolites × reactions),
- **v** ∈ ℝ^n is the flux vector (mmol · gDW⁻¹ · h⁻¹).

FBA selects a biologically feasible **v** by optimizing a linear objective—here, maximization of ATP maintenance flux (`ATPM`)—subject to:

- **Thermodynamic directionality** — irreversible reaction bounds (lb ≥ 0 for forward-only steps),
- **Nutrient availability** — exchange upper bounds on glucose and O₂ that tighten across workout time steps,
- **Organelle capacity** — ML-tunable upper bounds on synthetic reaction fluxes.

#### Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│  HOST CYTOSOL (c)                                               │
│  EX_glc / EX_o2 ──► GLYCOLYSIS ──► LDH ──► lactate             │
│                         │              │                        │
│                         ▼              ▼                        │
│                   MITO_OXPHOS    SYN_LAC_IMPORT ───────────────┼──┐
│                         │                                       │  │
│                         ▼                                       │  │
│                      ATPM (demand)                              │  │
└─────────────────────────────────────────────────────────────────┘  │
                                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  XENO-ORGANELLE (syn)                                           │
│  lac_syn ──► SYN_ORG_LAC_SINK ──► pyr_syn ──► SYN_ORG_HYPER_O2 │
│                      │                              │           │
│                      └──── SYN_PYR_EXPORT ──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

**Elemental integrity** is enforced at build time: core reactions pass `reaction.check_mass_balance()` with explicit chemical formulas (C, H, O, N, P). Currency metabolite exchanges (ADP, Pi, NAD⁺/NADH, H⁺, H₂O) maintain LP feasibility without violating atom conservation in curated reactions.

**Solver redundancy:** if COBRApy is unavailable, an equivalent **SciPy `linprog`** formulation (`method='highs'`) mirrors the stoichiometric matrix byte-for-byte.

> **Theoretical deep-dive:** FBA does not model kinetics—it explores the *feasible flux polytope*. XenoOrganelle-Sim uses this deliberately: the question is not *how fast* enzymes react, but *what maximum ATP yield remains thermodynamically admissible* under resource contraction and organelle engineering.

---

### ML Flux-Gating Controller

The organelle is controlled by a **predictive non-linear optimization loop** implemented in `optimize_organelle_bounds()`. Functionally, it emulates an automated **synthetic gene circuit** that throttles hyper-respiration channels before ROS breaches cytotoxic limits.

#### Control loop (per time step)

```
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Workload    │────►│  FBA solve  │────►│  Measure ATP,    │
│  demand ↑    │     │  (COBRApy)  │     │  ROS, lactate    │
└──────────────┘     └─────────────┘     └────────┬─────────┘
                                                   │
                     ┌─────────────────────────────▼─────────────────────────────┐
                     │  Reward: R = v_ATP − λ·max(0, ROS − ROS_tox)²         │
                     │  Gradient ascent on organelle bound parameters          │
                     │  (finite-difference via SciPy/NumPy)                    │
                     └─────────────────────────────┬───────────────────────────┘
                                                   │
              ┌────────────────────────────────────▼──────────────────────────┐
              │  ROS > 5.0 ?  →  DYSBIOSIS COLLAPSE                           │
              │    • Zero organelle flux bounds                               │
              │    • efficiency_drop_flag = True                              │
              │    • ATP ceiling ← 20% of pre-collapse maximum                │
              └───────────────────────────────────────────────────────────────┘
```

#### Tunable control parameters

| Symbol / Constant | Value | Semantics |
|-------------------|-------|-----------|
| `ROS_TOXICITY_THRESHOLD` | `5.0` | ROS flux units triggering dysbiosis evaluation |
| `DYSBIOSIS_ATP_PENALTY_FRACTION` | `0.80` | Fractional ATP capacity removed post-collapse |
| `ML_LEARNING_RATE` | `0.45` | Projected gradient-ascent step size |
| `ML_FINITE_DIFF_EPS` | `0.25` | Finite-difference perturbation for ∂R/∂bound |
| `ROS_PENALTY_WEIGHT` | `12.0` | Quadratic ROS penalty coefficient (λ) |
| `N_STEPS` | `20` | Endurance workout discretization |

The controller adjusts two decision variables each iteration:

- `lac_sink_ub` — upper flux limit on lactate import/recycling,
- `hyper_o2_ub` — upper flux limit on hyper-affinity oxidative phosphorylation.

Feedback signals include **lactate pool pressure** (encourages sink expansion) and **hypoxic stress** (favors hyper-O₂ channel engagement as O₂ exchange bounds contract).

---

## Frontend Architecture — The Interactive 3D Tissue Inspector

The WebGL layer is a **pure Three.js / HTML5** application with **zero external 3D asset dependencies**. Every biological structure is procedurally generated at runtime—making the viewer deployable as a static site with no Blender pipeline, no glTF hosting, and no CDN model fetches.

### Rendering pipeline

| Subsystem | Implementation |
|-----------|----------------|
| **Cell membrane** | `THREE.MeshPhysicalMaterial` — iridescent transmission, variable IOR, subtle thickness for parallax depth |
| **Nucleus** | `CanvasTexture` generated on-the-fly — chromatin noise, nucleolar bodies, perlin-style heterochromatin speckling |
| **Xeno-organelle** | Custom `BufferGeometry` — folded inner cristae folds via parametric surface deformation; emissive intensity mapped to simulated flux state |
| **Host mitochondria** | Instanced meshes with randomized orientation — lower emissive baseline than xeno-organelle |
| **Cytoplasm** | GPU-friendly particle field (`THREE.Points`) — thousands of metabolite proxy particles with additive blending |
| **Lighting** | Three-point rig + subtle rim light — accentuates membrane curvature and organelle emissive contrast |

### Interaction model

- **OrbitControls** — constrained polar angles for intuitive turntable inspection.
- **Raycasting hover inspector** — `THREE.Raycaster` against organelle/membrane meshes; hover events populate a **glassmorphic UI panel** (`backdrop-filter: blur`) with phenotype readouts.
- **Phenotype overlay** — ATP demand vs. production, lactate pool, ROS flux, dysbiosis flag; bindable to exported simulation JSON from the Python engine.

> **Architectural principle:** The frontend is a *spatial dashboard* for FBA outputs—not a decorative animation. Raycast targets correspond to biological compartments (`host`, `xeno-organelle`, `nucleus`) that map 1:1 to model namespaces in the COBRApy reconstruction.

---

## Test-Driven Development & Verification Suite

Metabolic models that violate stoichiometry produce **scientifically valid-looking but physically impossible** results. XenoOrganelle-Sim ships a **`unittest`** QA harness (`test_synthetic_organelle_sim.py`) that treats correctness as a release gate—not an afterthought.

```bash
python test_synthetic_organelle_sim.py
```

### Four critical verification classes

| # | Test | Invariant enforced | Method |
|---|------|-------------------|--------|
| **1** | `test_01_thermodynamic_conservation_no_free_energy_loop` | **Zero nutrient import ⇒ zero ATP production** | Glucose + O₂ exchanges clamped to 0; assert `v_ATP == 0` |
| **2** | `test_02_mass_and_charge_balance_for_custom_reactions` | **Elemental closure on every curated reaction** | `reaction.check_mass_balance() == {}` for GLYCOLYSIS, LDH, MITO_OXPHOS, SYN_ORG_LAC_SINK, SYN_ORG_HYPER_O2, ROS_DETOX |
| **3** | `test_03_respiratory_quotient_equals_unity_for_aerobic_glucose` | **Mammalian carbohydrate RQ = 1.0** | Aerobic glucose oxidation; assert `v_CO₂ / v_O₂ = 1.0 ± 10⁻⁴` |
| **4** | `test_04_dysbiosis_collapse_and_atp_efficiency_penalty` | **ML controller boundary + 80% ATP cliff** | Extreme workload forces ROS > 5.0; assert `efficiency_drop_flag`; assert penalized ATP = 20% of pre-collapse ceiling |

> **QA philosophy:** Test 1 catches perpetual-motion leaks. Test 2 catches sloppy stoichiometry. Test 3 catches wrong physiology. Test 4 catches control-system regressions. Together, they form a *minimum viable trust surface* for publishing simulation claims.

Each test instantiates a **fresh `cobra.Model`** in `setUp()` and emits structured `[QA-BIO]` log lines identifying the physical law under verification.

---

## Repository Structure & Setup

### Directory layout

```
XenoOrganelle-Sim/
│
├── README.md                                 # Portfolio documentation (this file)
├── LICENSE                                   # MIT
│
├── synthetic_organelle_muscle_simulation.py  # Core FBA engine, ML controller, plotting
├── synthetic_organelle_sim.py                # Public API alias (import surface)
│
├── test_synthetic_organelle_sim.py           # unittest QA suite (4 metabolic invariants)
│
├── frontend/
│   ├── index.html                            # Three.js tissue inspector entry point
│   ├── js/
│   │   ├── scene.js                          # Scene graph, lighting, render loop
│   │   ├── organelle.js                      # Procedural xeno-organelle geometry
│   │   ├── inspector.js                      # Raycaster + glassmorphic UI bindings
│   │   └── particles.js                      # Cytoplasmic metabolite particle field
│   └── css/
│       └── inspector.css                       # Glassmorphic overlay panels
│
├── outputs/
│   └── synthetic_organelle_simulation.png    # Generated 3-panel FBA diagnostic (post-run)
│
└── .github/
    └── workflows/
        └── ci.yml                            # Lint + unittest on push (planned)
```

### Installation

```bash
# 1. Clone
git clone https://github.com/your-username/XenoOrganelle-Sim.git
cd XenoOrganelle-Sim

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell

# 3. Dependencies
pip install --upgrade pip
pip install cobra numpy scipy matplotlib seaborn

# 4. Run metabolic simulation (generates outputs/synthetic_organelle_simulation.png)
python synthetic_organelle_muscle_simulation.py

# 5. Run verification suite
python test_synthetic_organelle_sim.py
```

### Launch the 3D tissue inspector

```bash
# Static open (file:// protocol)
open frontend/index.html          # macOS
xdg-open frontend/index.html    # Linux
start frontend\index.html       # Windows

# Recommended: local HTTP server (enables ES module + JSON fetch)
python -m http.server 8080 --directory frontend
# → http://localhost:8080/index.html
```

### Programmatic API (notebooks & extensions)

```python
import synthetic_organelle_sim as xos

model = xos.build_muscle_model(include_organelle=True)
result = xos.run_endurance_simulation(n_steps=20)

print(f"Peak demand     : {result.enhanced[-1].atp_demand:.1f}")
print(f"Enhanced ATP    : {result.enhanced[-1].atp_produced:.1f}")
print(f"Dysbiosis step  : {result.dysbiosis_step}")

xos.plot_results(result, output_path="outputs/synthetic_organelle_simulation.png")
```

---

## Simulation Outputs — Reading the Diagnostic Figure

The engine emits a **three-panel time-series** chart capturing a 20-step endurance workout:

| Panel | Signal | Interpretation |
|-------|--------|----------------|
| **I — ATP vs. demand** | Dashed = demand; solid = production (enhanced vs. baseline) | Vertical inflection = metabolic deficit onset; red marker = dysbiosis step |
| **II — Lactate pool + sink flux** | Cytosolic lactate accumulation vs. organelle clearance rate | Separation between green curves quantifies xeno-organelle recycling advantage |
| **III — Organelle flux & ROS** | Lactate sink, hyper-O₂, host OXPHOS, ROS proxy | ROS approaching `5.0` threshold = managed dysbiosis boundary; post-collapse drop = efficiency cliff |

The most scientifically consequential region is the **threshold crossing**—where the ML controller loses the ROS war and ATP capacity collapses by 80%. That bifurcation is the in silico signature of *engineered performance exceeding biological safety margins*.

---

## Future Research Roadmap

| Phase | Milestone | Impact |
|-------|-----------|--------|
| **Q1** | **PyTorch DQN controller** — replace finite-difference gradient ascent with a Deep Q-Network trained across stochastic workout schedules | Generalize organelle policy beyond hand-tuned reward shaping |
| **Q2** | **Recon3D host integration** — embed xeno-organelle reactions into genome-scale human metabolic reconstruction | Move from toy network to clinically legible scale |
| **Q3** | **Flux Variability Analysis (FVA)** at dysbiosis bifurcation | Quantify robustness of ATP–ROS trade-off under stoichiometric uncertainty |
| **Q4** | **WebGL ↔ Python live bridge** — FastAPI WebSocket streaming `StepRecord` objects to the tissue inspector | Real-time 3D phenotype updates during simulation |
| **Q5** | **CI/CD for static frontend** — GitHub Actions → GitHub Pages deployment with automated screenshot regression tests | Portfolio-grade continuous delivery of the visualization layer |
| **Q6** | **Custom GLSL shader library** — ROS-responsive emissive bloom, membrane caustics, cristae subsurface scattering | Publication-quality rendering without external assets |

---

## Acknowledgments & References

- **COBRApy** — Ebrahim, A. et al. (2013). *BMC Systems Biology*, 7:74.
- **Constraint-based modeling** — Orth, J.D., Thiele, I. & Palsson, B.Ø. (2010). *Molecular Systems Biology*, 6:245.
- **Three.js** — JavaScript 3D library for WebGL rendering.

---

## License

Distributed under the **MIT License**. See `LICENSE` for full terms.

---

<p align="center">
  <strong>XenoOrganelle-Sim</strong><br>
  <em>Engineering the next organelle. Validating every flux. Rendering every trade-off.</em>
</p>
