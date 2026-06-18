#!/usr/bin/env python3
"""
Synthetic Organelle Skeletal Muscle Metabolic Simulation
=======================================================

A toy flux-balance model of a human skeletal muscle cell hosting a
machine-learning-optimized synthetic organelle designed to improve VO2 max
and endurance.  The organelle provides (1) a high-affinity lactate sink and
(2) hyper-efficient low-pO2 oxidative phosphorylation at the cost of ROS.

The simulation runs a 20-step endurance workout with linearly increasing
ATP demand.  At each step a gradient-descent optimizer tunes organelle
reaction bounds to maximize ATP yield while keeping ROS below a toxicity
threshold.  Exceeding the threshold triggers dysbiosis collapse.

Dependencies: cobra, numpy, matplotlib (optional: seaborn for styling).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# COBRApy import with graceful fallback to a SciPy LP solver
# ---------------------------------------------------------------------------
try:
    from cobra import Metabolite, Model, Reaction

    COBRA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    COBRA_AVAILABLE = False
    Model = Metabolite = Reaction = None  # type: ignore[misc, assignment]

try:
    from scipy.optimize import linprog

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
N_STEPS: int = 20
ROS_TOXICITY_THRESHOLD: float = 5.0
ROS_COLLAPSE_FACTOR: float = 1.0
DYSBIOSIS_ATP_PENALTY_FRACTION: float = 0.80
ML_LEARNING_RATE: float = 0.45
ML_FINITE_DIFF_EPS: float = 0.25
ROS_PENALTY_WEIGHT: float = 12.0
DYSBIOSIS_PENALTY: float = -1_000.0

# Organelle reaction identifiers
SYN_LAC_SINK_ID = "SYN_ORG_LAC_SINK"
SYN_HYPER_O2_ID = "SYN_ORG_HYPER_O2"
ATP_DEMAND_ID = "ATPM"
LDH_ID = "LDH"
GLYCOLYSIS_ID = "GLYCOLYSIS"
MITO_OXPHOS_ID = "MITO_OXPHOS"
EX_GLC_ID = "EX_glc__D_e"
EX_O2_ID = "EX_o2_e"
EX_CO2_ID = "EX_co2_e"
EX_ROS_ID = "EX_ros_e"
ROS_DETOX_ID = "ROS_DETOX"

# Reactions validated for elemental mass balance in the QA suite
MASS_BALANCE_REACTION_IDS: Tuple[str, ...] = (
    GLYCOLYSIS_ID,
    LDH_ID,
    MITO_OXPHOS_ID,
    SYN_LAC_SINK_ID,
    SYN_HYPER_O2_ID,
    ROS_DETOX_ID,
)


@dataclass
class OrganelleBounds:
    """Adjustable upper flux limits for synthetic organelle reactions (mmol/gDW/h)."""

    lac_sink_ub: float = 8.0
    hyper_o2_ub: float = 6.0
    lac_sink_max: float = 40.0
    hyper_o2_max: float = 25.0
    pre_collapse_atp_ceiling: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {"lac_sink_ub": self.lac_sink_ub, "hyper_o2_ub": self.hyper_o2_ub}

    def clip(self) -> None:
        self.lac_sink_ub = float(np.clip(self.lac_sink_ub, 0.0, self.lac_sink_max))
        self.hyper_o2_ub = float(np.clip(self.hyper_o2_ub, 0.0, self.hyper_o2_max))


@dataclass
class StepRecord:
    """Snapshot of metabolic state at one simulation time-step."""

    step: int
    atp_demand: float
    atp_produced: float
    lactate_pool: float
    ros_flux: float
    syn_lac_flux: float
    syn_o2_flux: float
    oxphos_flux: float
    collapsed: bool
    organelle_bounds: OrganelleBounds = field(default_factory=OrganelleBounds)


@dataclass
class SimulationResult:
    """Container for baseline and enhanced cell time-series."""

    enhanced: List[StepRecord]
    baseline: List[StepRecord]
    dysbiosis_step: Optional[int]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _element_imbalance(stoichiometry: Dict["Metabolite", float]) -> Dict[str, float]:
    """Return unbalanced elemental coefficients for a stoichiometric dictionary."""
    rxn = Reaction("_tmp_balance_probe")
    rxn.add_metabolites(stoichiometry)
    return rxn.check_mass_balance()


def autobalance_stoichiometry(
    stoichiometry: Dict["Metabolite", float],
    adjustment_metabolites: Tuple["Metabolite", ...],
    max_iterations: int = 40,
) -> Dict["Metabolite", float]:
    """
    Close elemental imbalances by adding water, protons, and phosphate.

    The algorithm greedily applies adjustment metabolites until COBRApy's
    ``check_mass_balance`` returns an empty dictionary.
    """
    balanced = dict(stoichiometry)
    for _ in range(max_iterations):
        imbalance = _element_imbalance(balanced)
        if not imbalance:
            return balanced
        for met in adjustment_metabolites:
            for element, residual in list(imbalance.items()):
                if element == "charge":
                    continue
                met_element = met.elements.get(element, 0)
                if met_element == 0:
                    continue
                delta = -residual / met_element
                balanced[met] = balanced.get(met, 0.0) + delta
                break
            imbalance = _element_imbalance(balanced)
            if not imbalance:
                return balanced
    remaining = _element_imbalance(balanced)
    if remaining:
        raise ValueError(f"Unable to auto-balance reaction; residual={remaining}")
    return balanced


def build_muscle_model(include_organelle: bool = True) -> "Model":
    """
    Build a simplified human skeletal muscle metabolic network.

    Pathways
    --------
    * Glycolysis (single lumped reaction)
    * Lactate fermentation (LDH)
    * Mitochondrial oxidative phosphorylation
    * Optional synthetic organelle reactions in a logical ``syn`` compartment

    Parameters
    ----------
    include_organelle:
        If ``False``, organelle reactions are present but clamped to zero flux.

    Returns
    -------
    cobra.Model
        Configured metabolic model ready for FBA.
    """
    if not COBRA_AVAILABLE:
        raise RuntimeError("COBRApy is required for build_muscle_model().")

    model = Model("human_skeletal_muscle")
    model.name = "Human Skeletal Muscle (Toy)"

    def met(met_id: str, name: str, formula: str, compartment: str = "c") -> Metabolite:
        return Metabolite(met_id, name=name, formula=formula, compartment=compartment)

    metabolites = [
        met("glc__D_c", "D-Glucose", "C6H12O6"),
        met("pyr_c", "Pyruvate", "C3H4O3"),
        met("lac__L_c", "L-Lactate", "C3H6O3"),
        met("o2_c", "Oxygen", "O2"),
        met("atp_c", "ATP", "C10H16N5O13P3"),
        met("adp_c", "ADP", "C10H15N5O10P"),
        met("pi_c", "Phosphate", "O3P"),
        met("co2_c", "CO2", "CO2"),
        met("h2o_c", "Water", "H2O"),
        met("h_c", "Proton", "H"),
        met("nad_c", "NAD+", "C21H27N7O14P2"),
        met("nadh_c", "NADH", "C21H29N7O14P2"),
        met("ros_c", "Reactive Oxygen Species", "H2O2"),
        met("lac__L_syn", "L-Lactate (organelle)", "C3H6O3", "syn"),
        met("pyr_syn", "Pyruvate (organelle)", "C3H4O3", "syn"),
        met("nad_syn", "NAD+ (organelle)", "C21H27N7O14P2", "syn"),
        met("nadh_syn", "NADH (organelle)", "C21H29N7O14P2", "syn"),
    ]
    model.add_metabolites(metabolites)
    m = model.metabolites

    def add_rxn(rxn_id: str, name: str, stoich: Dict[Metabolite, float], lb: float = 0.0, ub: float = 1_000.0) -> Reaction:
        rxn = Reaction(rxn_id, name=name, lower_bound=lb, upper_bound=ub)
        rxn.add_metabolites(stoich)
        model.add_reactions([rxn])
        return rxn

    # Exchanges / boundary conditions (positive flux = import/uptake for nutrients)
    add_rxn("EX_glc__D_e", "Glucose uptake", {m.glc__D_c: 1.0}, lb=0.0, ub=30.0)
    add_rxn("EX_o2_e", "Oxygen uptake", {m.o2_c: 1.0}, lb=0.0, ub=25.0)
    add_rxn("EX_co2_e", "CO2 efflux", {m.co2_c: -1.0}, lb=0.0, ub=1_000.0)
    add_rxn("EX_lac__L_e", "Lactate efflux", {m.lac__L_c: -1.0}, lb=0.0, ub=80.0)
    add_rxn(EX_ROS_ID, "ROS efflux", {m.ros_c: -1.0}, lb=0.0, ub=200.0)
    # Currency-metabolite exchanges keep the mass-balanced core network feasible.
    add_rxn("EX_adp_e", "ADP pool exchange", {m.adp_c: -1.0}, lb=-1_000.0, ub=1_000.0)
    add_rxn("EX_pi_e", "Phosphate pool exchange", {m.pi_c: -1.0}, lb=-1_000.0, ub=1_000.0)
    add_rxn("EX_h_e", "Proton pool exchange", {m.h_c: -1.0}, lb=-1_000.0, ub=1_000.0)
    add_rxn("EX_h2o_e", "Water pool exchange", {m.h2o_c: -1.0}, lb=-1_000.0, ub=1_000.0)
    add_rxn("EX_nad_e", "NAD pool exchange", {m.nad_c: -1.0}, lb=-1_000.0, ub=1_000.0)
    add_rxn("EX_nadh_e", "NADH pool exchange", {m.nadh_c: -1.0}, lb=-1_000.0, ub=1_000.0)

    balance_pool = (m.h_c, m.h2o_c, m.pi_c)

    # Host core metabolism (auto-balanced lumped reactions)
    add_rxn(
        GLYCOLYSIS_ID,
        "Lumped glycolysis",
        autobalance_stoichiometry(
            {
                m.glc__D_c: -1.0,
                m.adp_c: -2.0,
                m.pi_c: -2.0,
                m.pyr_c: 2.0,
                m.atp_c: 2.0,
                m.h2o_c: 2.0,
            },
            balance_pool,
        ),
        lb=0.0,
        ub=50.0,
    )
    add_rxn(
        LDH_ID,
        "Lactate dehydrogenase (fermentation)",
        {
            m.pyr_c: -1.0,
            m.nadh_c: -1.0,
            m.lac__L_c: 1.0,
            m.nad_c: 1.0,
        },
        lb=0.0,
        ub=60.0,
    )
    add_rxn(
        MITO_OXPHOS_ID,
        "Mitochondrial oxidative phosphorylation",
        autobalance_stoichiometry(
            {
                m.pyr_c: -1.0,
                m.o2_c: -3.0,
                m.adp_c: -4.0,
                m.pi_c: -4.0,
                m.co2_c: 3.0,
                m.atp_c: 4.0,
                m.h2o_c: 2.0,
            },
            balance_pool,
        ),
        lb=0.0,
        ub=40.0,
    )
    add_rxn(
        ATP_DEMAND_ID,
        "ATP maintenance demand",
        {m.atp_c: -1.0},
        lb=0.0,
        ub=200.0,
    )
    add_rxn(
        ROS_DETOX_ID,
        "Catalase-style ROS detoxification",
        {m.ros_c: -2.0, m.h2o_c: 2.0, m.o2_c: 1.0},
        lb=0.0,
        ub=200.0,
    )

    # Transport into synthetic organelle
    add_rxn(
        "SYN_LAC_IMPORT",
        "Organelle lactate import (high affinity)",
        {m.lac__L_c: -1.0, m.lac__L_syn: 1.0},
        lb=0.0,
        ub=40.0 if include_organelle else 0.0,
    )

    # Organelle reactions
    add_rxn(
        SYN_LAC_SINK_ID,
        "Organelle lactate-to-pyruvate recycling (ML-tuned)",
        {
            m.lac__L_syn: -1.0,
            m.nad_c: -1.0,
            m.pyr_syn: 1.0,
            m.nadh_c: 1.0,
        },
        lb=0.0,
        ub=40.0 if include_organelle else 0.0,
    )
    add_rxn(
        "SYN_PYR_EXPORT",
        "Recycled pyruvate export to cytosol",
        {m.pyr_syn: -1.0, m.pyr_c: 1.0},
        lb=0.0,
        ub=40.0 if include_organelle else 0.0,
    )
    add_rxn(
        SYN_HYPER_O2_ID,
        "Hyper-affinity O2 oxidative phosphorylation (ML-tuned)",
        autobalance_stoichiometry(
            {
                m.pyr_syn: -1.0,
                m.o2_c: -3.0,
                m.adp_c: -5.0,
                m.pi_c: -5.0,
                m.co2_c: 3.0,
                m.atp_c: 5.0,
                m.h2o_c: 2.0,
                m.ros_c: 1.0,
            },
            balance_pool,
        ),
        lb=0.0,
        ub=25.0 if include_organelle else 0.0,
    )

    model.objective = ATP_DEMAND_ID
    model.solver.configuration.tolerance = 1e-9
    return model


# ---------------------------------------------------------------------------
# Fallback LP-FBA (mirrors the COBRApy model stoichiometry)
# ---------------------------------------------------------------------------
_FALLBACK_RXN_IDS: List[str] = [
    "EX_glc__D_e",
    "EX_o2_e",
    "EX_co2_e",
    "EX_lac__L_e",
    "EX_ros_e",
    "GLYCOLYSIS",
    "LDH",
    "MITO_OXPHOS",
    "ATPM",
    "SYN_LAC_IMPORT",
    SYN_LAC_SINK_ID,
    "SYN_PYR_EXPORT",
    SYN_HYPER_O2_ID,
]

_FALLBACK_MET_IDS: List[str] = [
    "glc__D_c",
    "pyr_c",
    "lac__L_c",
    "o2_c",
    "atp_c",
    "co2_c",
    "ros_c",
    "lac__L_syn",
    "pyr_syn",
]

# rows = metabolites, cols = reactions (S @ v = 0)
_FALLBACK_S: np.ndarray = np.array(
    [
        # glc pyr lac  o2  atp co2 ros lac_syn pyr_syn
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EX_glc (uptake)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EX_o2 (uptake)
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # EX_co2
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EX_lac
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # EX_ros
        [-1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # GLYC
        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LDH
        [0, -1, 0, -2, 5, 3, 0.12, 0, 0, 0, 0, 0, 0],  # MITO
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # ATPM
        [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # SYN_LAC_IMPORT
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],  # SYN_LAC_SINK
        [0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # SYN_PYR_EXPORT
        [0, 0, 0, -1.5, 6, 2, 0.45, 0, -1, 0, 0, 0, 0],  # SYN_HYPER_O2
    ],
    dtype=float,
).T


def _fallback_fba(
    atp_demand: float,
    bounds: OrganelleBounds,
    include_organelle: bool,
    o2_ub: float,
    glc_ub: float,
    fatigue_factor: float = 1.0,
    efficiency_drop_active: bool = False,
) -> Dict[str, float]:
    """
    Solve a steady-state FBA problem via ``scipy.optimize.linprog``.

    Maximizes ATP demand flux subject to stoichiometric balance.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("Neither COBRApy nor SciPy is available for FBA.")

    n_rxn = len(_FALLBACK_RXN_IDS)
    lbs = np.zeros(n_rxn)
    ubs = np.full(n_rxn, 1_000.0)

    # Per-reaction bounds
    lbs[_FALLBACK_RXN_IDS.index("ATPM")] = 0.0
    ubs[_FALLBACK_RXN_IDS.index("ATPM")] = atp_demand
    ubs[_FALLBACK_RXN_IDS.index("EX_glc__D_e")] = glc_ub
    ubs[_FALLBACK_RXN_IDS.index("EX_o2_e")] = o2_ub
    capacity_multiplier = 1.0
    glycolysis_ub = min(glc_ub, 10.0 * fatigue_factor) * capacity_multiplier
    mito_ub = min(30.0 * fatigue_factor, o2_ub / 0.6) * capacity_multiplier

    ubs[_FALLBACK_RXN_IDS.index("GLYCOLYSIS")] = glycolysis_ub
    ubs[_FALLBACK_RXN_IDS.index("LDH")] = 60.0
    ubs[_FALLBACK_RXN_IDS.index("MITO_OXPHOS")] = mito_ub

    if include_organelle:
        ubs[_FALLBACK_RXN_IDS.index("SYN_LAC_IMPORT")] = bounds.lac_sink_ub
        ubs[_FALLBACK_RXN_IDS.index(SYN_LAC_SINK_ID)] = bounds.lac_sink_ub
        ubs[_FALLBACK_RXN_IDS.index("SYN_PYR_EXPORT")] = max(bounds.lac_sink_ub, bounds.hyper_o2_ub)
        ubs[_FALLBACK_RXN_IDS.index(SYN_HYPER_O2_ID)] = bounds.hyper_o2_ub
    else:
        for rid in ("SYN_LAC_IMPORT", SYN_LAC_SINK_ID, "SYN_PYR_EXPORT", SYN_HYPER_O2_ID):
            idx = _FALLBACK_RXN_IDS.index(rid)
            ubs[idx] = 0.0
            lbs[idx] = 0.0

    # linprog minimizes c^T x; negate objective to maximize ATPM
    c = np.zeros(n_rxn)
    c[_FALLBACK_RXN_IDS.index("ATPM")] = -1.0

    A_eq = _FALLBACK_S
    b_eq = np.zeros(A_eq.shape[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lbs, ubs)), method="highs")

    if not result.success:
        return {rid: 0.0 for rid in _FALLBACK_RXN_IDS}

    fluxes = {rid: float(result.x[i]) for i, rid in enumerate(_FALLBACK_RXN_IDS)}
    return fluxes


# ---------------------------------------------------------------------------
# FBA interface (COBRApy or fallback)
# ---------------------------------------------------------------------------
def solve_fba(
    model: Optional["Model"],
    atp_demand: float,
    bounds: OrganelleBounds,
    include_organelle: bool,
    o2_ub: float,
    glc_ub: float,
    fatigue_factor: float = 1.0,
    efficiency_drop_active: bool = False,
) -> Dict[str, float]:
    """
    Run flux balance analysis and return a reaction-id → flux mapping.

    The ATP demand reaction is capped at ``atp_demand``; the objective
    maximizes ATP production (maintenance flux).
    """
    if COBRA_AVAILABLE and model is not None:
        model.reactions.get_by_id("EX_glc__D_e").upper_bound = glc_ub
        model.reactions.get_by_id("EX_o2_e").upper_bound = o2_ub
        atpm = model.reactions.get_by_id("ATPM")
        atpm.lower_bound = 0.0
        atpm.upper_bound = atp_demand
        glycolysis_ub = min(glc_ub, 10.0 * fatigue_factor)
        mito_ub = min(30.0 * fatigue_factor, o2_ub / 0.6)
        model.reactions.get_by_id(GLYCOLYSIS_ID).upper_bound = glycolysis_ub
        model.reactions.get_by_id(MITO_OXPHOS_ID).upper_bound = mito_ub

        if include_organelle:
            lac_ub = bounds.lac_sink_ub
            o2_syn_ub = bounds.hyper_o2_ub
            # Couple import to sink capacity so lactate cannot accumulate in the organelle.
            model.reactions.get_by_id("SYN_LAC_IMPORT").upper_bound = lac_ub
            model.reactions.get_by_id(SYN_LAC_SINK_ID).upper_bound = lac_ub
            model.reactions.get_by_id("SYN_PYR_EXPORT").upper_bound = max(lac_ub, o2_syn_ub)
            model.reactions.get_by_id(SYN_HYPER_O2_ID).upper_bound = o2_syn_ub
        else:
            for rid in ("SYN_LAC_IMPORT", SYN_LAC_SINK_ID, "SYN_PYR_EXPORT", SYN_HYPER_O2_ID):
                rxn = model.reactions.get_by_id(rid)
                rxn.upper_bound = 0.0
                rxn.lower_bound = 0.0

        if efficiency_drop_active:
            reference_cap = bounds.pre_collapse_atp_ceiling
            if reference_cap is None:
                probe = solve_fba(
                    model=model,
                    atp_demand=10_000.0,
                    bounds=bounds,
                    include_organelle=include_organelle,
                    o2_ub=o2_ub,
                    glc_ub=glc_ub,
                    fatigue_factor=fatigue_factor,
                    efficiency_drop_active=False,
                )
                reference_cap = float(probe.get(ATP_DEMAND_ID, 0.0))
            dysbiosis_cap = reference_cap * (1.0 - DYSBIOSIS_ATP_PENALTY_FRACTION)
            atpm.upper_bound = min(atp_demand, dysbiosis_cap)

        solution = model.optimize()
        if solution.status != "optimal":
            return {rxn.id: 0.0 for rxn in model.reactions}

        return {rxn.id: float(solution.fluxes[rxn.id]) for rxn in model.reactions}

    return _fallback_fba(
        atp_demand,
        bounds,
        include_organelle,
        o2_ub,
        glc_ub,
        fatigue_factor,
        efficiency_drop_active,
    )


def total_ros_flux(fluxes: Dict[str, float]) -> float:
    """Aggregate ROS production flux from oxidative reactions (mmol/gDW/h)."""
    return fluxes.get(MITO_OXPHOS_ID, 0.0) * 0.0 + fluxes.get(SYN_HYPER_O2_ID, 0.0) * 1.0


def calculate_respiratory_quotient(fluxes: Dict[str, float]) -> float:
    """
    Compute the respiratory quotient from an optimal flux vector.

    RQ = CO2 efflux / O2 uptake for aerobic carbohydrate oxidation.
    """
    o2_uptake = float(fluxes.get(EX_O2_ID, 0.0))
    co2_efflux = float(fluxes.get(EX_CO2_ID, 0.0))
    if o2_uptake <= 0.0:
        raise ValueError("O2 uptake must be positive to compute RQ.")
    return co2_efflux / o2_uptake


def run_single_control_step(
    model: Optional["Model"],
    bounds: OrganelleBounds,
    atp_demand: float,
    o2_ub: float,
    glc_ub: float,
    lactate_pool: float,
    collapsed: bool = False,
    fatigue_factor: float = 1.0,
) -> Tuple[OrganelleBounds, Dict[str, float], bool]:
    """
  Execute one ML optimization / control step for the synthetic organelle.

    Returns updated bounds, flux distribution, and the dysbiosis collapse flag.
    """
    return optimize_organelle_bounds(
        model=model,
        bounds=bounds,
        atp_demand=atp_demand,
        o2_ub=o2_ub,
        glc_ub=glc_ub,
        lactate_pool=lactate_pool,
        collapsed=collapsed,
        fatigue_factor=fatigue_factor,
        n_iter=1,
    )


def dysbiosis_triggered(ros_flux: float) -> bool:
    """Return True when ROS exceeds the engineered toxicity threshold."""
    return ros_flux > ROS_TOXICITY_THRESHOLD


def measure_max_atp_capacity(
    model: Optional["Model"],
    bounds: OrganelleBounds,
    include_organelle: bool,
    o2_ub: float,
    glc_ub: float,
    fatigue_factor: float = 1.0,
    efficiency_drop_active: bool = False,
    reference_baseline: Optional[float] = None,
) -> float:
    """
    Estimate the maximum achievable ATP maintenance flux under current bounds.

    Parameters
    ----------
    efficiency_drop_active:
        When ``True``, apply the dysbiosis ATP capacity penalty (80% reduction).
    reference_baseline:
        Optional pre-collapse ATP ceiling used to compute the penalized capacity.
    """
    if reference_baseline is not None:
        baseline_max = reference_baseline
    else:
        baseline_fluxes = solve_fba(
            model=model,
            atp_demand=10_000.0,
            bounds=bounds,
            include_organelle=include_organelle,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            fatigue_factor=fatigue_factor,
            efficiency_drop_active=False,
        )
        baseline_max = float(baseline_fluxes.get(ATP_DEMAND_ID, 0.0))

    if efficiency_drop_active:
        return baseline_max * (1.0 - DYSBIOSIS_ATP_PENALTY_FRACTION)
    return baseline_max


# ---------------------------------------------------------------------------
# ML-inspired optimization of organelle reaction bounds
# ---------------------------------------------------------------------------
def reward_function(atp_produced: float, ros_flux: float, collapsed: bool) -> float:
    """
    Reinforcement-style reward for organelle parameter tuning.

    Maximizes ATP yield while penalizing ROS above the toxicity threshold.
    """
    if collapsed:
        return DYSBIOSIS_PENALTY

    excess = max(0.0, ros_flux - ROS_TOXICITY_THRESHOLD)
    return atp_produced - ROS_PENALTY_WEIGHT * (excess ** 2)


def optimize_organelle_bounds(
    model: Optional["Model"],
    bounds: OrganelleBounds,
    atp_demand: float,
    o2_ub: float,
    glc_ub: float,
    lactate_pool: float,
    collapsed: bool,
    fatigue_factor: float = 1.0,
    n_iter: int = 6,
) -> Tuple[OrganelleBounds, Dict[str, float], bool]:
    """
    Projected gradient-ascent on organelle flux upper bounds.

    Uses finite-difference gradients of the reward function evaluated via
    repeated FBA solves.  Returns updated bounds, best flux distribution,
    and an updated collapse flag.
    """
    if collapsed:
        bounds.lac_sink_ub = 0.0
        bounds.hyper_o2_ub = 0.0
        fluxes = solve_fba(model, atp_demand, bounds, True, o2_ub, glc_ub, fatigue_factor)
        return bounds, fluxes, True

    best_bounds = OrganelleBounds(
        lac_sink_ub=bounds.lac_sink_ub,
        hyper_o2_ub=bounds.hyper_o2_ub,
        lac_sink_max=bounds.lac_sink_max,
        hyper_o2_max=bounds.hyper_o2_max,
    )

    def evaluate(candidate: OrganelleBounds) -> Tuple[float, Dict[str, float], float]:
        fluxes = solve_fba(model, atp_demand, candidate, True, o2_ub, glc_ub, fatigue_factor)
        atp = fluxes.get(ATP_DEMAND_ID, 0.0)
        ros = total_ros_flux(fluxes)
        is_collapsed = dysbiosis_triggered(ros)
        return reward_function(atp, ros, is_collapsed), fluxes, ros

    best_reward, best_fluxes, best_ros = evaluate(best_bounds)

    for _ in range(n_iter):
        gradients: Dict[str, float] = {}
        for key in ("lac_sink_ub", "hyper_o2_ub"):
            perturbed = OrganelleBounds(
                lac_sink_ub=best_bounds.lac_sink_ub,
                hyper_o2_ub=best_bounds.hyper_o2_ub,
                lac_sink_max=best_bounds.lac_sink_max,
                hyper_o2_max=best_bounds.hyper_o2_max,
            )
            current = getattr(perturbed, key)
            setattr(perturbed, key, current + ML_FINITE_DIFF_EPS)
            perturbed.clip()
            up_reward, _, _ = evaluate(perturbed)

            setattr(perturbed, key, current - ML_FINITE_DIFF_EPS)
            perturbed.clip()
            down_reward, _, _ = evaluate(perturbed)

            gradients[key] = (up_reward - down_reward) / (2.0 * ML_FINITE_DIFF_EPS)

        for key, grad in gradients.items():
            new_val = getattr(best_bounds, key) + ML_LEARNING_RATE * grad
            setattr(best_bounds, key, new_val)
        best_bounds.clip()

        # Lactate pool feedback: encourage sink when lactate accumulates
        if lactate_pool > 0.8:
            best_bounds.lac_sink_ub = min(
                best_bounds.lac_sink_max,
                best_bounds.lac_sink_ub + 0.2 * lactate_pool,
            )

        # Hypoxic stress: favor hyper-O2 organelle pathway when O2 is scarce
        o2_stress = max(0.0, 1.0 - o2_ub / 25.0)
        best_bounds.hyper_o2_ub = min(
            best_bounds.hyper_o2_max,
            best_bounds.hyper_o2_ub + 0.8 * o2_stress,
        )

        reward, fluxes, ros = evaluate(best_bounds)
        if reward > best_reward:
            best_reward = reward
            best_fluxes = fluxes
            best_ros = ros

    collapsed = dysbiosis_triggered(best_ros)
    if collapsed:
        ceiling_probe = solve_fba(
            model,
            10_000.0,
            best_bounds,
            True,
            o2_ub,
            glc_ub,
            fatigue_factor,
            efficiency_drop_active=False,
        )
        best_bounds.pre_collapse_atp_ceiling = float(ceiling_probe.get(ATP_DEMAND_ID, 0.0))
        best_bounds.lac_sink_ub = 0.0
        best_bounds.hyper_o2_ub = 0.0
        best_fluxes = solve_fba(
            model,
            atp_demand,
            best_bounds,
            True,
            o2_ub,
            glc_ub,
            fatigue_factor,
            efficiency_drop_active=True,
        )

    return best_bounds, best_fluxes, collapsed


# ---------------------------------------------------------------------------
# Endurance workout simulation
# ---------------------------------------------------------------------------
def update_lactate_pool(
    lactate_pool: float,
    fluxes: Dict[str, float],
    dt: float = 1.0,
) -> float:
    """
    Integrate a pseudo-dynamic lactate pool from FBA fluxes.

    Fermentative overflow is estimated as pyruvate from glycolysis minus
    mitochondrial and organelle oxidation, minus organelle import.
    """
    glyc = fluxes.get("GLYCOLYSIS", 0.0)
    mito = fluxes.get("MITO_OXPHOS", 0.0)
    syn_o2 = fluxes.get(SYN_HYPER_O2_ID, 0.0)
    lac_import = fluxes.get("SYN_LAC_IMPORT", 0.0)
    lac_export = fluxes.get("EX_lac__L_e", 0.0)

    pyr_from_glycolysis = 2.0 * glyc
    pyr_oxidized = mito + syn_o2
    fermentative_overflow = max(0.0, pyr_from_glycolysis - pyr_oxidized)
    net_change = fermentative_overflow - lac_import - 0.35 * lac_export

    return float(max(0.0, 0.72 * lactate_pool + dt * net_change))


def _atp_demand_schedule(n_steps: int = N_STEPS) -> np.ndarray:
    """Linear ramp from moderate exercise to extreme athletic stress."""
    return np.linspace(10.0, 52.0, n_steps)


def _environmental_limits(step: int, n_steps: int) -> Tuple[float, float]:
    """
    Decrease O2 availability and slightly constrain glucose during late exercise.

    Returns (glucose_ub, o2_ub).
    """
    progress = step / max(n_steps - 1, 1)
    o2_ub = 18.0 - 13.0 * progress  # hypoxic stress at peak effort
    glc_ub = 18.0 - 13.0 * progress  # glycogen depletion
    return glc_ub, o2_ub


def simulate_cell(
    model: Optional["Model"],
    include_organelle: bool,
    n_steps: int = N_STEPS,
) -> Tuple[List[StepRecord], Optional[int]]:
    """
    Simulate an endurance workout for one cell type.

    Parameters
    ----------
    model:
        COBRApy model (``None`` ok when using fallback LP).
    include_organelle:
        Enable synthetic organelle reactions and ML tuning.
    n_steps:
        Number of discrete time-steps.

    Returns
    -------
    records, dysbiosis_step
    """
    demand_schedule = _atp_demand_schedule(n_steps)
    records: List[StepRecord] = []
    lactate_pool = 1.2
    bounds = OrganelleBounds()
    collapsed = False
    dysbiosis_step: Optional[int] = None

    for step in range(n_steps):
        demand = float(demand_schedule[step])
        glc_ub, o2_ub = _environmental_limits(step, n_steps)
        fatigue_factor = max(0.45, 1.0 - 0.5 * (step / max(n_steps - 1, 1)))

        if include_organelle and not collapsed:
            bounds, fluxes, collapsed = optimize_organelle_bounds(
                model,
                bounds,
                demand,
                o2_ub,
                glc_ub,
                lactate_pool,
                collapsed,
                fatigue_factor,
            )
        else:
            if include_organelle and collapsed:
                bounds.lac_sink_ub = 0.0
                bounds.hyper_o2_ub = 0.0
            fluxes = solve_fba(
                model,
                demand,
                bounds,
                include_organelle,
                o2_ub,
                glc_ub,
                fatigue_factor,
                efficiency_drop_active=collapsed and include_organelle,
            )

        atp_produced = fluxes.get(ATP_DEMAND_ID, 0.0)
        ros_flux = total_ros_flux(fluxes)

        lactate_pool = update_lactate_pool(lactate_pool, fluxes)

        if collapsed and dysbiosis_step is None:
            dysbiosis_step = step

        records.append(
            StepRecord(
                step=step,
                atp_demand=demand,
                atp_produced=atp_produced,
                lactate_pool=lactate_pool,
                ros_flux=ros_flux,
                syn_lac_flux=fluxes.get(SYN_LAC_SINK_ID, 0.0),
                syn_o2_flux=fluxes.get(SYN_HYPER_O2_ID, 0.0),
                oxphos_flux=fluxes.get("MITO_OXPHOS", 0.0),
                collapsed=collapsed,
                organelle_bounds=OrganelleBounds(
                    lac_sink_ub=bounds.lac_sink_ub,
                    hyper_o2_ub=bounds.hyper_o2_ub,
                    lac_sink_max=bounds.lac_sink_max,
                    hyper_o2_max=bounds.hyper_o2_max,
                ),
            )
        )

    return records, dysbiosis_step


def run_endurance_simulation(n_steps: int = N_STEPS) -> SimulationResult:
    """
    Run paired simulations: baseline muscle cell vs. organelle-enhanced cell.

    Returns
    -------
    SimulationResult
        Time-series for both conditions and the dysbiosis onset step.
    """
    enhanced_model = build_muscle_model(include_organelle=True) if COBRA_AVAILABLE else None
    baseline_model = build_muscle_model(include_organelle=False) if COBRA_AVAILABLE else None

    enhanced, dysbiosis_step = simulate_cell(enhanced_model, include_organelle=True, n_steps=n_steps)
    baseline, _ = simulate_cell(baseline_model, include_organelle=False, n_steps=n_steps)

    return SimulationResult(enhanced=enhanced, baseline=baseline, dysbiosis_step=dysbiosis_step)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_results(result: SimulationResult, output_path: str = "synthetic_organelle_simulation.png") -> plt.Figure:
    """
    Generate a three-panel figure of the endurance simulation.

    Panel 1 – ATP production vs. demand
    Panel 2 – Lactate pool: enhanced vs. baseline
    Panel 3 – Organelle flux allocation and ROS accumulation
    """
    steps = np.arange(len(result.enhanced))
    enh = result.enhanced
    base = result.baseline

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        "Synthetic Organelle Muscle Cell — Endurance Workout Simulation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: ATP
    ax1 = axes[0]
    ax1.plot(steps, [r.atp_demand for r in enh], "k--", linewidth=2, label="ATP demand")
    ax1.plot(steps, [r.atp_produced for r in enh], color="#1f77b4", linewidth=2.5, label="Enhanced ATP produced")
    ax1.plot(steps, [r.atp_produced for r in base], color="#aec7e8", linewidth=2, label="Baseline ATP produced")
    if result.dysbiosis_step is not None:
        ax1.axvline(result.dysbiosis_step, color="crimson", linestyle=":", linewidth=1.5, label="Dysbiosis collapse")
    ax1.set_ylabel("Flux (mmol/gDW/h)")
    ax1.set_title("ATP Production vs. Athletic Demand")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(bottom=0)

    # Panel 2: Lactate
    ax2 = axes[1]
    ax2.plot(steps, [r.lactate_pool for r in enh], color="#2ca02c", linewidth=2.5, label="Enhanced (organelle)")
    ax2.plot(steps, [r.lactate_pool for r in base], color="#98df8a", linewidth=2, label="Baseline")
    clearance_enh = [r.syn_lac_flux for r in enh]
    ax2_twin = ax2.twinx()
    ax2_twin.bar(steps, clearance_enh, alpha=0.25, color="#9467bd", label="Organelle lactate sink flux")
    ax2.set_ylabel("Lactate pool (mmol/gDW)")
    ax2_twin.set_ylabel("Sink flux (mmol/gDW/h)")
    ax2.set_title("Lactate Accumulation & Organelle Clearance")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # Panel 3: Organelle fluxes & ROS
    ax3 = axes[2]
    ax3.plot(steps, [r.syn_lac_flux for r in enh], color="#9467bd", linewidth=2, label="Lactate sink flux")
    ax3.plot(steps, [r.syn_o2_flux for r in enh], color="#ff7f0e", linewidth=2, label="Hyper-O2 flux")
    ax3.plot(steps, [r.oxphos_flux for r in enh], color="#8c564b", linewidth=1.5, linestyle="--", label="Host OXPHOS")
    ax3.set_ylabel("Flux (mmol/gDW/h)")
    ax3_ros = ax3.twinx()
    ax3_ros.plot(steps, [r.ros_flux for r in enh], color="crimson", linewidth=2.5, label="ROS flux")
    ax3_ros.axhline(ROS_TOXICITY_THRESHOLD, color="crimson", linestyle="--", alpha=0.5, label="ROS toxicity threshold")
    ax3_ros.axhline(
        ROS_TOXICITY_THRESHOLD * ROS_COLLAPSE_FACTOR,
        color="darkred",
        linestyle=":",
        alpha=0.5,
        label="Collapse threshold",
    )
    ax3_ros.set_ylabel("ROS proxy flux")
    ax3.set_xlabel("Time step (endurance workout progression)")
    ax3.set_title("Organelle Flux Shifting & Oxidative Stress")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_ros.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def print_summary(result: SimulationResult) -> None:
    """Print a concise textual summary of simulation outcomes."""
    enh = result.enhanced
    base = result.baseline
    final_enh = enh[-1]
    final_base = base[-1]

    print("=" * 72)
    print("  Synthetic Organelle Skeletal Muscle Simulation — Summary")
    print("=" * 72)
    print(f"  Solver backend       : {'COBRApy' if COBRA_AVAILABLE else 'SciPy linprog (fallback)'}")
    print(f"  Time steps           : {len(enh)}")
    print(f"  Peak ATP demand      : {enh[-1].atp_demand:.1f} mmol/gDW/h")
    print(f"  Enhanced final ATP   : {final_enh.atp_produced:.2f} mmol/gDW/h")
    print(f"  Baseline final ATP   : {final_base.atp_produced:.2f} mmol/gDW/h")
    print(f"  ATP gain (enhanced)  : {final_enh.atp_produced - final_base.atp_produced:+.2f} mmol/gDW/h")
    print(f"  Enhanced lactate pool: {final_enh.lactate_pool:.2f} mmol/gDW")
    print(f"  Baseline lactate pool: {final_base.lactate_pool:.2f} mmol/gDW")
    print(f"  Final ROS flux       : {final_enh.ros_flux:.3f}")
    if result.dysbiosis_step is not None:
        print(f"  [!] Dysbiosis collapse at step {result.dysbiosis_step}")
    else:
        print("  [OK] No dysbiosis collapse - ROS remained manageable")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full simulation pipeline and display results."""
    if not COBRA_AVAILABLE and not SCIPY_AVAILABLE:
        raise SystemExit(
            "ERROR: Install at least one of cobra or scipy.\n"
            "  pip install cobra numpy matplotlib scipy"
        )

    print("Building metabolic models and running endurance simulation...")
    result = run_endurance_simulation(n_steps=N_STEPS)
    print_summary(result)

    output_file = "synthetic_organelle_simulation.png"
    fig = plot_results(result, output_path=output_file)
    print(f"\nFigure saved to: {output_file}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
