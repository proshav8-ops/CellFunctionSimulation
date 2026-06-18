#!/usr/bin/env python3
"""
Flux → phenotype bridge for cell-function simulations.

Scores endurance-relevant phenotypes from a small flux dictionary.
Includes demo profiles for native host vs. xeno-organelle-enhanced cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PhenotypeScores:
    vo2_proxy: float
    lactate_clearance: float
    fatigue_resistance: float
    oxidative_stress_risk: float
    endurance_index: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "vo2_proxy": self.vo2_proxy,
            "lactate_clearance": self.lactate_clearance,
            "fatigue_resistance": self.fatigue_resistance,
            "oxidative_stress_risk": self.oxidative_stress_risk,
            "endurance_index": self.endurance_index,
        }


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def score_phenotypes(fluxes: Mapping[str, float], *, atp_demand: float = 20.0) -> PhenotypeScores:
    """
    Map fluxes (mmol/gDW/h) to normalized phenotype scores in [0, 1].

    Expected keys (defaults to 0 if missing):
    atp_produced, lactate_export, o2_uptake, ros_flux, syn_lac_flux, oxphos_flux
    """
    atp = float(fluxes.get("atp_produced", fluxes.get("atp_flux", 0.0)))
    lac = float(fluxes.get("lactate_export", fluxes.get("lactate_pool", 0.0)))
    o2 = float(fluxes.get("o2_uptake", 0.0))
    ros = float(fluxes.get("ros_flux", 0.0))
    syn_lac = float(fluxes.get("syn_lac_flux", 0.0))
    oxphos = float(fluxes.get("oxphos_flux", 0.0))

    vo2_proxy = _clamp01(o2 / 40.0)
    lactate_clearance = _clamp01((syn_lac + oxphos * 0.2) / (lac + 1.0))
    fatigue_resistance = _clamp01(atp / max(atp_demand, 1e-6))
    oxidative_stress_risk = _clamp01(ros / 5.0)
    endurance_index = _clamp01(
        0.35 * vo2_proxy
        + 0.30 * lactate_clearance
        + 0.25 * fatigue_resistance
        + 0.10 * (1.0 - oxidative_stress_risk)
    )

    return PhenotypeScores(
        vo2_proxy=vo2_proxy,
        lactate_clearance=lactate_clearance,
        fatigue_resistance=fatigue_resistance,
        oxidative_stress_risk=oxidative_stress_risk,
        endurance_index=endurance_index,
    )


def plot_radar(profiles: Dict[str, PhenotypeScores], out_path: Path) -> None:
    labels = ["VO₂ proxy", "Lactate clearance", "Fatigue resistance", "Low ROS risk", "Endurance index"]
    keys = ["vo2_proxy", "lactate_clearance", "fatigue_resistance", "oxidative_stress_risk", "endurance_index"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = {"Native host": "#3498db", "Xeno-organelle": "#e74c3c"}

    for name, scores in profiles.items():
        vals = [getattr(scores, k) for k in keys]
        if keys[3] == "oxidative_stress_risk":
            vals[3] = 1.0 - vals[3]  # invert for "Low ROS risk" axis
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=colors.get(name, None))
        ax.fill(angles, vals, alpha=0.15, color=colors.get(name, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Structure → Function: Phenotype Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    native = score_phenotypes(
        {
            "atp_produced": 18.0,
            "lactate_export": 6.5,
            "o2_uptake": 22.0,
            "ros_flux": 1.2,
            "syn_lac_flux": 0.0,
            "oxphos_flux": 8.0,
        },
        atp_demand=22.0,
    )
    xeno = score_phenotypes(
        {
            "atp_produced": 26.0,
            "lactate_export": 2.1,
            "o2_uptake": 28.0,
            "ros_flux": 4.2,
            "syn_lac_flux": 7.5,
            "oxphos_flux": 10.0,
        },
        atp_demand=26.0,
    )

    print("=== Phenotype Bridge ===")
    for label, scores in [("Native host", native), ("Xeno-organelle", xeno)]:
        print(f"\n{label}:")
        for k, v in scores.as_dict().items():
            print(f"  {k}: {v:.3f}")

    out = Path(__file__).resolve().parent / "outputs" / "phenotype_radar.png"
    plot_radar({"Native host": native, "Xeno-organelle": xeno}, out)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
