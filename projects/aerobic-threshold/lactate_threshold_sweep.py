#!/usr/bin/env python3
"""
Aerobic lactate threshold sweep — baseline host muscle FBA.

Reuses the mass-balanced host network from xeno-organelle (organelle off)
and sweeps ATP maintenance demand to estimate the lactate threshold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Reuse validated host model from sibling project
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xeno-organelle"))
from synthetic_organelle_muscle_simulation import (  # noqa: E402
    ATP_DEMAND_ID,
    EX_CO2_ID,
    EX_O2_ID,
    MITO_OXPHOS_ID,
    build_muscle_model,
)

EX_LAC_ID = "EX_lac__L_e"


@dataclass
class SweepPoint:
    step: int
    atp_demand: float
    atp_flux: float
    lactate_export: float
    o2_uptake: float
    co2_export: float
    oxphos_flux: float
    rq: float


def respiratory_quotient(co2_flux: float, o2_flux: float) -> float:
    if abs(o2_flux) < 1e-9:
        return float("nan")
    return abs(co2_flux) / abs(o2_flux)


def run_sweep(n_steps: int = 25, demand_start: float = 2.0, demand_end: float = 55.0) -> List[SweepPoint]:
    model = build_muscle_model(include_organelle=False)
    demands = np.linspace(demand_start, demand_end, n_steps)
    records: List[SweepPoint] = []

    for i, demand in enumerate(demands, start=1):
        o2_cap = max(6.0, 25.0 - 0.8 * i)
        model.reactions.get_by_id(EX_O2_ID).upper_bound = o2_cap
        atp_rxn = model.reactions.get_by_id(ATP_DEMAND_ID)
        atp_rxn.upper_bound = float(demand)
        atp_rxn.lower_bound = float(demand)

        sol = model.optimize()
        if sol.status != "optimal":
            records.append(
                SweepPoint(
                    step=i,
                    atp_demand=float(demand),
                    atp_flux=0.0,
                    lactate_export=0.0,
                    o2_uptake=0.0,
                    co2_export=0.0,
                    oxphos_flux=0.0,
                    rq=float("nan"),
                )
            )
            continue

        fluxes = sol.fluxes
        lac = float(fluxes.get(EX_LAC_ID, 0.0))
        o2 = float(fluxes.get(EX_O2_ID, 0.0))
        co2 = float(fluxes.get(EX_CO2_ID, 0.0))
        records.append(
            SweepPoint(
                step=i,
                atp_demand=float(demand),
                atp_flux=float(fluxes.get(ATP_DEMAND_ID, 0.0)),
                lactate_export=lac,
                o2_uptake=o2,
                co2_export=co2,
                oxphos_flux=float(fluxes.get(MITO_OXPHOS_ID, 0.0)),
                rq=respiratory_quotient(co2, o2),
            )
        )
    return records


def estimate_threshold(records: List[SweepPoint], lac_floor: float = 0.05) -> Tuple[int, float]:
    for i, rec in enumerate(records):
        if rec.lactate_export <= lac_floor:
            continue
        tail = records[i : i + 3]
        if len(tail) >= 2 and all(r.lactate_export > lac_floor for r in tail):
            return rec.step, rec.atp_demand
    last = records[-1]
    return last.step, last.atp_demand


def plot_sweep(records: List[SweepPoint], threshold_step: int, out_path: Path) -> None:
    steps = [r.step for r in records]
    lac = [r.lactate_export for r in records]
    atp = [r.atp_demand for r in records]
    rq = [r.rq for r in records]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, lac, "o-", color="#c0392b", label="Lactate export")
    axes[0].axvline(threshold_step, color="#2c3e50", ls="--", label="Estimated threshold")
    axes[0].set_ylabel("Lactate export (mmol/gDW/h)")
    axes[0].legend()
    axes[0].set_title("Aerobic Threshold — Lactate Inflection")

    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    tick_idx = list(range(0, len(steps), max(1, len(steps) // 8)))
    ax2.set_xticks([steps[j] for j in tick_idx])
    ax2.set_xticklabels([f"{atp[j]:.0f}" for j in tick_idx])
    ax2.set_xlabel("ATP demand (mmol/gDW/h)")

    axes[1].plot(steps, rq, "s-", color="#2980b9", label="RQ")
    axes[1].axhline(1.0, color="#7f8c8d", ls=":", label="RQ = 1.0")
    axes[1].set_xlabel("Work step")
    axes[1].set_ylabel("Respiratory quotient")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    records = run_sweep()
    threshold_step, threshold_demand = estimate_threshold(records)
    thr_rec = records[threshold_step - 1]

    print("=== Aerobic Threshold Sweep ===")
    print(f"Estimated lactate threshold: step {threshold_step}, ATP demand ~ {threshold_demand:.2f} mmol/gDW/h")
    print(f"At threshold - lactate export: {thr_rec.lactate_export:.3f}, RQ: {thr_rec.rq:.3f}, OXPHOS: {thr_rec.oxphos_flux:.3f}")

    out = Path(__file__).resolve().parent / "outputs" / "lactate_threshold_sweep.png"
    plot_sweep(records, threshold_step, out)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    main()
