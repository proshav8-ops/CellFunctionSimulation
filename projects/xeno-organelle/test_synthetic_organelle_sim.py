#!/usr/bin/env python3
"""
QA validation suite for the synthetic organelle skeletal muscle simulation.

Validates thermodynamic closure, elemental mass balance, mammalian respiratory
quotient (RQ) physiology, and dysbiosis-collapse control logic using the
standard library ``unittest`` framework.
"""

from __future__ import annotations

import logging
import sys
import unittest
from typing import Dict, Optional

import synthetic_organelle_sim as sim

try:
    from cobra import Model

    COBRA_AVAILABLE = True
except ImportError:  # pragma: no cover
    COBRA_AVAILABLE = False
    Model = object  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[QA-BIO] %(levelname)s | %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger("synthetic_organelle_qa")

FLOAT_TOLERANCE = 1e-4


@unittest.skipUnless(COBRA_AVAILABLE, "COBRApy is required for metabolic model QA tests.")
class SyntheticOrganelleModelQATestCase(unittest.TestCase):
    """Rigorous QA tests for the synthetic organelle metabolic simulation."""

    def setUp(self) -> None:
        """Instantiate a pristine metabolic model before each test."""
        LOGGER.info("Setting up fresh human skeletal muscle model instance.")
        self.model: Model = sim.build_muscle_model(include_organelle=True)
        self.bounds = sim.OrganelleBounds(
            lac_sink_ub=25.0,
            hyper_o2_ub=25.0,
            lac_sink_max=40.0,
            hyper_o2_max=25.0,
        )

    def tearDown(self) -> None:
        """Release model reference after each test."""
        self.model = None  # type: ignore[assignment]
        LOGGER.info("Tear-down complete.\n")

    # ------------------------------------------------------------------
    # Test Case 1: Thermodynamic conservation
    # ------------------------------------------------------------------
    def test_01_thermodynamic_conservation_no_free_energy_loop(self) -> None:
        """
        With all external nutrient imports clamped to zero, ATP production
        must be exactly zero (no perpetual motion / thermodynamic leak).
        """
        LOGGER.info(
            "TEST 1 | Validating first law closure: zero nutrients => zero ATP flux."
        )

        glucose_exchange = self.model.reactions.get_by_id(sim.EX_GLC_ID)
        oxygen_exchange = self.model.reactions.get_by_id(sim.EX_O2_ID)

        glucose_exchange.lower_bound = 0.0
        glucose_exchange.upper_bound = 0.0
        oxygen_exchange.lower_bound = 0.0
        oxygen_exchange.upper_bound = 0.0

        atpm = self.model.reactions.get_by_id(sim.ATP_DEMAND_ID)
        atpm.lower_bound = 0.0
        atpm.upper_bound = 10_000.0
        self.model.objective = sim.ATP_DEMAND_ID

        solution = self.model.optimize()
        atp_flux = float(solution.fluxes[sim.ATP_DEMAND_ID])

        LOGGER.info("  Optimizer status : %s", solution.status)
        LOGGER.info("  Optimal ATP flux : %.6f mmol/gDW/h", atp_flux)

        self.assertEqual(
            solution.status,
            "optimal",
            msg="FBA should remain feasible under sealed boundary conditions.",
        )
        self.assertAlmostEqual(
            atp_flux,
            0.0,
            places=9,
            msg=(
                "Thermodynamic leak detected: ATP was produced without "
                "glucose or oxygen import."
            ),
        )

    # ------------------------------------------------------------------
    # Test Case 2: Mass & charge balance
    # ------------------------------------------------------------------
    def test_02_mass_and_charge_balance_for_custom_reactions(self) -> None:
        """
        Every host and organelle core reaction must be elementally closed
        according to COBRApy's ``check_mass_balance()`` implementation.
        """
        LOGGER.info(
            "TEST 2 | Validating elemental mass conservation for curated reactions."
        )

        for reaction_id in sim.MASS_BALANCE_REACTION_IDS:
            reaction = self.model.reactions.get_by_id(reaction_id)
            imbalance: Dict[str, float] = reaction.check_mass_balance()

            LOGGER.info("  Reaction %-22s | imbalance: %s", reaction_id, imbalance or "{}")

            self.assertEqual(
                imbalance,
                {},
                msg=(
                    f"Reaction '{reaction_id}' violates mass balance: "
                    f"{imbalance}"
                ),
            )

    # ------------------------------------------------------------------
    # Test Case 3: Respiratory quotient (RQ) under aerobic glucose oxidation
    # ------------------------------------------------------------------
    def test_03_respiratory_quotient_equals_unity_for_aerobic_glucose(self) -> None:
        """
        Under purely aerobic glucose oxidation (organelle off, LDH blocked),
        RQ = CO2 production / O2 consumption must equal 1.0 for glucose.
        """
        LOGGER.info(
            "TEST 3 | Validating mammalian carbohydrate RQ under aerobic conditions."
        )

        aerobic_model = sim.build_muscle_model(include_organelle=False)

        for organelle_rxn in (
            sim.SYN_LAC_SINK_ID,
            sim.SYN_HYPER_O2_ID,
            "SYN_LAC_IMPORT",
            "SYN_PYR_EXPORT",
        ):
            rxn = aerobic_model.reactions.get_by_id(organelle_rxn)
            rxn.lower_bound = 0.0
            rxn.upper_bound = 0.0

        aerobic_model.reactions.get_by_id(sim.LDH_ID).upper_bound = 0.0
        aerobic_model.reactions.get_by_id(sim.EX_GLC_ID).upper_bound = 20.0
        aerobic_model.reactions.get_by_id(sim.EX_O2_ID).upper_bound = 60.0
        aerobic_model.reactions.get_by_id(sim.ATP_DEMAND_ID).lower_bound = 10.0
        aerobic_model.objective = sim.ATP_DEMAND_ID

        solution = aerobic_model.optimize()
        fluxes = {rxn.id: float(solution.fluxes[rxn.id]) for rxn in aerobic_model.reactions}

        rq = sim.calculate_respiratory_quotient(fluxes)
        co2_flux = fluxes[sim.EX_CO2_ID]
        o2_flux = fluxes[sim.EX_O2_ID]

        LOGGER.info("  Optimizer status : %s", solution.status)
        LOGGER.info("  CO2 efflux       : %.6f mmol/gDW/h", co2_flux)
        LOGGER.info("  O2 uptake        : %.6f mmol/gDW/h", o2_flux)
        LOGGER.info("  Respiratory RQ   : %.6f", rq)

        self.assertEqual(solution.status, "optimal")
        self.assertGreater(o2_flux, 0.0, msg="Aerobic test requires O2 consumption.")
        self.assertAlmostEqual(
            rq,
            1.0,
            delta=FLOAT_TOLERANCE,
            msg="Aerobic glucose RQ must equal 1.0 for mammalian physiology.",
        )

    # ------------------------------------------------------------------
    # Test Case 4: ML control loop & dysbiosis collapse
    # ------------------------------------------------------------------
    def test_04_dysbiosis_collapse_and_atp_efficiency_penalty(self) -> None:
        """
        Extreme exercise forcing hyper-organelle respiration must trigger
        dysbiosis when ROS > 5.0 and impose an exact 80% ATP capacity penalty.
        """
        LOGGER.info(
            "TEST 4 | Validating ML dysbiosis control and ATP efficiency cliff."
        )

        stress_model = sim.build_muscle_model(include_organelle=True)
        stress_model.reactions.get_by_id(sim.MITO_OXPHOS_ID).upper_bound = 0.0
        stress_model.reactions.get_by_id("EX_lac__L_e").upper_bound = 0.0

        extreme_bounds = sim.OrganelleBounds(
            lac_sink_ub=25.0,
            hyper_o2_ub=25.0,
            lac_sink_max=40.0,
            hyper_o2_max=25.0,
        )

        atp_demand = 120.0
        o2_ub = 30.0
        glc_ub = 30.0
        lactate_pool = 8.0

        pre_collapse_max_atp = sim.measure_max_atp_capacity(
            model=stress_model,
            bounds=extreme_bounds,
            include_organelle=True,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            fatigue_factor=1.0,
            efficiency_drop_active=False,
        )

        pre_control_fluxes = sim.solve_fba(
            model=stress_model,
            atp_demand=atp_demand,
            bounds=extreme_bounds,
            include_organelle=True,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            fatigue_factor=1.0,
            efficiency_drop_active=False,
        )
        pre_control_ros = sim.total_ros_flux(pre_control_fluxes)

        LOGGER.info("  Pre-collapse ATP capacity : %.4f mmol/gDW/h", pre_collapse_max_atp)
        LOGGER.info("  Pre-control ROS flux      : %.4f units", pre_control_ros)

        self.assertGreater(
            pre_control_ros,
            5.0,
            msg="Extreme exercise scenario must exceed ROS toxicity (> 5.0).",
        )
        self.assertTrue(
            sim.dysbiosis_triggered(pre_control_ros),
            msg="ROS toxicity detector failed to register dysbiosis.",
        )

        updated_bounds, fluxes, efficiency_drop_flag = sim.run_single_control_step(
            model=stress_model,
            bounds=extreme_bounds,
            atp_demand=atp_demand,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            lactate_pool=lactate_pool,
            collapsed=False,
            fatigue_factor=1.0,
        )

        ros_level = sim.total_ros_flux(fluxes)
        LOGGER.info("  Post-control ROS flux      : %.4f units", ros_level)
        LOGGER.info("  Dysbiosis flag             : %s", efficiency_drop_flag)

        self.assertTrue(
            efficiency_drop_flag,
            msg="Controller must raise the engineered efficiency-drop flag.",
        )

        expected_post_max = pre_collapse_max_atp * (
            1.0 - sim.DYSBIOSIS_ATP_PENALTY_FRACTION
        )

        if updated_bounds.pre_collapse_atp_ceiling is None:
            updated_bounds.pre_collapse_atp_ceiling = pre_collapse_max_atp

        penalized_fluxes = sim.solve_fba(
            model=stress_model,
            atp_demand=atp_demand,
            bounds=updated_bounds,
            include_organelle=True,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            fatigue_factor=1.0,
            efficiency_drop_active=True,
        )
        realized_atp = float(penalized_fluxes[sim.ATP_DEMAND_ID])
        post_collapse_max_atp = sim.measure_max_atp_capacity(
            model=stress_model,
            bounds=updated_bounds,
            include_organelle=True,
            o2_ub=o2_ub,
            glc_ub=glc_ub,
            fatigue_factor=1.0,
            efficiency_drop_active=True,
            reference_baseline=pre_collapse_max_atp,
        )

        LOGGER.info("  Post-collapse ATP capacity : %.4f mmol/gDW/h", post_collapse_max_atp)
        LOGGER.info("  Realized penalized ATP flux: %.4f mmol/gDW/h", realized_atp)
        LOGGER.info("  Expected penalized capacity: %.4f mmol/gDW/h", expected_post_max)
        LOGGER.info(
            "  Penalty fraction           : %.0f%%",
            sim.DYSBIOSIS_ATP_PENALTY_FRACTION * 100.0,
        )

        self.assertAlmostEqual(
            realized_atp,
            expected_post_max,
            delta=FLOAT_TOLERANCE,
            msg=(
                "Dysbiosis cliff must reduce achievable ATP flux by exactly "
                f"{sim.DYSBIOSIS_ATP_PENALTY_FRACTION:.0%}."
            ),
        )
        self.assertAlmostEqual(
            post_collapse_max_atp / pre_collapse_max_atp,
            1.0 - sim.DYSBIOSIS_ATP_PENALTY_FRACTION,
            delta=FLOAT_TOLERANCE,
            msg="Relative ATP capacity reduction must equal the penalty fraction.",
        )


class SyntheticOrganelleImportSanityTest(unittest.TestCase):
    """Lightweight import checks that do not require COBRApy."""

    def test_module_alias_imports_core_api(self) -> None:
        """The public alias module must expose the simulation API."""
        LOGGER.info("IMPORT | Validating synthetic_organelle_sim public API.")
        required = (
            "build_muscle_model",
            "solve_fba",
            "optimize_organelle_bounds",
            "run_single_control_step",
            "measure_max_atp_capacity",
            "calculate_respiratory_quotient",
            "dysbiosis_triggered",
            "MASS_BALANCE_REACTION_IDS",
            "ROS_TOXICITY_THRESHOLD",
            "DYSBIOSIS_ATP_PENALTY_FRACTION",
        )
        for name in required:
            self.assertTrue(
                hasattr(sim, name),
                msg=f"Missing required export: {name}",
            )


def run_qa_suite(verbosity: int = 2) -> unittest.TestResult:
    """Entry point for running the QA suite programmatically."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(SyntheticOrganelleImportSanityTest))
    suite.addTests(loader.loadTestsFromTestCase(SyntheticOrganelleModelQATestCase))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    LOGGER.info("=" * 72)
    LOGGER.info("Starting Synthetic Organelle Computational Biology QA Suite")
    LOGGER.info("=" * 72)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_qa_suite(verbosity=2)
    sys.exit(0 if result.wasSuccessful() else 1)
