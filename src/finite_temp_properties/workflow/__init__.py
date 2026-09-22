"""Jobs and flows.

``workflow.jobs``  -- one maker per LAMMPS stage, plus the analysis jobs.
``workflow.flows`` -- the three flows built out of them.

Everything public is re-exported here, so
``from finite_temp_properties.workflow import GibbsCurveMaker`` works.
"""
from finite_temp_properties.workflow.flows import (
    GibbsCurveMaker,
    LiquidFreeEnergyMaker,
    SolidFreeEnergyMaker,
)
from finite_temp_properties.workflow.jobs import (
    CrystalEnthalpyMaker,
    CrystalMSDMaker,
    FrenkelLaddMaker,
    MeltEquilibrationMaker,
    PotentialSwitchMaker,
    QuenchMaker,
    UFMSwitchLeg1Maker,
    UFMSwitchLeg2Maker,
    analyze_liquid_free_energy,
    analyze_potential_switch,
    analyze_solid_free_energy,
    build_gibbs_curve,
)

__all__ = [
    "CrystalEnthalpyMaker",
    "CrystalMSDMaker",
    "FrenkelLaddMaker",
    "GibbsCurveMaker",
    "LiquidFreeEnergyMaker",
    "MeltEquilibrationMaker",
    "PotentialSwitchMaker",
    "QuenchMaker",
    "SolidFreeEnergyMaker",
    "UFMSwitchLeg1Maker",
    "UFMSwitchLeg2Maker",
    "analyze_liquid_free_energy",
    "analyze_potential_switch",
    "analyze_solid_free_energy",
    "build_gibbs_curve",
]
