"""The jobs a flow is built from: LAMMPS stage makers, and analysis jobs."""
from finite_temp_properties.workflow.jobs.analysis import (
    analyze_liquid_free_energy,
    analyze_potential_switch,
    analyze_solid_free_energy,
    build_gibbs_curve,
)
from finite_temp_properties.workflow.jobs.makers import (
    CrystalEnthalpyMaker,
    CrystalMSDMaker,
    FrenkelLaddMaker,
    MeltEquilibrationMaker,
    PotentialSwitchMaker,
    QuenchMaker,
    UFMSwitchLeg1Maker,
    UFMSwitchLeg2Maker,
)

__all__ = [
    "CrystalEnthalpyMaker",
    "CrystalMSDMaker",
    "FrenkelLaddMaker",
    "MeltEquilibrationMaker",
    "PotentialSwitchMaker",
    "QuenchMaker",
    "UFMSwitchLeg1Maker",
    "UFMSwitchLeg2Maker",
    "analyze_liquid_free_energy",
    "analyze_potential_switch",
    "analyze_solid_free_energy",
    "build_gibbs_curve",
]
