"""Look at the LAMMPS inputs before spending node-hours on them.

Builds the full G(T) flow, prints its job graph, and writes out the in.lammps
each stage would run. Nothing is executed -- run this on a login node.

    python 01_inspect_inputs.py [out_dir]

Reading the switch inputs is the fastest way to catch a wrong potential path,
a barostat that should have been triclinic, or a thermostat placed before the
ti/spring fixes (which silently costs ~185 meV/atom).
"""
from __future__ import annotations

import sys

from pymatgen.core import Lattice, Structure

from common import (
    GRACE_FS, GRACE_SMAX, all_jobs, configure_gibbs, engine_of, render)
from finite_temp_properties.workflow.flows import GibbsCurveMaker
from finite_temp_properties.workflow.jobs import (
    CrystalEnthalpyMaker,
    CrystalMSDMaker,
    FrenkelLaddMaker,
    MeltEquilibrationMaker,
    PotentialSwitchMaker,
    QuenchMaker,
    UFMSwitchLeg1Maker,
    UFMSwitchLeg2Maker,
)

out_dir = sys.argv[1] if len(sys.argv) > 1 else "inputs_preview"

# An ideal cubic SrTiO3 3x3x3 supercell, 135 atoms. For production, start from
# a cell relaxed with the potential you are about to use.
cell = Structure(
    Lattice.cubic(3.905),
    ["Sr", "Ti", "O", "O", "O"],
    [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
)
cell.make_supercell(3)

# ---------------------------------------------------------------- the graph

maker = GibbsCurveMaker(temperature_crystal=950.0, temperature_melt=2900.0)
configure_gibbs(maker)
flow = maker.make(crystal=cell, melt=cell)

print(f"{flow.name}: {len(list(all_jobs(flow)))} jobs\n")
for job in all_jobs(flow):
    engine = engine_of(job)
    print(f"  {job.name:28s} {engine or '(no MD)'}")
print("""
  reference = the CPU build with ufm and ti/spring
  target    = the TensorFlow build with plain `grace`
""")

# ------------------------------------------------------------- the inputs
#
# Stages chained with make_from() only learn their numbers at runtime, so
# stand-ins go in here: three spring constants for the crystal, six pair
# sigmas for the three-species melt.

stand_in_springs = [2.0, 3.0, 1.5]                  # eV/A^2, one per species
stand_in_sigmas = [1.6, 1.4, 1.2, 1.3, 1.1, 1.0]    # A, one per type pair

previews = [
    ("crystal_msd", CrystalMSDMaker(settings={"temperature": 950.0}), (), {}),
    ("frenkel_ladd", FrenkelLaddMaker(settings={"temperature": 950.0}),
     (stand_in_springs,), {}),
    ("melt_equilibration", MeltEquilibrationMaker(settings={"temperature": 2900.0}),
     (), {}),
    ("ufm_switch_leg1", UFMSwitchLeg1Maker(settings={"temperature": 2900.0}),
     (stand_in_sigmas,), {}),
    ("ufm_switch_leg2", UFMSwitchLeg2Maker(settings={"temperature": 2900.0}),
     (stand_in_sigmas, 1.35), {}),
    ("potential_switch_crystal",
     PotentialSwitchMaker(settings={"temperature": 950.0, "barostat": "tri"}),
     (), {}),
    ("crystal_enthalpy", CrystalEnthalpyMaker(settings={"temperature": 950.0}),
     (), {}),
    ("quench_enthalpy", QuenchMaker(settings={"temperature": 2900.0}), (), {}),
]

print(f"writing inputs under {out_dir}/\n")
for label, stage, args, kwargs in previews:
    path = render(stage, cell, f"{out_dir}/{label}", *args, **kwargs)
    print(f"  {label:26s} {path}")

print(f"""
Check in these files that:
  - the TI legs point at   {GRACE_FS}
  - the switches and H(T)  {GRACE_SMAX}
  - every `fix ... ti/spring` comes BEFORE the thermostat fix
  - the crystal runs use a triclinic barostat, the melt runs an isotropic one
""")
