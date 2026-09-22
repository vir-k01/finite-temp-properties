"""The whole thing: G_crystal(T) and G_amorphous(T) under the target potential.

    sbatch 04_gibbs_curve.sbatch        # or, inside an allocation:
    python 04_gibbs_curve.py SrTiO3

Production settings. Both branches run their TI on the reference potential,
then one NPT switch per phase carries each free energy onto the target; every
enthalpy comes from target-potential MD. The result is written as JSON for
05_plot_gibbs_curve.py.

Needs BOTH engines: the CPU build for the TI legs and a GPU for the switches
and the H(T) runs. Allocate a GPU node.
"""
from __future__ import annotations

import json
import sys

from jobflow import run_locally
from py_oats.structure_generator.generator import get_amorphous_structure
from pymatgen.core import Composition, Lattice, Structure

from common import configure_gibbs, output_of
from finite_temp_properties.workflow.flows import GibbsCurveMaker

formula = sys.argv[1] if len(sys.argv) > 1 else "SrTiO3"
out_file = f"gibbs_{formula}.json"

T0 = 950.0        # crystal anchor: the middle of the window you report
TA = 2900.0       # melt anchor: ~1.3x the highest liquidus in the system

# Production: load a cell relaxed with the target potential instead.
crystal = Structure(
    Lattice.cubic(3.905),
    ["Sr", "Ti", "O", "O", "O"],
    [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
)
crystal.make_supercell(3)

melt = get_amorphous_structure(Composition(formula))

maker = GibbsCurveMaker(
    temperature_crystal=T0,
    temperature_melt=TA,
    temperatures=list(range(700, 1201, 25)),   # the grid G(T) is reported on
)
configure_gibbs(maker)

flow = maker.make(crystal=crystal, melt=melt)
responses = run_locally(flow, create_folders=True)
curve = output_of(flow, responses, "build_gibbs_curve")

with open(out_file, "w") as f:
    json.dump(curve.model_dump(), f, indent=2)

print(f"""
anchors
  crystal  T0 = {curve.t_crystal_anchor:.0f} K   G = {curve.g_crystal_anchor:.6f} eV/atom
           H(T0) = {curve.h_crystal_anchor:.6f}   S = {curve.s_crystal_anchor:.2f} J/mol-atom/K
  melt     Ta = {curve.t_melt_anchor:.0f} K   G = {curve.g_melt_anchor:.6f} eV/atom
           S = {curve.s_melt_anchor:.2f} J/mol-atom/K

dG(amorphous - crystal), meV/atom""")
for t, gc, ga in zip(curve.temperatures, curve.g_crystal, curve.g_amorphous):
    print(f"  {t:7.0f}  {1000 * (ga - gc):8.2f}")

print(f"\nwritten to {out_file}")

# S must be positive and the melt's must exceed the crystal's -- if not, an
# anchor is wrong, not the physics.
if curve.s_melt_anchor <= curve.s_crystal_anchor:
    print("\nWARNING: melt entropy is not above the crystal's. Check that "
          "both branches used the anchors they claim.")
