"""The crystal branch alone: F(T0) by a Frenkel-Ladd switch.

The cheapest real run in the package, and the one to do first -- if the
Einstein reference or the spring constants are wrong, everything downstream
is wrong by the same amount.

    salloc -N 1 -C cpu -q interactive -t 2:00:00
    python 02_solid_free_energy.py

Runs on the reference build only (with_switch=False), so no GPU is needed.
Set with_switch=True to also transport F onto the target potential, which
needs a GPU node.
"""
from __future__ import annotations

from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from common import configure_solid, output_of
from finite_temp_properties.workflow.flows import SolidFreeEnergyMaker
from finite_temp_properties.workflow.jobs import CrystalMSDMaker, FrenkelLaddMaker

T0 = 950.0          # anchor temperature: the middle of the window you report

cell = Structure(
    Lattice.cubic(3.905),
    ["Sr", "Ti", "O", "O", "O"],
    [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]],
)
cell.make_supercell(3)          # 135 atoms

maker = SolidFreeEnergyMaker(
    # Short settings so this finishes inside an interactive allocation.
    # Production: leave the defaults (15k equil / 20k msd, 40k per sweep).
    msd_maker=CrystalMSDMaker(settings={
        "temperature": T0, "n_equil": 5000, "n_msd": 10000}),
    frenkel_ladd_maker=FrenkelLaddMaker(settings={
        "temperature": T0, "n_equil": 5000, "n_switch": 15000}),
    with_switch=False,          # reference potential only; no GPU needed
)
configure_solid(maker)

flow = maker.make(cell)
responses = run_locally(flow, create_folders=True)
doc = output_of(flow, responses, "analyze_solid_free_energy")

print(f"""
crystal free energy at {doc.temperature:.0f} K
  F            {doc.free_energy:12.6f} eV/atom
  Einstein ref {doc.einstein_reference:12.6f} eV/atom
  work W       {doc.work:12.6f} eV/atom
  hysteresis   {doc.hysteresis:12.6f} eV/atom   <- the statistical error scale
  springs k_i  {['%.3f' % k for k in doc.spring_constants]} eV/A^2 for {doc.species}
  volume/atom  {doc.volume_per_atom:.4f} A^3   ({doc.natoms} atoms)
""")

if doc.hysteresis > 0.005:
    print("WARNING: hysteresis above 5 meV/atom -- switch more slowly "
          "(raise n_switch) before trusting F.")
