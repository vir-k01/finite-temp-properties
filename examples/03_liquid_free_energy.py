"""The melt branch alone: F(Ta) by the two-leg Uhlenbeck-Ford path.

    salloc -N 1 -C cpu -q interactive -t 4:00:00
    python 03_liquid_free_energy.py

The melt cell is packed from a composition by py-OATS. The anchor Ta should
sit well above the liquidus -- roughly 1.3x it -- because everything below is
reached by cooling from here, and a melt that is already arrested at the
anchor turns the whole G(T) branch into a glass branch. That is what the
diffusivity printed at the end is for.
"""
from __future__ import annotations

from jobflow import run_locally
from py_oats.structure_generator.generator import get_amorphous_structure
from pymatgen.core import Composition

from common import configure_liquid, output_of
from finite_temp_properties.workflow.flows import LiquidFreeEnergyMaker
from finite_temp_properties.workflow.jobs import (
    MeltEquilibrationMaker,
    UFMSwitchLeg1Maker,
    UFMSwitchLeg2Maker,
)

TA = 2900.0          # melt anchor

melt = get_amorphous_structure(Composition("SrTiO3"))
print(f"packed melt: {melt.composition.reduced_formula}, {len(melt)} atoms, "
      f"{melt.volume / len(melt):.3f} A^3/atom")

maker = LiquidFreeEnergyMaker(
    # Short settings for an interactive allocation; production uses the
    # defaults (20k equil / 40k sample, 40k per switching sweep).
    melt_maker=MeltEquilibrationMaker(settings={
        "temperature": TA, "n_equil": 8000, "n_sample": 20000,
        "dump_interval": 100}),
    leg1_maker=UFMSwitchLeg1Maker(settings={
        "temperature": TA, "n_equil": 8000, "n_switch": 15000}),
    leg2_maker=UFMSwitchLeg2Maker(settings={
        "temperature": TA, "n_equil": 8000, "n_switch": 15000}),
    with_switch=False,          # reference potential only; no GPU needed
)
configure_liquid(maker)

flow = maker.make(melt)
responses = run_locally(flow, create_folders=True)
doc = output_of(flow, responses, "analyze_liquid_free_energy")

print(f"""
melt free energy at {doc.temperature:.0f} K
  F              {doc.free_energy:12.6f} eV/atom
  UF + ideal gas {doc.reference_free_energy:12.6f} eV/atom
  W leg 1        {doc.work_leg1:12.6f} eV/atom   (real melt -> multi-sigma UFM)
  W leg 2        {doc.work_leg2:12.6f} eV/atom   (multi-sigma -> single-sigma)
  hysteresis     {doc.hysteresis:12.6f} eV/atom  <- both legs summed
  sigmas         {['%.3f' % s for s in doc.sigmas]} A
  sigma_single   {doc.sigma_single:.4f} A   (places x on the calibrated point)
  H(Ta)          {doc.enthalpy_per_atom:12.6f} eV/atom
""")

# The gate. A real oxide melt at 1.3x its liquidus runs 1e-5 - 1e-3 cm^2/s.
print("diffusivity at the anchor:")
for element, d in (doc.diffusivity or {}).items():
    flag = "  <- ARRESTED" if d < 1e-6 else ""
    print(f"  D({element}) = {d:.3e} cm^2/s{flag}")
if doc.diffusivity and min(doc.diffusivity.values()) < 1e-6:
    print("\nAt least one species is not diffusing: this is a glass, not a "
          "melt. Raise the anchor temperature and rerun before using F(Ta).")
