"""The two invariance gates, built by hand from the jobs layer.

    python 06_acceptance_gates.py solid      # k-invariance  (crystal)
    python 06_acceptance_gates.py liquid     # p-invariance  (melt)

Both gates work the same way: change something that the free energy must not
depend on, and check that it does not move. They are worth their cost because
each one has caught a real error -- springs so soft the crystal drifted off
its Einstein reference, and a UF coupling outside the calibrated range.

This is also the example to copy when you want a flow the ready-made ones do
not give you: the expensive equilibration runs ONCE and both variants branch
off it, which the stage makers support directly and the flow makers do not.
"""
from __future__ import annotations

import sys

from jobflow import Flow, run_locally
from py_oats.structure_generator.generator import get_amorphous_structure
from pymatgen.core import Composition, Lattice, Structure

from common import output_of, use_reference
from finite_temp_properties.workflow.jobs import (
    CrystalMSDMaker,
    FrenkelLaddMaker,
    MeltEquilibrationMaker,
    UFMSwitchLeg1Maker,
    UFMSwitchLeg2Maker,
    analyze_liquid_free_energy,
    analyze_solid_free_energy,
)

which = sys.argv[1] if len(sys.argv) > 1 else "solid"
TOLERANCE = 0.005          # eV/atom; the gate passes if F moves less than this


def k_invariance(t0=950.0):
    """Rerun Frenkel-Ladd at 4x the springs with the timestep halved.

    A stiffer Einstein crystal is a different reference and a different
    switching path, so W changes a lot -- but F = F_einstein - W must not.
    The timestep must come down with the springs, or the stiff oscillator is
    integrated badly and the gate fails for a reason that is not physics.
    """
    cell = Structure(
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    cell.make_supercell(3)

    msd = CrystalMSDMaker(settings={
        "temperature": t0, "n_equil": 5000, "n_msd": 10000})
    use_reference(msd)
    msd_job = msd.make(cell)

    jobs, docs = [msd_job], {}
    for label, scale, dt in [("nominal", 1.0, 0.001), ("4x springs", 4.0, 0.0005)]:
        fl = FrenkelLaddMaker(settings={
            "temperature": t0, "n_equil": 5000, "n_switch": 15000,
            "spring_scale": scale, "time_step": dt})
        use_reference(fl)
        fl_job = fl.make_from(msd_job.output.dir_name)
        fl_job.name = f"frenkel_ladd {label}"
        doc = analyze_solid_free_energy(
            msd_job.output.dir_name, fl_job.output.dir_name,
            temperature=t0, spring_scale=scale)
        doc.name = f"analyze {label}"
        jobs += [fl_job, doc]
        docs[label] = doc.name
    return Flow(jobs, name="k_invariance"), docs


def p_invariance(ta=2900.0):
    """Rerun both UFM legs at p = 25 instead of 50.

    p sets the stiffness of the reference fluid, eps = p kB T. It picks the
    path, not the endpoint, so F(Ta) must not depend on it. Both legs move
    together: leg 1 switches INTO the p-fluid and leg 2 switches within it.
    """
    melt = get_amorphous_structure(Composition("SrTiO3"))

    eq = MeltEquilibrationMaker(settings={
        "temperature": ta, "n_equil": 8000, "n_sample": 20000,
        "dump_interval": 100})
    use_reference(eq)
    melt_job = eq.make(melt)

    jobs, docs = [melt_job], {}
    for p in (50.0, 25.0):
        common = {"temperature": ta, "n_equil": 8000, "n_switch": 15000,
                  "ufm_p": p}
        leg1 = UFMSwitchLeg1Maker(settings=dict(common))
        leg2 = UFMSwitchLeg2Maker(settings=dict(common))
        use_reference(leg1)
        use_reference(leg2)

        leg1_job = leg1.make_from(melt_job.output.dir_name)
        leg2_job = leg2.make_from(melt_job.output.dir_name,
                                  leg1_job.output.dir_name)
        leg1_job.name, leg2_job.name = f"leg1 p={p:.0f}", f"leg2 p={p:.0f}"
        doc = analyze_liquid_free_energy(
            melt_job.output.dir_name, leg1_job.output.dir_name,
            leg2_job.output.dir_name, temperature=ta, ufm_p=p,
            time_step=eq.settings["time_step"],
            dump_interval=eq.settings["dump_interval"])
        doc.name = f"analyze p={p:.0f}"
        jobs += [leg1_job, leg2_job, doc]
        docs[f"p = {p:.0f}"] = doc.name
    return Flow(jobs, name="p_invariance"), docs


flow, doc_names = k_invariance() if which == "solid" else p_invariance()
responses = run_locally(flow, create_folders=True)

print(f"\n{flow.name}")
values = {}
for label, job_name in doc_names.items():
    doc = output_of(flow, responses, job_name)
    values[label] = doc.free_energy
    print(f"  {label:12s} F = {doc.free_energy:12.6f} eV/atom "
          f"(hysteresis {doc.hysteresis:.6f})")

spread = max(values.values()) - min(values.values())
verdict = "PASS" if spread < TOLERANCE else "FAIL"
print(f"\n  spread {1000 * spread:.2f} meV/atom -> {verdict} "
      f"(tolerance {1000 * TOLERANCE:.0f} meV/atom)")
if verdict == "FAIL":
    print("  F depends on something it must not. Do not use these numbers; "
          "switch more slowly and check the reference first.")
