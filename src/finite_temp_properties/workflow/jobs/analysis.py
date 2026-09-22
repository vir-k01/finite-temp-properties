"""Jobs that read what the LAMMPS stages wrote and return a document.

No MD here: these are pure post-processing, so they run anywhere and are the
easy part to test. The physics they apply -- the reference free energies, the
switching-work integrals, the two G(T) formulas -- lives in ``utils.helpers``.
"""
from __future__ import annotations

from jobflow import job

from finite_temp_properties.schemas.free_energy import (
    GibbsCurveDoc,
    LiquidFreeEnergyDoc,
    PotentialSwitchDoc,
    SolidFreeEnergyDoc,
)
from finite_temp_properties.utils import helpers as h

__all__ = [
    "analyze_solid_free_energy",
    "analyze_liquid_free_energy",
    "analyze_potential_switch",
    "build_gibbs_curve",
]


@job
def analyze_solid_free_energy(msd_dir: str, fl_dir: str, temperature: float,
                              spring_scale: float = 1.0) -> SolidFreeEnergyDoc:
    """F(T0) = F_einstein - W from the MSD and Frenkel-Ladd runs."""
    structure = h.structure_from(msd_dir, "equil.data")
    species = h.species_of(structure)
    k, vol_per_atom = h.spring_constants_from_msd(
        msd_dir, temperature, len(species))
    k = [spring_scale * ki for ki in k]
    work, hysteresis = h.frenkel_ladd_work(fl_dir)
    counts = h.atom_counts(msd_dir, "equil.data", len(species))
    einstein = h.einstein_crystal_fe(temperature, k, species, counts,
                                     vol_per_atom)
    return SolidFreeEnergyDoc(
        temperature=temperature, species=species,
        free_energy=einstein - work, einstein_reference=einstein,
        work=work, hysteresis=hysteresis, spring_constants=k,
        volume_per_atom=vol_per_atom, natoms=sum(counts),
        msd_dir=str(msd_dir), frenkel_ladd_dir=str(fl_dir))


@job
def analyze_liquid_free_energy(melt_dir: str, leg1_dir: str, leg2_dir: str,
                               temperature: float, ufm_p: float = 50.0,
                               time_step: float = 0.001,
                               dump_interval: int = 100) -> LiquidFreeEnergyDoc:
    """F(Ta) = F_UF + F_ideal_gas - W_leg2 - W_leg1 from the three melt runs.

    The melt trajectory also supplies the diffusivity gate: if the melt is not
    diffusing at the anchor, its H(T) branch is a glass branch and the
    Gibbs-Helmholtz descent below is meaningless.
    """
    structure = h.structure_from(melt_dir, "liq.data")
    species = h.species_of(structure)
    vol_per_atom, enthalpy_cell = h.volume_and_enthalpy(melt_dir)
    trajectory = h.read_trajectory(melt_dir, "melt.dump", temperature,
                                   time_step, dump_interval)
    sigmas = h.contact_sigmas(trajectory, species)
    diffusivity = h.diffusivities(trajectory)
    w1, hyst1 = h.forward_backward_work(leg1_dir)
    w2, hyst2 = h.forward_backward_work(leg2_dir)
    counts = h.atom_counts(melt_dir, "liq.data", len(species))
    reference = h.uf_reference_fe(temperature, vol_per_atom, ufm_p, species,
                                  counts)
    return LiquidFreeEnergyDoc(
        temperature=temperature, species=species,
        free_energy=reference - w2 - w1, reference_free_energy=reference,
        work_leg1=w1, work_leg2=w2, hysteresis=hyst1 + hyst2, ufm_p=ufm_p,
        sigmas=sigmas, sigma_single=h.sigma_for_target_x(vol_per_atom),
        diffusivity=diffusivity,
        enthalpy_per_atom=enthalpy_cell / sum(counts),
        volume_per_atom=vol_per_atom, natoms=sum(counts),
        melt_dir=str(melt_dir), leg1_dir=str(leg1_dir), leg2_dir=str(leg2_dir))


@job
def analyze_potential_switch(switch_dir: str,
                             temperature: float) -> PotentialSwitchDoc:
    """dG(reference -> target) and the endpoint volumes of one switch."""
    dg, hysteresis = h.forward_backward_work(switch_dir)
    v_ref, v_tgt = h.endpoint_volumes(switch_dir)
    return PotentialSwitchDoc(
        temperature=temperature, dG=dg, hysteresis=hysteresis,
        volume_reference=v_ref, volume_target=v_tgt,
        dir_name=str(switch_dir))


@job
def build_gibbs_curve(
    temperatures: list[float],
    solid: SolidFreeEnergyDoc | None = None,
    solid_switch: PotentialSwitchDoc | None = None,
    crystal_ht_dir: str | None = None,
    liquid: LiquidFreeEnergyDoc | None = None,
    liquid_switch: PotentialSwitchDoc | None = None,
    quench_dir: str | None = None,
) -> GibbsCurveDoc:
    """Assemble G(T) from the TI anchors and the target-potential H(T) MD.

    crystal:  G0 = F_TI + dG_switch, S(T0) = (H(T0) - G0)/T0, harmonic carry.
    amorphous: Ga = F_TI + dG_switch, Gibbs-Helmholtz descent over the
    binned quench H(T). Either branch may be omitted.
    """
    import numpy as np

    ts = np.asarray(temperatures, dtype=float)
    doc = GibbsCurveDoc(temperatures=list(ts), solid=solid, liquid=liquid,
                        solid_switch=solid_switch, liquid_switch=liquid_switch)

    if solid is not None and crystal_ht_dir is not None:
        structure = h.structure_from(crystal_ht_dir, "final.data")
        t_measured, h_measured = h.mean_ht(crystal_ht_dir, len(structure))
        t0 = solid.temperature
        # ht.dat reports the MEASURED mean T, a few K off the setpoint; carry
        # H the short distance to T0 so F and H refer to the same temperature.
        h0 = h_measured + h.THREE_R * (t0 - t_measured)
        g0 = solid.free_energy + (solid_switch.dG if solid_switch else 0.0)
        doc.t_crystal_anchor, doc.h_crystal_anchor, doc.g_crystal_anchor = t0, h0, g0
        doc.s_crystal_anchor = (h0 - g0) / t0 * h.EV_ATOM_TO_J_MOL
        doc.g_crystal = list(h.crystal_gibbs_curve(ts, t0, h0, g0))

    if liquid is not None and quench_dir is not None:
        structure = h.structure_from(quench_dir, "final.data")
        ta = liquid.temperature
        quench_T, quench_H = h.binned_quench_ht(quench_dir, len(structure), ta)
        ga = liquid.free_energy + (liquid_switch.dG if liquid_switch else 0.0)
        h_at_anchor = float(np.interp(ta, quench_T, quench_H))
        doc.t_melt_anchor, doc.g_melt_anchor = ta, ga
        doc.s_melt_anchor = (h_at_anchor - ga) / ta * h.EV_ATOM_TO_J_MOL
        doc.g_amorphous = list(h.amorphous_gibbs_curve(ts, ta, ga, quench_T,
                                                       quench_H))

    return doc
