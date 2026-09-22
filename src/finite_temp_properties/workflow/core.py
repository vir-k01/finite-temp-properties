"""Makers and flows for finite-temperature free energies.

Same logic as py-OATS: every LAMMPS stage is a CustomLammpsMaker whose
``settings`` are merged on top of module defaults and templates are filled
from settings. A stage that needs numbers measured by a PREVIOUS job (spring
constants, sigmas) has a second entry point, ``make_from(<previous dir>)``:
one job whose body reads those files at runtime, configures the input, and
runs LAMMPS -- jobflow resolves the directory reference when the job starts,
so no separate resolver/replace job is needed.

The flows this module is built around:

    SolidFreeEnergyMaker.make(structure)   -> F_crystal(T0)
    LiquidFreeEnergyMaker.make(structure)  -> F_melt(Ta)
    GibbsCurveMaker.make(crystal, melt)    -> G_crystal(T) and G_amorphous(T)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, job
from pymatgen.core import Composition, Structure

from atomate2.lammps.jobs.base import BaseLammpsMaker, lammps_job
from atomate2.lammps.jobs.core import CustomLammpsMaker

from finite_temp_properties.schemas.free_energy import (
    GibbsCurveDoc,
    LiquidFreeEnergyDoc,
    PotentialSwitchDoc,
    SolidFreeEnergyDoc,
)
from finite_temp_properties.utils import helpers as h
from finite_temp_properties.utils.helpers import TEMPLATE_DIR


def _template(name: str) -> Path:
    return TEMPLATE_DIR / name


# ---------------------------------------------------------------------------
# stage makers, one per template
# ---------------------------------------------------------------------------

@dataclass
class CrystalMSDMaker(CustomLammpsMaker):
    """Equilibrate a crystal and measure the per-species MSD.

    Fixes the Einstein reference for the Frenkel-Ladd switch:
    k_i = 3 kB T / <dr^2>_i. Writes msd.dat and equil.data.
    """

    name: str = "crystal_msd"
    inputfile: str | Path = field(default_factory=lambda: _template("crystal_msd.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.CRYSTAL_MSD_DEFAULTS, **self.settings}
        super().__post_init__()

    def make(self, structure: Structure, **kwargs):
        self.input_set_generator.update_settings(
            h.species_settings(structure) | h.msd_blocks(structure.n_elems),
            validate_params=False)
        return super().make(input_structure=structure, **kwargs)


@dataclass
class FrenkelLaddMaker(CustomLammpsMaker):
    """Switch the crystal to an Einstein crystal and back (fix ti/spring).

    The spring constants come from a CrystalMSDMaker run: chain with
    ``make_from(msd_dir)``, which reads them when the job starts. Writes
    ti.dat (forward sweep, then backward).
    """

    name: str = "frenkel_ladd"
    inputfile: str | Path = field(default_factory=lambda: _template("frenkel_ladd.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.FRENKEL_LADD_DEFAULTS, **self.settings}
        super().__post_init__()

    def _configure(self, structure: Structure, spring_constants: list[float]):
        s = self.settings
        k = [s["spring_scale"] * ki for ki in spring_constants]
        self.input_set_generator.update_settings(
            h.species_settings(structure)
            | h.spring_blocks(k, s["n_switch"], s["n_equil"]),
            validate_params=False)

    def make(self, structure: Structure, spring_constants: list[float], **kwargs):
        self._configure(structure, spring_constants)
        return super().make(input_structure=structure, **kwargs)

    @lammps_job
    def make_from(self, msd_dir: str):
        """One job: read k_i = 3kBT/<dr^2>_i from the MSD run, then switch."""
        structure = h.structure_from(str(msd_dir), "equil.data")
        k, _ = h.spring_constants_from_msd(
            str(msd_dir), self.settings["temperature"], structure.n_elems)
        self._configure(structure, k)
        return BaseLammpsMaker.make.original(self, input_structure=structure)


@dataclass
class MeltEquilibrationMaker(CustomLammpsMaker):
    """Equilibrate a melt at the anchor and record a trajectory + density.

    Fixes the Uhlenbeck-Ford reference (per-pair contact sigmas, from the
    trajectory via py-OATS's CoordinationAnalyzer) and supplies the melt
    diffusivity gate. Writes melt.dump, vol.dat, liq.data.
    """

    name: str = "melt_equilibration"
    inputfile: str | Path = field(
        default_factory=lambda: _template("melt_equilibration.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.MELT_EQUILIBRATION_DEFAULTS, **self.settings}
        super().__post_init__()

    def make(self, structure: Structure, **kwargs):
        self.input_set_generator.update_settings(
            h.species_settings(structure), validate_params=False)
        return super().make(input_structure=structure, **kwargs)


@dataclass
class UFMSwitchLeg1Maker(CustomLammpsMaker):
    """Switch the real melt to a multi-sigma Uhlenbeck-Ford fluid.

    Sigmas come from the melt run's RDFs: chain with ``make_from(melt_dir)``.
    Writes fwd.dat, bwd.dat and the endpoint cell ufm_endpoint.data.
    """

    name: str = "ufm_switch_leg1"
    inputfile: str | Path = field(
        default_factory=lambda: _template("ufm_switch_leg1.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.UFM_SWITCH_DEFAULTS, **self.settings}
        super().__post_init__()

    def _configure(self, structure: Structure, sigmas: list[float]):
        s = self.settings
        eps = s["ufm_p"] * h.KB * s["temperature"]
        self.input_set_generator.update_settings(
            h.species_settings(structure) | {
                "ufm_cutoff": 5 * max(sigmas),
                "ufm_pair_coeffs": h.ufm_coeff_block(
                    eps, sigmas, structure.n_elems),
            },
            validate_params=False)

    def make(self, structure: Structure, sigmas: list[float], **kwargs):
        self._configure(structure, sigmas)
        return super().make(input_structure=structure, **kwargs)

    @lammps_job
    def make_from(self, melt_dir: str):
        """One job: sigmas from the melt's partial RDFs, then leg 1."""
        structure = h.structure_from(str(melt_dir), "liq.data")
        sigmas = h.sigmas_from_melt(str(melt_dir), h.species_of(structure),
                                    self.settings["temperature"])
        self._configure(structure, sigmas)
        return BaseLammpsMaker.make.original(self, input_structure=structure)


@dataclass
class UFMSwitchLeg2Maker(CustomLammpsMaker):
    """Switch the multi-sigma UFM to the single-sigma fluid whose free energy
    is calibrated. Chain with ``make_from(melt_dir, leg1_dir)``. Rerunning at
    ufm_p = 25 is the p-invariance test.
    """

    name: str = "ufm_switch_leg2"
    inputfile: str | Path = field(
        default_factory=lambda: _template("ufm_switch_leg2.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.UFM_SWITCH_DEFAULTS, **self.settings}
        super().__post_init__()

    def _configure(self, structure: Structure, sigmas: list[float],
                   sigma_single: float):
        s = self.settings
        n = structure.n_elems
        eps = s["ufm_p"] * h.KB * s["temperature"]
        self.input_set_generator.update_settings(
            h.species_settings(structure) | {
                "ufm_cutoff_multi": 5 * max(sigmas),
                "ufm_cutoff_single": 5 * sigma_single,
                "ufm_pair_coeffs_multi": h.ufm_coeff_block(
                    eps, sigmas, n, "ufm 1"),
                "ufm_pair_coeffs_single": h.ufm_coeff_block(
                    eps, [sigma_single] * len(h.type_pairs(n)), n, "ufm 2"),
            },
            validate_params=False)

    def make(self, structure: Structure, sigmas: list[float],
             sigma_single: float, **kwargs):
        self._configure(structure, sigmas, sigma_single)
        return super().make(input_structure=structure, **kwargs)

    @lammps_job
    def make_from(self, melt_dir: str, leg1_dir: str):
        """One job: sigma0 from the melt density, start from leg 1's endpoint."""
        structure = h.structure_from(str(leg1_dir), "ufm_endpoint.data")
        sigmas = h.sigmas_from_melt(str(melt_dir), h.species_of(structure),
                                    self.settings["temperature"])
        vol_per_atom, _ = h.volume_and_enthalpy(str(melt_dir))
        self._configure(structure, sigmas, h.sigma_for_target_x(vol_per_atom))
        return BaseLammpsMaker.make.original(self, input_structure=structure)


@dataclass
class PotentialSwitchMaker(CustomLammpsMaker):
    """Transport a phase from the reference to the target potential (NPT).

    Chain with ``make_from(source_dir, source_file)`` so it starts from the
    exact cell the TI legs used (equil.data for a crystal, liq.data for a
    melt). Needs the TARGET engine (e.g. the TF build) -- set
    run_lammps_kwargs accordingly.
    """

    name: str = "potential_switch"
    inputfile: str | Path = field(
        default_factory=lambda: _template("potential_switch.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.POTENTIAL_SWITCH_DEFAULTS, **self.settings}
        super().__post_init__()

    def _configure(self, structure: Structure):
        self.input_set_generator.update_settings(
            h.species_settings(structure)
            | {"maybe_triclinic": h.triclinic_setting(structure)
               if self.settings["barostat"] == "tri"
               else "### orthogonal barostat"},
            validate_params=False)

    def make(self, structure: Structure, **kwargs):
        self._configure(structure)
        return super().make(input_structure=structure, **kwargs)

    @lammps_job
    def make_from(self, source_dir: str, source_file: str):
        """One job: load the TI cell, then switch reference -> target."""
        structure = h.structure_from(str(source_dir), source_file)
        self._configure(structure)
        return BaseLammpsMaker.make.original(self, input_structure=structure)


@dataclass
class CrystalEnthalpyMaker(CustomLammpsMaker):
    """Crystal H(T0) under the target potential -> ht.dat."""

    name: str = "crystal_enthalpy"
    inputfile: str | Path = field(
        default_factory=lambda: _template("crystal_enthalpy.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.CRYSTAL_ENTHALPY_DEFAULTS, **self.settings}
        super().__post_init__()

    def make(self, structure: Structure, **kwargs):
        self.input_set_generator.update_settings(
            h.species_settings(structure)
            | {"maybe_triclinic": h.triclinic_setting(structure)
               if self.settings["barostat"] == "tri"
               else "### orthogonal barostat"},
            validate_params=False)
        return super().make(input_structure=structure, **kwargs)


@dataclass
class QuenchMaker(CustomLammpsMaker):
    """H(T) along an NPT quench under the target potential -> ht_leg*.dat.

    The Gibbs-Helmholtz integrand. Legs are generated from (temperature,
    t_floor, quench_rate, n_legs) in the settings.
    """

    name: str = "quench_enthalpy"
    inputfile: str | Path = field(
        default_factory=lambda: _template("quench_enthalpy.in"))
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.settings = {**h.QUENCH_DEFAULTS, **self.settings}
        super().__post_init__()

    def make(self, structure: Structure, **kwargs):
        s = self.settings
        self.input_set_generator.update_settings(
            h.species_settings(structure) | {
                "quench_legs": h.quench_leg_block(
                    s["temperature"], s["t_floor"], s["quench_rate"],
                    s["n_legs"], s["time_step"]),
            },
            validate_params=False)
        return super().make(input_structure=structure, **kwargs)


# ---------------------------------------------------------------------------
# analysis jobs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# flow makers -- what you build off of
# ---------------------------------------------------------------------------

@dataclass
class SolidFreeEnergyMaker(Maker):
    """F_crystal(T0), optionally transported to a target potential.

    Flow: CrystalMSDMaker -> FrenkelLaddMaker -> analysis
          (-> PotentialSwitchMaker -> switch analysis when with_switch=True).
    Output: (SolidFreeEnergyDoc, PotentialSwitchDoc | None).
    """

    name: str = "solid_free_energy"
    msd_maker: CrystalMSDMaker = field(default_factory=CrystalMSDMaker)
    frenkel_ladd_maker: FrenkelLaddMaker = field(default_factory=FrenkelLaddMaker)
    switch_maker: PotentialSwitchMaker = field(
        default_factory=lambda: PotentialSwitchMaker(
            settings={"barostat": "tri"}))
    with_switch: bool = True

    def __post_init__(self):
        # one temperature for every stage, set on the MSD maker
        t = self.msd_maker.settings["temperature"]
        self.frenkel_ladd_maker.settings["temperature"] = t
        self.switch_maker.settings["temperature"] = t

    def make(self, structure: Structure) -> Flow:
        msd = self.msd_maker.make(structure)
        fl = self.frenkel_ladd_maker.make_from(msd.output.dir_name)
        doc = analyze_solid_free_energy(
            msd.output.dir_name, fl.output.dir_name,
            temperature=self.msd_maker.settings["temperature"],
            spring_scale=self.frenkel_ladd_maker.settings["spring_scale"])
        jobs = [msd, fl, doc]
        switch_doc = None
        if self.with_switch:
            sw = self.switch_maker.make_from(msd.output.dir_name, "equil.data")
            switch_doc = analyze_potential_switch(
                sw.output.dir_name, self.switch_maker.settings["temperature"])
            jobs += [sw, switch_doc]
        return Flow(jobs, output=(doc.output,
                                  switch_doc.output if switch_doc else None),
                    name=self.name)


@dataclass
class LiquidFreeEnergyMaker(Maker):
    """F_melt(Ta), optionally transported to a target potential.

    Flow: MeltEquilibrationMaker -> UFM leg 1 -> UFM leg 2 -> analysis
          (-> PotentialSwitchMaker -> switch analysis when with_switch=True).
    Output: (LiquidFreeEnergyDoc, PotentialSwitchDoc | None).
    """

    name: str = "liquid_free_energy"
    melt_maker: MeltEquilibrationMaker = field(
        default_factory=MeltEquilibrationMaker)
    leg1_maker: UFMSwitchLeg1Maker = field(default_factory=UFMSwitchLeg1Maker)
    leg2_maker: UFMSwitchLeg2Maker = field(default_factory=UFMSwitchLeg2Maker)
    switch_maker: PotentialSwitchMaker = field(
        default_factory=lambda: PotentialSwitchMaker(
            settings={"barostat": "iso"}))
    with_switch: bool = True

    def __post_init__(self):
        t = self.melt_maker.settings["temperature"]
        self.leg1_maker.settings["temperature"] = t
        self.leg2_maker.settings["temperature"] = t
        self.switch_maker.settings["temperature"] = t

    def make(self, structure: Structure) -> Flow:
        melt = self.melt_maker.make(structure)
        leg1 = self.leg1_maker.make_from(melt.output.dir_name)
        leg2 = self.leg2_maker.make_from(melt.output.dir_name,
                                         leg1.output.dir_name)
        doc = analyze_liquid_free_energy(
            melt.output.dir_name, leg1.output.dir_name, leg2.output.dir_name,
            temperature=self.melt_maker.settings["temperature"],
            ufm_p=self.leg2_maker.settings["ufm_p"],
            time_step=self.melt_maker.settings["time_step"],
            dump_interval=self.melt_maker.settings["dump_interval"])
        jobs = [melt, leg1, leg2, doc]
        switch_doc = None
        if self.with_switch:
            sw = self.switch_maker.make_from(melt.output.dir_name, "liq.data")
            switch_doc = analyze_potential_switch(
                sw.output.dir_name, self.switch_maker.settings["temperature"])
            jobs += [sw, switch_doc]
        return Flow(jobs, output=(doc.output,
                                  switch_doc.output if switch_doc else None),
                    name=self.name)


@dataclass
class GibbsCurveMaker(Maker):
    """G(T) of a crystal and its melt on one grid, under the target potential.

    Runs both free-energy flows, the two target-potential H(T) runs the G(T)
    formulas need (crystal H(T0), quench H(T)), and assembles a GibbsCurveDoc.

    A melt given as a Composition is packed into an amorphous cell with
    py-OATS's structure generator (at build time); a Structure is used as-is.

    ``temperature_crystal`` (the anchor T0) and ``temperature_melt`` (the
    anchor Ta) are pushed into every stage they govern; leave them None to
    keep whatever the sub-makers were built with.
    """

    name: str = "gibbs_curve"
    solid_maker: SolidFreeEnergyMaker = field(
        default_factory=SolidFreeEnergyMaker)
    liquid_maker: LiquidFreeEnergyMaker = field(
        default_factory=LiquidFreeEnergyMaker)
    crystal_enthalpy_maker: CrystalEnthalpyMaker = field(
        default_factory=CrystalEnthalpyMaker)
    quench_maker: QuenchMaker = field(default_factory=QuenchMaker)
    temperatures: list[float] = field(
        default_factory=lambda: list(range(700, 1201, 25)))
    temperature_crystal: float | None = None
    temperature_melt: float | None = None

    def __post_init__(self):
        if self.temperature_crystal is not None:
            for maker in (self.solid_maker.msd_maker,
                          self.solid_maker.frenkel_ladd_maker,
                          self.solid_maker.switch_maker):
                maker.settings["temperature"] = self.temperature_crystal
        if self.temperature_melt is not None:
            for maker in (self.liquid_maker.melt_maker,
                          self.liquid_maker.leg1_maker,
                          self.liquid_maker.leg2_maker,
                          self.liquid_maker.switch_maker):
                maker.settings["temperature"] = self.temperature_melt
        # the H(T) runs must sit at the same anchors as the TI
        self.crystal_enthalpy_maker.settings["temperature"] = \
            self.solid_maker.msd_maker.settings["temperature"]
        self.quench_maker.settings["temperature"] = \
            self.liquid_maker.melt_maker.settings["temperature"]

    def make(self, crystal: Structure,
             melt: Structure | Composition) -> Flow:
        if isinstance(melt, Composition):
            from py_oats.structure_generator.generator import (
                get_amorphous_structure,
            )
            melt = get_amorphous_structure(melt)

        solid_flow = self.solid_maker.make(crystal)
        liquid_flow = self.liquid_maker.make(melt)
        crystal_ht = self.crystal_enthalpy_maker.make(crystal)
        quench = self.quench_maker.make(melt)

        solid_doc, solid_switch = solid_flow.output
        liquid_doc, liquid_switch = liquid_flow.output
        curve = build_gibbs_curve(
            temperatures=self.temperatures,
            solid=solid_doc, solid_switch=solid_switch,
            crystal_ht_dir=crystal_ht.output.dir_name,
            liquid=liquid_doc, liquid_switch=liquid_switch,
            quench_dir=quench.output.dir_name)
        return Flow([solid_flow, liquid_flow, crystal_ht, quench, curve],
                    output=curve.output, name=self.name)
