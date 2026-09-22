"""One maker per LAMMPS stage.

Same logic as py-OATS: every stage is a CustomLammpsMaker whose ``settings``
are merged on top of module defaults, and the template is filled from those
settings. A stage that needs numbers measured by a PREVIOUS run (spring
constants, sigmas) has a second entry point, ``make_from(<previous dir>)``:
one job whose body reads those files at runtime, configures the input, and
then runs LAMMPS -- jobflow resolves the directory reference when the job
starts, so no separate resolver/replace job is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pymatgen.core import Structure

from atomate2.lammps.jobs.base import BaseLammpsMaker, lammps_job
from atomate2.lammps.jobs.core import CustomLammpsMaker

from finite_temp_properties.utils import helpers as h
from finite_temp_properties.utils.helpers import TEMPLATE_DIR

__all__ = [
    "CrystalMSDMaker",
    "FrenkelLaddMaker",
    "MeltEquilibrationMaker",
    "UFMSwitchLeg1Maker",
    "UFMSwitchLeg2Maker",
    "PotentialSwitchMaker",
    "CrystalEnthalpyMaker",
    "QuenchMaker",
]


def _template(name: str) -> Path:
    return TEMPLATE_DIR / name


# ------------------------------------------------------------------- crystal

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


# -------------------------------------------------------------------- liquid

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


# ----------------------------------------------- target-potential MD (H, dG)

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
