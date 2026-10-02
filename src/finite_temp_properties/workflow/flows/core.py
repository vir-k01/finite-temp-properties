"""The flows you build off of.

    SolidFreeEnergyMaker.make(structure)   -> F_crystal(T0)
    LiquidFreeEnergyMaker.make(structure)  -> F_melt(Ta)
    GibbsCurveMaker.make(crystal, melt)    -> G_crystal(T) and G_amorphous(T)

Each one wires stage makers from ``workflow.jobs.makers`` to the analysis jobs
in ``workflow.jobs.analysis``. Swap any stage maker out through its field --
that is where per-stage settings and the LAMMPS engine (run_lammps_kwargs)
are set.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from jobflow import Flow, Maker
from pymatgen.core import Composition, Structure

from finite_temp_properties.utils import helpers as h
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

__all__ = ["SolidFreeEnergyMaker", "LiquidFreeEnergyMaker", "GibbsCurveMaker"]


def _set_temperature(maker, temperature: float) -> None:
    """Set a stage's temperature AFTER it was built.

    A CustomLammpsMaker copies ``settings`` into its input-set generator in
    __post_init__, so editing ``maker.settings`` alone changes what the
    analysis reads but NOT what LAMMPS runs. Update both.
    """
    maker.settings["temperature"] = temperature
    maker.input_set_generator.update_settings({"temperature": temperature},
                                              validate_params=False)


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
        _set_temperature(self.frenkel_ladd_maker, t)
        _set_temperature(self.switch_maker, t)

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
        _set_temperature(self.leg1_maker, t)
        _set_temperature(self.leg2_maker, t)
        _set_temperature(self.switch_maker, t)

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

    Runs both free-energy flows, the target-potential H(T) runs the G(T)
    formulas need, and assembles a GibbsCurveDoc:

    - crystal: one NPT H run per ``crystal_temperatures`` (independent jobs).
      G(T) is integrated by Gibbs-Helmholtz from the TI anchor through the
      measured H(T), so Cp(T) -- anharmonicity, thermal expansion -- comes
      from the MD. None (default) builds a ladder at most 100 K apart over
      the report grid and T0; [] runs H(T0) only and falls back to the legacy
      Cp = 3R carry, which freezes formation enthalpies and entropies at T0.
    - melt: one NPT quench from Ta, Gibbs-Helmholtz descent.

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
    crystal_temperatures: list[float] | None = None

    def __post_init__(self):
        if self.temperature_crystal is not None:
            for maker in (self.solid_maker.msd_maker,
                          self.solid_maker.frenkel_ladd_maker,
                          self.solid_maker.switch_maker):
                _set_temperature(maker, self.temperature_crystal)
        if self.temperature_melt is not None:
            for maker in (self.liquid_maker.melt_maker,
                          self.liquid_maker.leg1_maker,
                          self.liquid_maker.leg2_maker,
                          self.liquid_maker.switch_maker):
                _set_temperature(maker, self.temperature_melt)
        # the H(T) runs must sit at the same anchors as the TI
        t0 = self.solid_maker.msd_maker.settings["temperature"]
        _set_temperature(self.crystal_enthalpy_maker, t0)
        _set_temperature(self.quench_maker,
                         self.liquid_maker.melt_maker.settings["temperature"])
        if self.crystal_temperatures is None:
            self.crystal_temperatures = h.crystal_temperature_ladder(
                self.temperatures, t0)

    def crystal_enthalpy_makers(self) -> list[CrystalEnthalpyMaker]:
        """One NPT H maker per sweep temperature (just T0 for the legacy carry).

        Built with dataclasses.replace so each copy constructs its OWN input
        generator at its own temperature; engine and settings are inherited.
        """
        t0 = self.solid_maker.msd_maker.settings["temperature"]
        temps = sorted(set(self.crystal_temperatures)) or [t0]
        base = self.crystal_enthalpy_maker
        return [replace(base, name=f"{base.name}_{t:.0f}K",
                        settings={**base.settings, "temperature": float(t)})
                for t in temps]

    def make(self, crystal: Structure,
             melt: Structure | Composition) -> Flow:
        if isinstance(melt, Composition):
            from py_oats.structure_generator.generator import (
                get_amorphous_structure,
            )
            melt = get_amorphous_structure(melt)

        solid_flow = self.solid_maker.make(crystal)
        liquid_flow = self.liquid_maker.make(melt)
        crystal_ht = [m.make(crystal) for m in self.crystal_enthalpy_makers()]
        quench = self.quench_maker.make(melt)

        solid_doc, solid_switch = solid_flow.output
        liquid_doc, liquid_switch = liquid_flow.output
        curve = build_gibbs_curve(
            temperatures=self.temperatures,
            solid=solid_doc, solid_switch=solid_switch,
            crystal_ht_dirs=[j.output.dir_name for j in crystal_ht],
            liquid=liquid_doc, liquid_switch=liquid_switch,
            quench_dir=quench.output.dir_name)
        return Flow([solid_flow, liquid_flow, *crystal_ht, quench, curve],
                    output=curve.output, name=self.name)
