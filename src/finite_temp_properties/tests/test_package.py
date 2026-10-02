"""Tests that run anywhere: block builders, parsers on synthetic files,
the G(T) math against analytic results, template rendering, and flow
construction. No LAMMPS, no network."""
from __future__ import annotations

import gzip
import re

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

from finite_temp_properties.utils import helpers as h


@pytest.fixture
def srtio3():
    return Structure(
        Lattice.cubic(8.0), ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


# ---------------------------------------------------------------- species

def test_species_follow_lammps_type_order(srtio3):
    # electronegativity order, the order LammpsData numbers types in --
    # NOT alphabetical (that would be O Sr Ti)
    assert h.species_of(srtio3) == ["Sr", "Ti", "O"]


def test_type_pairs_three_species():
    assert h.type_pairs(3) == [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


# ------------------------------------------------------------ input blocks

def test_msd_blocks():
    b = h.msd_blocks(2)
    assert "group g2 type 2" in b["msd_groups"]
    assert "compute m2 g2 msd com yes" in b["msd_computes"]
    assert b["msd_columns"] == "c_m1[4] c_m2[4]"


def test_spring_blocks_order_and_total():
    b = h.spring_blocks([1.5, 0.3], n_switch=40000, n_equil=15000)
    lines = b["spring_fixes"].splitlines()
    assert lines[0] == "fix ti1 g1 ti/spring 1.500000 40000 15000 function 2"
    assert b["spring_energy_sum"] == "f_ti1+f_ti2"
    assert b["n_total"] == 2 * 15000 + 2 * 40000


def test_ufm_coeff_block_covers_all_pairs():
    block = h.ufm_coeff_block(0.5, [1.0] * 6, 3, "ufm 1")
    assert block.count("pair_coeff") == 6
    assert block.splitlines()[1] == "pair_coeff 1 2 ufm 1 0.500000 1.000000"


def test_quench_legs_span_anchor_to_floor():
    block = h.quench_leg_block(2900.0, 600.0, 10.0, 5, 0.001)
    assert block.count("run 46000") == 5           # 460 K / (10 K/ps) / 1 fs
    assert "temp 2900.0 2440.0" in block and "temp 1060.0 600.0" in block


def test_triclinic_setting(srtio3):
    assert h.triclinic_setting(srtio3) == "change_box all triclinic"
    skewed = Structure(Lattice([[8, 0, 0], [1, 8, 0], [0, 0, 8]]),
                       ["Sr"], [[0, 0, 0]])
    assert h.triclinic_setting(skewed).startswith("###")


# --------------------------------------------------------------- parsers

def test_forward_backward_work_reads_gz(tmp_path):
    lam = np.linspace(0, 1, 101)
    # dU = 2*lambda both ways -> W = 1, hysteresis = 0
    fwd = "l dU\n" + "\n".join(f"{x} {2*x}" for x in lam)
    bwd = "l dU\n" + "\n".join(f"{x} {2*x}" for x in lam[::-1])
    (tmp_path / "fwd.dat").write_text(fwd)
    with gzip.open(tmp_path / "bwd.dat.gz", "wt") as f:
        f.write(bwd)
    W, hyst = h.forward_backward_work(str(tmp_path))
    assert abs(W - 1.0) < 1e-6 and hyst < 1e-9


def test_frenkel_ladd_work_splits_the_two_sweeps(tmp_path):
    lam = np.linspace(0, 1, 101)
    rows = [f"{x} {2*x}" for x in lam] + [f"{x} {2*x}" for x in lam[::-1]]
    (tmp_path / "ti.dat").write_text("lambda dU\n" + "\n".join(rows))
    W, hyst = h.frenkel_ladd_work(str(tmp_path))
    assert abs(W - 1.0) < 1e-6 and hyst < 1e-9


def test_spring_constants_from_msd(tmp_path):
    # constant <dr^2>: 0.02 and 0.05 A^2; vol/atom 12.0
    rows = "\n".join(f"{i} 0.02 0.05 12.0" for i in range(100))
    (tmp_path / "msd.dat").write_text("# a\n# b\n" + rows)
    k, vpa = h.spring_constants_from_msd(str(tmp_path), 1000.0, 2)
    assert abs(k[0] - 3 * h.KB * 1000.0 / 0.02) < 1e-9
    assert vpa == 12.0


def test_contact_sigmas_is_half_the_first_crossing(monkeypatch):
    """sigma = half the r where g(r) first exceeds 0.5, per pair, in
    type_pairs order. The RDF itself is py-OATS's; stub it so the test stays
    a unit test of the sigma rule."""
    import types

    r = np.linspace(0.0, 8.0, 401)

    class _Result:
        def __init__(self, first_above):
            self.r = r
            self.rdf = np.where(r >= first_above, 1.0, 0.0)

    crossings = {("Sr", "Sr"): 3.2, ("Sr", "O"): 2.0, ("O", "O"): 2.4}

    class _Analyzer:
        def __init__(self, *a, **k):
            pass

        def analyze(self):
            pass

        def get_rdf(self, s1, s2):
            return _Result(crossings[(s1, s2)])

    module = types.ModuleType("py_oats.analyzers.coordination")
    module.CoordinationAnalyzer = _Analyzer
    monkeypatch.setitem(__import__("sys").modules,
                        "py_oats.analyzers.coordination", module)

    sigmas = h.contact_sigmas(trajectory=None, species=["Sr", "O"])
    assert sigmas == pytest.approx([1.6, 1.0, 1.2], abs=0.02)


def test_dilute_pair_never_gets_zero_sigma():
    """Two Ca atoms in a Ca-Nb-O melt never meet within rmax: the Ca-Ca pair
    must borrow the largest sigma on Ca (Ca-Nb), not collapse to 0."""
    pairs = h.type_pairs(3)                       # Ca, Nb, O
    measured = [None, 1.57, 1.02, 1.49, 0.82, 1.17]
    assert h.fill_missing_sigmas(pairs, measured) == pytest.approx(
        [1.57, 1.57, 1.02, 1.49, 0.82, 1.17])
    with pytest.raises(ValueError):
        h.fill_missing_sigmas(pairs, [None] * 6)


def test_binned_quench_drops_startup_spike(tmp_path):
    T = np.linspace(600, 2900, 500)
    H = -8.0 + 1e-3 * (T - 600)
    rows = "\n".join(f"{i} {t} {108*hh} 1300" for i, (t, hh) in
                     enumerate(zip(T, H)))
    spike = "\n0 7358.0 -400.0 1300"          # packed-cell transient
    (tmp_path / "ht_leg0.dat").write_text("# a\n# b\n" + rows + spike)
    (tmp_path / "ht_equil.dat").write_text("# a\n# b\n" + rows.split("\n")[0])
    Tq, Hq = h.binned_quench_ht(str(tmp_path), 108, t_anchor=2900.0)
    assert Tq.max() < 3000                     # spike gone
    assert abs(np.interp(1000.0, Tq, Hq) - (-8.0 + 0.4)) < 5e-3


# ---------------------------------------------------------------- G(T) math

def test_amorphous_curve_matches_constant_H_analytic():
    Ts = np.array([800.0])
    Ta, Ga, Hc = 2000.0, -8.0, -6.0
    G = h.amorphous_gibbs_curve(Ts, Ta, Ga, np.array([500.0, 2100.0]),
                                np.array([Hc, Hc]))
    exact = Ts[0] * (Ga / Ta + Hc * (1 / Ts[0] - 1 / Ta))
    assert abs(G[0] - exact) < 1e-5


def test_crystal_curve_recovers_anchor():
    G = h.crystal_gibbs_curve(np.array([950.0]), 950.0, h0=-7.7, g0=-8.2)
    assert abs(G[0] - (-8.2)) < 1e-12


def test_amorphous_curve_refuses_uncovered_range():
    with pytest.raises(ValueError, match="descent needs"):
        h.amorphous_gibbs_curve(np.array([700.0]), 2900.0, -9.0,
                                np.array([1000.0, 2000.0]),
                                np.array([-6.0, -6.0]))


def _sweep_points(Ts, H, V=None):
    V = V if V is not None else 12.0 + 1e-4 * np.asarray(Ts)
    return list(zip(map(float, Ts), map(float, H), map(float, V)))


def test_gibbs_helmholtz_matches_quadratic_H_analytic():
    # Cp = b + 2cT rising with T, both sides of the anchor:
    # G(T)/T = G0/T0 - [a (1/T0 - 1/T) + b ln(T/T0) + c (T - T0)]
    a, b, c, T0, G0 = -8.0, h.THREE_R, 2e-8, 950.0, -8.3
    data_T = np.linspace(600.0, 1300.0, 701)
    data_H = a + b * data_T + c * data_T ** 2
    Ts = np.array([700.0, 950.0, 1200.0])
    G = h.gibbs_helmholtz(Ts, T0, G0, data_T, data_H)
    exact = Ts * (G0 / T0 - (a * (1 / T0 - 1 / Ts) + b * np.log(Ts / T0)
                             + c * (Ts - T0)))
    assert np.allclose(G, exact, atol=2e-6)


def test_crystal_sweep_with_cp_3R_reproduces_legacy_carry():
    # a sweep whose H rises at exactly 3R must give back the 3R formula
    T0, h0, g0 = 950.0, -7.7, -8.2
    ladder = h.crystal_temperature_ladder(range(700, 1201, 25), T0)
    sweep = h.crystal_enthalpy_sweep(
        _sweep_points(ladder, [h0 + h.THREE_R * (t - T0) for t in ladder]), T0)
    Ts = np.arange(700.0, 1201.0, 25.0)
    G = h.crystal_gibbs_helmholtz(Ts, T0, g0, sweep)
    assert np.allclose(G, h.crystal_gibbs_curve(Ts, T0, h0, g0), atol=2e-6)
    assert sweep["transition"] is None and not sweep["warnings"]


def test_rising_cp_moves_G_away_from_the_3R_carry():
    # Cp/3R 1.0 -> 1.3 over the window, as the Y-Al-O oxides show: the carry
    # is exact at T0 and wrong away from it, by ~ -1/2 dCp (T-T0)^2 / T0
    T0, g0 = 950.0, -8.2
    Ts_data = np.linspace(700.0, 1200.0, 6)
    H = [-7.7 + h.THREE_R * ((t - T0) + 0.3 * (t - 700) ** 2 / 1000.0
                             - 0.3 * (T0 - 700) ** 2 / 1000.0) for t in Ts_data]
    sweep = h.crystal_enthalpy_sweep(_sweep_points(Ts_data, H), T0)
    h0 = float(np.interp(T0, sweep["T"], sweep["H"]))
    Ts = np.array([700.0, 950.0, 1200.0])
    G = h.crystal_gibbs_helmholtz(Ts, T0, g0, sweep)
    G3 = h.crystal_gibbs_curve(Ts, T0, h0, g0)
    assert abs(G[1] - g0) < 1e-9 and abs(G3[1] - g0) < 1e-9
    assert (G - G3)[2] < -1e-4        # rising Cp lowers G above T0


def test_sweep_cuts_at_latent_heat_and_continues_the_parent():
    T0 = 950.0
    Ts = np.array([700.0, 800.0, 950.0, 1100.0, 1200.0])
    H = -7.7 + h.THREE_R * (Ts - T0)
    H[-1] += 0.05                       # 50 meV/atom latent heat in 1100-1200
    sweep = h.crystal_enthalpy_sweep(_sweep_points(Ts, H), T0)
    assert sweep["transition"] == (1100.0, 1200.0)
    assert any("metastable parent" in w for w in sweep["warnings"])
    # the parent is continued at its own slope, so G stays the 3R one
    G = h.crystal_gibbs_helmholtz(np.array([1200.0]), T0, -8.2, sweep)
    assert abs(G[0] - h.crystal_gibbs_curve(np.array([1200.0]), T0, -7.7, -8.2)[0]) < 2e-6


def test_sweep_refuses_a_transition_below_the_anchor():
    Ts = np.array([700.0, 800.0, 950.0, 1100.0])
    V = np.array([12.0, 12.1, 11.9, 12.0])          # shrinks on heating at 950
    with pytest.raises(ValueError, match="below the anchor"):
        h.crystal_enthalpy_sweep(_sweep_points(Ts, -7.7 + h.THREE_R * (Ts - 1100), V), 1100.0)


def test_crystal_curve_refuses_to_extrapolate():
    Ts = np.array([800.0, 950.0, 1100.0])
    sweep = h.crystal_enthalpy_sweep(
        _sweep_points(Ts, -7.7 + h.THREE_R * (Ts - 950)), 950.0)
    with pytest.raises(ValueError, match="covers"):
        h.crystal_gibbs_helmholtz(np.array([700.0, 1200.0]), 950.0, -8.2, sweep)


def test_ladder_spans_grid_and_holds_the_anchor():
    ladder = h.crystal_temperature_ladder(range(700, 1201, 25), 975.0)
    assert ladder[0] == 700.0 and ladder[-1] == 1200.0 and 975.0 in ladder
    assert max(np.diff(ladder)) <= 100.0 + 1e-9
    # an anchor outside the report window is still covered
    assert h.crystal_temperature_ladder([700, 800], 1000.0)[-1] == 1000.0


def _ht_run_dir(path, structure, T, H, V):
    """A finished CrystalEnthalpyMaker directory: ht.dat + final.data."""
    from pymatgen.io.lammps.data import LammpsData
    path.mkdir()
    n = len(structure)
    rows = "\n".join(f"{500*(i+1)} {T} {H*n} {V*n}" for i in range(20))
    (path / "ht.dat").write_text("# Time-averaged data\n# TimeStep v_tt v_hh v_vv\n" + rows)
    LammpsData.from_structure(structure, atom_style="atomic").write_file(str(path / "final.data"))
    return str(path)


def test_build_gibbs_curve_job_both_crystal_methods(srtio3, tmp_path):
    from finite_temp_properties.schemas.free_energy import SolidFreeEnergyDoc
    from finite_temp_properties.workflow.jobs.analysis import build_gibbs_curve
    T0, g0 = 950.0, -8.2
    solid = SolidFreeEnergyDoc(temperature=T0, species=["Sr", "Ti", "O"], free_energy=g0,
                               einstein_reference=-9.0, work=-0.8, hysteresis=0.001,
                               spring_constants=[1.0, 2.0, 1.5], volume_per_atom=12.0,
                               natoms=len(srtio3))
    ladder = [700.0, 800.0, 900.0, 950.0, 1000.0, 1100.0, 1200.0]
    # Cp/3R = 1.1 throughout
    dirs = [_ht_run_dir(tmp_path / f"ht{t:.0f}", srtio3, t, -7.7 + 1.1 * h.THREE_R * (t - T0),
                        12.0 + 1e-4 * t) for t in ladder]
    temps = [700.0, 950.0, 1200.0]
    gh = build_gibbs_curve.original(temps, solid=solid, crystal_ht_dirs=dirs)
    assert gh.crystal_method == "gibbs_helmholtz"
    assert gh.crystal_cp_over_3R == pytest.approx([1.1] * 6)
    assert abs(gh.g_crystal[1] - g0) < 1e-9
    legacy = build_gibbs_curve.original(temps, solid=solid,
                                        crystal_ht_dirs=[dirs[3]])
    assert legacy.crystal_method == "harmonic_3R" and legacy.crystal_warnings
    assert gh.s_crystal_anchor == pytest.approx(legacy.s_crystal_anchor)
    # 10 % excess Cp lowers G on both sides of the anchor (G is concave)
    assert gh.g_crystal[0] < legacy.g_crystal[0] and gh.g_crystal[2] < legacy.g_crystal[2]


# --------------------------------------------------- templates and flows

def test_msd_template_renders_fully(srtio3, tmp_path):
    from finite_temp_properties.workflow.jobs import CrystalMSDMaker
    m = CrystalMSDMaker(settings={"temperature": 950.0})
    m.input_set_generator.update_settings(
        h.species_settings(srtio3) | h.msd_blocks(srtio3.n_elems),
        validate_params=False)
    m.input_set_generator.get_input_set(srtio3).write_input(str(tmp_path))
    text = (tmp_path / "in.lammps").read_text()
    assert " Sr Ti O" in text and "compute m3 g3 msd com yes" in text
    # nothing template-shaped left except LAMMPS's own runtime variables
    leftovers = {w for w in text.split() if w.startswith("${")}
    assert leftovers <= {"${temperature}", "${vpa}"} - {"${temperature}"}, leftovers


def test_every_template_renders(srtio3, tmp_path):
    """Every stage must produce a parseable in.lammps. A generated block with
    a blank line in it renders as an empty command and raises -- which is how
    the quench template was broken without any other test noticing."""
    from finite_temp_properties.workflow.jobs import (
        CrystalEnthalpyMaker, CrystalMSDMaker, FrenkelLaddMaker,
        MeltEquilibrationMaker, PotentialSwitchMaker, QuenchMaker,
        UFMSwitchLeg1Maker, UFMSwitchLeg2Maker)

    springs = [2.0, 3.0, 1.5]
    sigmas = [1.6, 1.4, 1.2, 1.3, 1.1, 1.0]
    stages = [
        (CrystalMSDMaker(), ()),
        (FrenkelLaddMaker(), (springs,)),
        (MeltEquilibrationMaker(), ()),
        (UFMSwitchLeg1Maker(), (sigmas,)),
        (UFMSwitchLeg2Maker(), (sigmas, 1.35)),
        (PotentialSwitchMaker(), ()),
        (CrystalEnthalpyMaker(), ()),
        (QuenchMaker(), ()),
    ]
    # LAMMPS's own runtime variables, declared inside the inputs -- these are
    # meant to survive substitution
    runtime_vars = {"${lam}", "${dU}", "${vpa}"}

    for maker, args in stages:
        # make() fills the input set from the settings as a side effect
        maker.make(srtio3, *args)
        out = tmp_path / maker.name
        maker.input_set_generator.get_input_set(srtio3).write_input(str(out))
        text = (out / "in.lammps").read_text()
        leftover = set(re.findall(r"\$\{\w+\}", text)) - runtime_vars
        assert not leftover, f"{maker.name}: {leftover}"
        # a generated block must not open a run of blank lines: pymatgen reads
        # that as an empty stage and raises (this is what broke the quench)
        assert "\n\n\n" not in text, f"{maker.name}: empty stage in the input"


def test_solid_flow_shape(srtio3):
    from finite_temp_properties.workflow.flows import SolidFreeEnergyMaker
    flow = SolidFreeEnergyMaker().make(srtio3)
    names = [j.name for j in flow.jobs]
    assert names == ["BaseLammpsMaker.make", "frenkel_ladd",
                     "analyze_solid_free_energy", "potential_switch",
                     "analyze_potential_switch"]


def test_leg1_runs_at_the_mean_melt_volume(srtio3, tmp_path, monkeypatch):
    """The UF reference is evaluated at the mean NPT density, so leg 1 must
    switch at that density too -- not in the last-step box liq.data holds,
    which was 3-8% off and cost ~7 meV/atom per percent."""
    import types

    from pymatgen.io.lammps.data import LammpsData

    from finite_temp_properties.workflow.jobs import makers
    LammpsData.from_structure(srtio3, atom_style="atomic").write_file(
        str(tmp_path / "liq.data"))
    snapshot = srtio3.volume / len(srtio3)
    mean = 1.05 * snapshot
    (tmp_path / "vol.dat").write_text(
        "# a\n# b\n" + "\n".join(f"{i} {mean} -100.0" for i in range(10)))
    monkeypatch.setattr(h, "sigmas_from_melt", lambda *a, **k: [1.0] * 6)
    seen = {}
    fake = types.SimpleNamespace(original=lambda self, input_structure: (
        seen.setdefault("v", input_structure.volume / len(input_structure))))
    monkeypatch.setattr(makers.BaseLammpsMaker, "make", fake)
    maker = makers.UFMSwitchLeg1Maker()
    maker.make_from.original(maker, str(tmp_path))
    assert seen["v"] == pytest.approx(mean, rel=1e-9)


def test_liquid_flow_shape(srtio3):
    from finite_temp_properties.workflow.flows import LiquidFreeEnergyMaker
    flow = LiquidFreeEnergyMaker(with_switch=False).make(srtio3)
    assert [j.name for j in flow.jobs] == [
        "BaseLammpsMaker.make", "ufm_switch_leg1", "ufm_switch_leg2",
        "analyze_liquid_free_energy"]


def test_gibbs_flow_shares_anchor_temperatures(srtio3):
    from finite_temp_properties.workflow.flows import (
        GibbsCurveMaker, LiquidFreeEnergyMaker, SolidFreeEnergyMaker)
    from finite_temp_properties.workflow.jobs import (
        CrystalMSDMaker, MeltEquilibrationMaker)
    maker = GibbsCurveMaker(
        solid_maker=SolidFreeEnergyMaker(
            msd_maker=CrystalMSDMaker(settings={"temperature": 900.0})),
        liquid_maker=LiquidFreeEnergyMaker(
            melt_maker=MeltEquilibrationMaker(settings={"temperature": 2500.0})))
    assert maker.crystal_enthalpy_maker.settings["temperature"] == 900.0
    assert maker.quench_maker.settings["temperature"] == 2500.0
    flow = maker.make(crystal=srtio3, melt=srtio3)
    assert flow.jobs[-1].name == "build_gibbs_curve"


def test_pushed_temperatures_reach_the_rendered_inputs(srtio3, tmp_path):
    """temperature_crystal / temperature_melt must change what LAMMPS RUNS,
    not just maker.settings: the generator holds its own copy, and editing
    the dict alone once left every stage at the 950 / 2900 K defaults."""
    from finite_temp_properties.workflow.flows import GibbsCurveMaker
    maker = GibbsCurveMaker(temperature_crystal=800.0, temperature_melt=2500.0)
    stages = [(maker.solid_maker.msd_maker, 800.0, ()),
              (maker.solid_maker.frenkel_ladd_maker, 800.0, ([2.0, 3.0, 1.5],)),
              (maker.crystal_enthalpy_maker, 800.0, ()),
              (maker.liquid_maker.melt_maker, 2500.0, ()),
              (maker.quench_maker, 2500.0, ())]
    for stage, T, args in stages:
        stage.make(srtio3, *args)
        out = tmp_path / stage.name
        stage.input_set_generator.get_input_set(srtio3).write_input(str(out))
        text = (out / "in.lammps").read_text()
        assert f"velocity all create {T}" in text, stage.name


def test_gibbs_flow_runs_one_enthalpy_job_per_ladder_temperature(srtio3, tmp_path):
    from finite_temp_properties.workflow.flows import GibbsCurveMaker
    maker = GibbsCurveMaker(temperature_crystal=950.0,
                            temperatures=list(range(700, 1201, 25)))
    assert maker.crystal_temperatures == [700.0, 800.0, 900.0, 950.0, 1000.0, 1100.0, 1200.0]
    makers = maker.crystal_enthalpy_makers()
    assert [m.input_set_generator.settings.temperature for m in makers] == maker.crystal_temperatures
    makers[0].make(srtio3)
    makers[0].input_set_generator.get_input_set(srtio3).write_input(str(tmp_path))
    assert "fix p all npt temp 700.0 700.0" in (tmp_path / "in.lammps").read_text()
    flow = maker.make(crystal=srtio3, melt=srtio3)
    assert len(flow.jobs) == 2 + len(makers) + 1 + 1
    legacy = GibbsCurveMaker(crystal_temperatures=[])
    assert len(legacy.crystal_enthalpy_makers()) == 1
