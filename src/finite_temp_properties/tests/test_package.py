"""Tests that run anywhere: block builders, parsers on synthetic files,
the G(T) math against analytic results, template rendering, and flow
construction. No LAMMPS, no network."""
from __future__ import annotations

import gzip

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


def test_sigmas_from_rdf_takes_last_block_and_half_contact(tmp_path):
    # one pair; g crosses 0.5 at r = 2.0 in the last block -> sigma = 1.0
    block1 = "1000 3\n1 1.0 0.0 0\n2 2.0 0.0 0\n3 3.0 1.0 0\n"
    block2 = "2000 3\n1 1.0 0.0 0\n2 2.0 0.9 0\n3 3.0 1.0 0\n"
    (tmp_path / "rdf.dat").write_text("# h1\n# h2\n# h3\n" + block1 + block2)
    assert h.sigmas_from_rdf(str(tmp_path), 1) == [1.0]


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


# --------------------------------------------------- templates and flows

def test_msd_template_renders_fully(srtio3, tmp_path):
    from finite_temp_properties.workflow.core import CrystalMSDMaker
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


def test_solid_flow_shape(srtio3):
    from finite_temp_properties.workflow.core import SolidFreeEnergyMaker
    flow = SolidFreeEnergyMaker().make(srtio3)
    names = [j.name for j in flow.jobs]
    assert names == ["BaseLammpsMaker.make", "frenkel_ladd",
                     "analyze_solid_free_energy", "potential_switch",
                     "analyze_potential_switch"]


def test_liquid_flow_shape(srtio3):
    from finite_temp_properties.workflow.core import LiquidFreeEnergyMaker
    flow = LiquidFreeEnergyMaker(with_switch=False).make(srtio3)
    assert [j.name for j in flow.jobs] == [
        "BaseLammpsMaker.make", "ufm_switch_leg1", "ufm_switch_leg2",
        "analyze_liquid_free_energy"]


def test_gibbs_flow_shares_anchor_temperatures(srtio3):
    from finite_temp_properties.workflow.core import (
        GibbsCurveMaker, LiquidFreeEnergyMaker, MeltEquilibrationMaker,
        SolidFreeEnergyMaker, CrystalMSDMaker)
    maker = GibbsCurveMaker(
        solid_maker=SolidFreeEnergyMaker(
            msd_maker=CrystalMSDMaker(settings={"temperature": 900.0})),
        liquid_maker=LiquidFreeEnergyMaker(
            melt_maker=MeltEquilibrationMaker(settings={"temperature": 2500.0})))
    assert maker.crystal_enthalpy_maker.settings["temperature"] == 900.0
    assert maker.quench_maker.settings["temperature"] == 2500.0
    flow = maker.make(crystal=srtio3, melt=srtio3)
    assert flow.jobs[-1].name == "build_gibbs_curve"
