"""Everything the workflow makers share: default settings, the input blocks
built per species, the parsers for what LAMMPS writes, the reference free
energies, and the two G(T) formulas.

All numerical conventions are inherited from a validated campaign
(SW-Si checked against calphy; Gibbs-Helmholtz vs TI closed to 2.25 meV/atom).
Energies are eV/atom, temperatures K, pressures zero throughout.
"""
from __future__ import annotations

import glob
import gzip
import os
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

KB = 8.617333262e-5          # Boltzmann constant, eV/K
THREE_R = 3 * 8.314 / 96485.0  # Dulong-Petit heat capacity 3R, eV/K/atom
EV_ATOM_TO_J_MOL = 96485.0

#: Calibrated Uhlenbeck-Ford EXCESS free energy, in units of kT, versus the
#: dimensionless density x = (pi^(3/2)/2) sigma^3 rho. Measured once by
#: nonequilibrium switching up a sigma ladder and anchored where the fluid is
#: dilute enough for calphy's analytic form. F_UF/kT depends only on (p, x) --
#: not on T, masses, or composition -- which is what makes it transferable.
#: Do not refit; valid only at the two couplings measured.
UF_LADDER = {
    50.0: (np.array([0.001260, 0.010124, 0.034169, 0.080927,
                     0.158060, 0.273129]),
           np.array([0.009345, 0.075414, 0.274228, 0.729054,
                     1.714247, 3.859768])),
    25.0: (np.array([0.001260, 0.010124, 0.034169, 0.080927,
                     0.158060, 0.273129]),
           np.array([0.007452, 0.060540, 0.213580, 0.551430,
                     1.227257, 2.553107])),
}
#: The ladder point every production run is placed on (sigma0 is chosen so
#: the reduced density x lands here exactly).
X_TARGET = 0.158060
_KX = 0.5 * np.pi ** 1.5


# ---------------------------------------------------------------------------
# default settings, one dict per template
# ---------------------------------------------------------------------------
# Times are LAMMPS steps at time_step = 0.001 ps (1 fs). ${pair_style} and
# ${potential_path} default to the GRACE-FS reference; the target-potential
# runs (potential switch, enthalpies) override them.

_COMMON = {
    "atom_style": "atomic",     # what LammpsData is written as
    "pair_style": "grace/fs",
    "potential_path": str(Path.home() / ".cache/grace/checkpoints/GRACE-FS-OAM/GRACE-FS-OAM.yaml"),
    "time_step": 0.001,
    "thermo_interval": 1000,
    "seed": 12345,
}

CRYSTAL_MSD_DEFAULTS = _COMMON | {
    "temperature": 950.0,
    "n_equil": 15000,
    "n_msd": 20000,
}

FRENKEL_LADD_DEFAULTS = _COMMON | {
    "temperature": 950.0,
    "n_equil": 15000,
    "n_switch": 40000,
    "spring_scale": 1.0,   # multiply the measured k's (k-invariance test: 4.0)
}

MELT_EQUILIBRATION_DEFAULTS = _COMMON | {
    "temperature": 2900.0,
    "n_equil": 20000,
    "n_sample": 40000,
}

UFM_SWITCH_DEFAULTS = _COMMON | {
    "temperature": 2900.0,
    "n_equil": 20000,
    "n_switch": 40000,
    "ufm_p": 50.0,         # coupling: eps = p * kB * T (p-invariance test: 25)
    "seed": 55123,
}

POTENTIAL_SWITCH_DEFAULTS = _COMMON | {
    "reference_pair_style": "grace/fs",
    "reference_potential_path": _COMMON["potential_path"],
    "target_pair_style": "grace",
    "target_potential_path": str(Path.home() / ".cache/grace/GRACE-2L-SMAX-OMAT-medium"),
    "temperature": 950.0,
    "barostat": "iso",     # "tri" for crystals, "iso" for liquids
    "n_equil": 8000,
    "n_switch": 25000,
    "thermo_interval": 2000,
    "seed": 20261,
}

CRYSTAL_ENTHALPY_DEFAULTS = _COMMON | {
    "pair_style": "grace",
    "potential_path": POTENTIAL_SWITCH_DEFAULTS["target_potential_path"],
    "temperature": 950.0,
    "barostat": "tri",
    "n_run": 35000,
}

QUENCH_DEFAULTS = _COMMON | {
    "pair_style": "grace",
    "potential_path": POTENTIAL_SWITCH_DEFAULTS["target_potential_path"],
    "temperature": 2900.0,   # the melt anchor the quench starts from
    "t_floor": 600.0,        # quench down to here (below the report window)
    "quench_rate": 10.0,     # K/ps
    "n_legs": 5,
    "n_equil": 15000,
}


# ---------------------------------------------------------------------------
# species order and the per-species input blocks
# ---------------------------------------------------------------------------

def species_of(structure: Structure) -> list[str]:
    """Element symbols in the order LammpsData assigns atom types.

    pymatgen numbers types by sorted(Element), i.e. Pauling electronegativity
    -- NOT alphabetically, NOT first appearance. Everything per-species
    (pair_coeff, groups, springs, masses) must follow this order or elements
    get silently swapped.
    """
    return [el.symbol for el in sorted(structure.composition.elements)]


def species_settings(structure: Structure) -> dict:
    return {"species_string": " ".join(species_of(structure))}


def type_pairs(n_species: int) -> list[tuple[int, int]]:
    """All (i, j) type pairs with i <= j, 1-indexed like LAMMPS."""
    return [(i, j) for i in range(1, n_species + 1)
            for j in range(i, n_species + 1)]


def msd_blocks(n_species: int) -> dict:
    """The group/compute/column lines for a per-species MSD measurement."""
    ids = range(1, n_species + 1)
    return {
        "msd_groups": "\n".join(f"group g{i} type {i}" for i in ids),
        "msd_computes": "\n".join(f"compute m{i} g{i} msd com yes" for i in ids),
        "msd_columns": " ".join(f"c_m{i}[4]" for i in ids),
    }


def spring_blocks(spring_constants: list[float], n_switch: int,
                  n_equil: int) -> dict:
    """The Frenkel-Ladd fixes, one ti/spring per species group.

    fix ti/spring takes a single k, so each species gets its own fix on its
    own (disjoint) type group; the work integrand sums the spring energies.
    The template places these BEFORE the Langevin thermostat -- keep it that
    way (see the template header).
    """
    ids = range(1, len(spring_constants) + 1)
    return {
        "spring_groups": "\n".join(f"group g{i} type {i}" for i in ids),
        "spring_fixes": "\n".join(
            f"fix ti{i} g{i} ti/spring {k:.6f} {n_switch} {n_equil} function 2"
            for i, k in zip(ids, spring_constants)),
        "spring_energy_sum": "+".join(f"f_ti{i}" for i in ids),
        "n_total": 2 * n_equil + 2 * n_switch,
    }


def rdf_pairs_setting(n_species: int) -> str:
    """The pair list for `compute rdf`, all i <= j."""
    return " ".join(f"{i} {j}" for i, j in type_pairs(n_species))


def ufm_coeff_block(eps: float, sigmas: list[float], n_species: int,
                    substyle: str = "ufm") -> str:
    """pair_coeff lines for a (possibly multi-sigma) UFM, one per type pair."""
    return "\n".join(
        f"pair_coeff {i} {j} {substyle} {eps:.6f} {s:.6f}"
        for (i, j), s in zip(type_pairs(n_species), sigmas))


def quench_leg_block(t_anchor: float, t_floor: float, rate_K_per_ps: float,
                     n_legs: int, time_step_ps: float) -> str:
    """The NPT cooling legs, each logging its own ht_leg<i>.dat."""
    span = (t_anchor - t_floor) / n_legs
    steps = int(round(span / rate_K_per_ps / time_step_ps))
    legs = []
    for i in range(n_legs):
        hi, lo = t_anchor - i * span, t_anchor - (i + 1) * span
        legs.append(
            f"fix q all npt temp {hi:.1f} {lo:.1f} 0.1 iso 0.0 0.0 1.0\n"
            f"fix a{i} all ave/time 10 20 200 v_tt v_hh v_vv file ht_leg{i}.dat\n"
            f"run {steps}\nunfix q\nunfix a{i}\n")
    return "\n".join(legs)


def triclinic_setting(structure: Structure) -> str:
    """`change_box all triclinic` when the cell is orthogonal and a `tri`
    barostat will need tilts; a dropped line otherwise (the generator removes
    lines that substitute to '###')."""
    ortho = np.allclose(structure.lattice.matrix, np.diag(np.diag(
        structure.lattice.matrix)), atol=1e-8)
    return "change_box all triclinic" if ortho else "### box already triclinic"


# ---------------------------------------------------------------------------
# reading what the runs wrote (atomate2 gzips job dirs -- every reader here
# accepts plain or .gz)
# ---------------------------------------------------------------------------

def _read(path_no_gz: str) -> str:
    for p in (path_no_gz, path_no_gz + ".gz"):
        if os.path.exists(p):
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", errors="replace") as f:
                return f.read()
    raise FileNotFoundError(path_no_gz)


def _table(path_no_gz: str, skiprows: int) -> np.ndarray:
    lines = _read(path_no_gz).splitlines()[skiprows:]
    return np.array([[float(x) for x in l.split()] for l in lines if l.split()])


def structure_from(run_dir: str, filename: str) -> Structure:
    """The structure a previous run wrote (equil.data, liq.data, ...)."""
    from pymatgen.io.lammps.data import LammpsData
    for p in (f"{run_dir}/{filename}", f"{run_dir}/{filename}.gz"):
        if os.path.exists(p):
            return LammpsData.from_file(p, atom_style="atomic").structure
    raise FileNotFoundError(f"{filename} not in {run_dir}")


def forward_backward_work(run_dir: str) -> tuple[float, float]:
    """Mean irreversible work and hysteresis from fwd.dat/bwd.dat.

    W = (W_fwd + W_bwd)/2 cancels the linear-response dissipation; the
    difference |W_fwd - W_bwd| is the error scale.
    """
    f = _table(f"{run_dir}/fwd.dat", skiprows=1)
    b = _table(f"{run_dir}/bwd.dat", skiprows=1)
    w_fwd = np.trapezoid(f[:, 1], f[:, 0])
    w_bwd = np.trapezoid(b[::-1, 1], b[::-1, 0])
    return 0.5 * (w_fwd + w_bwd), abs(w_fwd - w_bwd)


def frenkel_ladd_work(run_dir: str) -> tuple[float, float]:
    """Same, from the single ti.dat that fix ti/spring writes (fwd then bwd)."""
    t = _table(f"{run_dir}/ti.dat", skiprows=1)
    n = len(t) // 2
    w_fwd = np.trapezoid(t[:n, 1], t[:n, 0])
    w_bwd = np.trapezoid(t[n:][::-1, 1], t[n:][::-1, 0])
    return 0.5 * (w_fwd + w_bwd), abs(w_fwd - w_bwd)


def spring_constants_from_msd(run_dir: str, temperature: float,
                              n_species: int) -> tuple[list[float], float]:
    """k_i = 3 kB T / <dr^2>_i and the volume/atom, from msd.dat (2nd half)."""
    m = _table(f"{run_dir}/msd.dat", skiprows=2)
    m = m[len(m) // 2:]
    k = [3 * KB * temperature / m[:, 1 + i].mean() for i in range(n_species)]
    return k, float(m[:, 1 + n_species].mean())


def sigmas_from_rdf(run_dir: str, n_species: int) -> list[float]:
    """sigma_ij = half the r where the partial RDF first reaches 0.5.

    The UFM repulsion falls to ~kT near r = 2 sigma, so this is half an
    effective contact diameter -- measured per pair. rdf.dat is written by
    `fix ave/time ... mode vector`: one block per window, each headed by a
    two-field line; use the LAST (best-equilibrated) block.
    """
    blocks, current = [], []
    for line in _read(f"{run_dir}/rdf.dat").splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 2:
            if current:
                blocks.append(current)
            current = []
        elif fields:
            current.append([float(v) for v in fields])
    if current:
        blocks.append(current)
    data = np.array(blocks[-1])
    r = data[:, 1]
    sigmas = []
    for k in range(len(type_pairs(n_species))):
        g = data[:, 2 + 2 * k]
        i = int(np.argmax(g > 0.5))
        sigmas.append(0.5 * r[i] if g[i] > 0.5 else 0.5 * r[0])
    return sigmas


def volume_and_enthalpy(run_dir: str) -> tuple[float, float]:
    """Mean volume/atom and enthalpy (per CELL) over the back half of vol.dat."""
    v = _table(f"{run_dir}/vol.dat", skiprows=2)
    half = v[len(v) // 2:]
    return float(half[:, 1].mean()), float(half[:, 2].mean())


def endpoint_volumes(run_dir: str) -> tuple[float, float]:
    """The two endpoint volumes/atom of a potential switch (last 3 windows)."""
    ref = _table(f"{run_dir}/vol_reference.dat", skiprows=2)
    tgt = _table(f"{run_dir}/vol_target.dat", skiprows=2)
    return float(ref[-3:, 1].mean()), float(tgt[-3:, 1].mean())


def atom_counts(run_dir: str, filename: str, n_species: int) -> list[int]:
    """Atoms of each type in a LAMMPS data file."""
    text = _read(f"{run_dir}/{filename}")
    counts = [0] * n_species
    for line in text.split("Atoms")[1].splitlines():
        fields = line.split()
        if len(fields) >= 5:
            try:
                counts[int(fields[1]) - 1] += 1
            except (ValueError, IndexError):
                pass
    return counts


def mean_ht(run_dir: str, natoms: int) -> tuple[float, float]:
    """Measured mean T and enthalpy/atom over the back half of ht.dat."""
    a = _table(f"{run_dir}/ht.dat", skiprows=2)
    half = a[len(a) // 2:]
    return float(half[:, 1].mean()), float(half[:, 2].mean()) / natoms


def binned_quench_ht(run_dir: str, natoms: int,
                     t_anchor: float) -> tuple[np.ndarray, np.ndarray]:
    """H(T) per atom, binned along the quench, for the Gibbs-Helmholtz integrand.

    Points above 1.02 * anchor are the startup transient of a packed cell and
    are dropped (they would stretch the bins, not enter the integral); a
    20-MAD pass on H removes corrupt rows without touching a genuine smooth
    descent.
    """
    T, H = [], []
    for f in sorted(glob.glob(f"{run_dir}/ht_leg*.dat*")) + [
            f"{run_dir}/ht_equil.dat"]:
        base = f[:-3] if f.endswith(".gz") else f
        try:
            a = _table(base, skiprows=2)
        except FileNotFoundError:
            continue
        if len(a):
            T += list(a[:, 1])
            H += list(a[:, 2] / natoms)
    T, H = np.array(T), np.array(H)
    keep = T <= 1.02 * t_anchor
    T, H = T[keep], H[keep]
    med = np.median(H)
    mad = np.median(np.abs(H - med)) or 1e-9
    ok = np.abs(H - med) < 20 * mad
    T, H = T[ok], H[ok]
    order = np.argsort(T)
    T, H = T[order], H[order]
    edges = np.linspace(T.min(), T.max(), 41)
    tc, hc = [], []
    for i in range(40):
        m = (T >= edges[i]) & (T < edges[i + 1])
        if m.sum() >= 3:
            tc.append(T[m].mean())
            hc.append(H[m].mean())
    return np.array(tc), np.array(hc)


# ---------------------------------------------------------------------------
# reference free energies (via calphy, so the convention matches the numbers
# this package was validated against)
# ---------------------------------------------------------------------------

def einstein_crystal_fe(temperature: float, spring_constants: list[float],
                        species: list[str], counts: list[int],
                        vol_per_atom: float) -> float:
    """Classical Einstein-crystal F (eV/atom), with the centre-of-mass
    correction for the fixed centroid the switching imposes."""
    from ase.data import atomic_masses, chemical_symbols
    from calphy.integrators import get_einstein_crystal_fe

    class _Calc:  # calphy reads exactly these three attributes
        pass
    calc = _Calc()
    calc.element = list(species)
    calc._temperature = temperature
    calc._element_dict = {
        s: {"count": c, "mass": float(atomic_masses[chemical_symbols.index(s)])}
        for s, c in zip(species, counts)}
    return get_einstein_crystal_fe(
        calc, vol_per_atom, list(spring_constants), cm_correction=True)


def sigma_for_target_x(vol_per_atom: float) -> float:
    """The single-fluid sigma that puts this density exactly on X_TARGET."""
    rho = 1.0 / vol_per_atom
    return (X_TARGET / (_KX * rho)) ** (1.0 / 3.0)


def uf_reference_fe(temperature: float, vol_per_atom: float, p: float,
                    species: list[str], counts: list[int]) -> float:
    """F of the single-sigma UF fluid at X_TARGET: calibrated excess + ideal gas."""
    from ase.data import atomic_masses, chemical_symbols
    from calphy.integrators import get_ideal_gas_fe

    xs, fs = UF_LADDER[p]
    excess = float(np.interp(X_TARGET, xs, fs)) * KB * temperature
    natoms = sum(counts)
    rho = 1.0 / vol_per_atom
    masses = [float(atomic_masses[chemical_symbols.index(s)]) for s in species]
    ideal = get_ideal_gas_fe(temperature, rho, natoms, masses,
                             [c / natoms for c in counts])
    return excess + ideal


# ---------------------------------------------------------------------------
# the two G(T) formulas
# ---------------------------------------------------------------------------

def crystal_gibbs_curve(temperatures: np.ndarray, t0: float, h0: float,
                        g0: float) -> np.ndarray:
    """Classical-harmonic carry from the anchor (Cp = 3R):

        S(T0) = (H(T0) - G(T0)) / T0
        G(T)  = H(T0) + 3R (T - T0) - T [ S(T0) + 3R ln(T/T0) ]
    """
    s0 = (h0 - g0) / t0
    return h0 + THREE_R * (temperatures - t0) - temperatures * (
        s0 + THREE_R * np.log(temperatures / t0))


def amorphous_gibbs_curve(temperatures: np.ndarray, t_anchor: float,
                          g_anchor: float, quench_T: np.ndarray,
                          quench_H: np.ndarray) -> np.ndarray:
    """Gibbs-Helmholtz descent from the melt anchor (sign is +, T -> Ta):

        G(T) = T [ G(Ta)/Ta + int_T^Ta H(T')/T'^2 dT' ]

    Valid only at fixed pressure -- quench_H must be NPT enthalpies.
    """
    if temperatures.min() < quench_T.min() - 25 or \
            t_anchor > quench_T.max() + 50:
        raise ValueError(
            f"quench H(T) covers {quench_T.min():.0f}-{quench_T.max():.0f} K, "
            f"the descent needs {temperatures.min():.0f}-{t_anchor:.0f} K")
    out = []
    for T in temperatures:
        grid = np.linspace(T, t_anchor, 800)
        integral = float(np.trapezoid(
            np.interp(grid, quench_T, quench_H) / grid ** 2, grid))
        out.append(T * (g_anchor / t_anchor + integral))
    return np.array(out)
