# finite-temp-properties

Finite-temperature free energies G(T) of crystals and melts from
thermodynamic integration (TI) with machine-learned potentials, packaged as
[jobflow](https://github.com/materialsproject/jobflow) flows that drive LAMMPS
through [atomate2](https://github.com/materialsproject/atomate2) — the same
maker/flow logic as [py-OATS](https://github.com/vir-k01/py-OATS).

## What it computes

For one composition:

- **Crystal:** F(T0) by a Frenkel–Ladd switch to an Einstein crystal whose
  per-species spring constants are measured (k_i = 3 kB T / ⟨dr²⟩_i), then
  G(T) by a classical-harmonic carry (Cp = 3R).
- **Liquid:** F(Ta) by a two-leg switch — real melt → multi-sigma
  Uhlenbeck–Ford fluid (sigmas measured from partial RDFs) → single-sigma UF
  fluid with a calibrated free energy — then G(T) by a Gibbs–Helmholtz
  descent over a quenched H(T) curve.
- **Optionally**, both free energies are transported from a cheap *reference*
  potential to an expensive *target* potential by one NPT potential switch:
  `G_target = F_reference_TI + dG(reference → target)`.

All MD is at P = 0; energies are eV/atom.

## Why two potentials

TI needs `pair_style ufm` and `fix ti/spring`, which the TensorFlow GRACE
LAMMPS build does not carry (and its chunk styles segfault inside
`hybrid/scaled`). So the reference legs run under a potential the CPU build
supports (e.g. GRACE-FS via `grace/fs`), and one NPT switch per phase carries
the result onto the target (e.g. a SMAX SavedModel via plain `grace`). Every
*enthalpy* comes from target-potential MD; only the entropy anchor is
transported.

Each maker inherits atomate2's `run_lammps_kwargs`, so the two engines are
set per maker:

```python
reference_kwargs = {"lammps_cmd": "srun -N1 -n16 --exact /path/to/cpu_build/lmp"}
target_kwargs    = {"lammps_cmd": "srun -N1 -n1 --gpus=1 /path/to/build_with_tf/lmp"}
```

## Quick usage

```python
from jobflow import run_locally
from pymatgen.core import Composition, Structure

from finite_temp_properties.workflow.flows import GibbsCurveMaker

crystal = Structure.from_file("relaxed_supercell.json")

maker = GibbsCurveMaker(
    temperature_crystal=950.0,     # anchor T0, mid report window
    temperature_melt=2900.0,       # anchor Ta, ~1.3x the highest liquidus
)
flow = maker.make(crystal=crystal, melt=Composition("SrTiO3"))
result = run_locally(flow, create_folders=True)
# final job returns a GibbsCurveDoc: T grid, G_crystal(T), G_amorphous(T)
```

A melt given as a `Composition` is packed into an amorphous cell with
py-OATS's structure generator; a `Structure` is used as-is. The lower-level
`SolidFreeEnergyMaker` and `LiquidFreeEnergyMaker` run either branch alone,
and every stage maker can be used on its own.

## Acceptance gates — run them, each has caught a real error

- **Hysteresis** |W_fwd − W_bwd| per switching leg: the statistical error scale.
- **k-invariance** (crystal): rerun Frenkel–Ladd at 4× the spring constants
  with the timestep halved; F must not move.
- **p-invariance** (liquid): rerun leg 2 at p = 25 instead of 50; F must not move.
- **Switch volume**: the λ = 1 volume of a potential switch must match an
  independent target-potential NPT of the same phase.
- **The melt must actually diffuse** at the anchor, otherwise its H(T) branch
  is a glass branch and the descent is meaningless. Measured automatically:
  `LiquidFreeEnergyDoc.diffusivity` holds the per-species D from the melt
  trajectory.
- **The crystal must survive finite-T MD** — check bond angles, not
  coordination numbers, which stay put in an averaged structure that is
  actually unstable.

## Fixed calibration (do not refit)

The Uhlenbeck–Ford excess free energy ladder (`UF_LADDER` in
`utils/helpers.py`) is F_UF/kT versus the dimensionless density
x = (π^{3/2}/2) σ³ρ, measured once by nonequilibrium switching and valid at
p ∈ {50, 25}. σ0 is chosen per composition so x lands exactly on the
calibrated point X_TARGET — no interpolation of the reference enters.

## What comes from py-OATS

Trajectory analysis is not reimplemented here. The melt run writes a dump,
and py-OATS supplies:

| need | py-OATS |
|---|---|
| read a LAMMPS dump | `io.trajectory.TrajectoryData` |
| partial RDFs → contact sigmas | `analyzers.coordination.CoordinationAnalyzer` |
| melt diffusivity gate | `analyzers.transport.TransportAnalyzer` |
| species order = LAMMPS type order | `utils.workflow.helpers._species_string` |
| amorphous cell from a composition | `structure_generator.get_amorphous_structure` |

What is implemented here is only what is specific to thermodynamic
integration: the reference free energies, the switching-work integrals, and
the two G(T) formulas.

## Layout

```
src/finite_temp_properties/
├── templates/             LAMMPS input templates (${...} filled from maker settings)
├── utils/helpers.py       defaults, per-species input blocks, output parsers,
│                          reference free energies (Einstein, ideal gas, UF ladder)
├── schemas/               pydantic documents the analysis jobs return
└── workflow/
    ├── jobs/makers.py     one maker per LAMMPS stage
    ├── jobs/analysis.py   the jobs that parse a stage and return a document
    └── flows/core.py      SolidFreeEnergyMaker, LiquidFreeEnergyMaker,
                           GibbsCurveMaker -- what you build off of
```

Stages that need numbers measured by the previous run (spring constants from
msd.dat, sigmas from rdf.dat) are chained with the maker's
``make_from(<previous dir>)``: one job that reads those files when it starts
and then runs LAMMPS -- jobflow resolves the directory reference at runtime,
so there is no separate resolver job.
