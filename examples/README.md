# Examples

Run them from this directory, in order. `common.py` holds the one thing that
is specific to your machine — which LAMMPS build runs which stage — and every
script imports it, so **edit `common.py` first.**

```bash
module load cpu                 # the reference build is a plain CPU MPI binary
source activate oats-9-26
export PYTHONPATH=$PWD/../src:$PYTHONPATH
```

| script | what it does | where to run it |
|---|---|---|
| `01_inspect_inputs.py` | builds the flow, prints the job graph, writes every stage's `in.lammps` without running anything | login node |
| `02_solid_free_energy.py` | crystal F(T0) by Frenkel–Ladd | `salloc -C cpu -t 2:00:00` |
| `03_liquid_free_energy.py` | melt F(Ta) by the two-leg UF path, plus the diffusivity gate | `salloc -C cpu -t 4:00:00` |
| `04_gibbs_curve.py` | the whole thing: both branches, both potential switches, G(T) → JSON | `sbatch 04_gibbs_curve.sbatch` |
| `05_plot_gibbs_curve.py` | reads that JSON, prints and plots G(T) and the gap | login node |
| `06_acceptance_gates.py` | k-invariance and p-invariance, sharing one equilibration | `salloc -C cpu` |

Start with `01`, then `02`. If the crystal branch is wrong nothing downstream
can be right, and it is the cheapest run in the package.

## Two builds, not one

No single LAMMPS build can run this workflow, which is why `common.py` defines
two engines rather than one path:

| build | has | used for |
|---|---|---|
| `lammps_grace/lammps/build` | `ufm`, `fix ti/spring`, `grace/fs` — CPU only | the TI legs |
| `lammps_grace/lammps/build_with_tf` | plain `grace` (TF SavedModel, GPU) | potential switches, every H(T) |

The target build has no `ufm` and no `fix ti/spring`; the reference build has
no plain `grace`. So the entropy anchor is measured under the cheap potential
and carried onto the expensive one by a single NPT switch per phase, while
every enthalpy comes from target-potential MD directly.

The reference build is not linked against GPU-aware MPI, so its `srun` line
sets `MPICH_GPU_SUPPORT_ENABLED=0`. Without that it aborts at startup with
`GPU_SUPPORT_ENABLED is requested, but GTL library is not linked`.

## Cost

Roughly, from measured GRACE-FS throughput (~0.04 core-s per MD step for a
100-atom cell, scaling with atom count):

- `02` at the short settings in the script: ~150k MD steps, a couple of
  core-hours, minutes of wall time on a node.
- `04` at production settings: ~1.5M MD steps across all stages per
  composition. Budget a GPU node for several hours; the two switches and the
  quench dominate because they run the expensive potential.

Small cells do not scale to a whole node — 16–32 ranks is usually where the
reference legs stop getting faster.

## Building your own flow

`02`–`04` use the ready-made flows in `workflow/flows`. `06` does not: it
builds a flow directly out of `workflow/jobs`, so one equilibration feeds two
variants. Copy `06` when the shape you want is not one of the three.

The chaining idiom is `make_from(<previous job>.output.dir_name)`. That is a
single job which reads the previous run's files when it starts and then runs
LAMMPS, so a stage whose input is measured (spring constants, sigmas) needs no
separate resolver job.
