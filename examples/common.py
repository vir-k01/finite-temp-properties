"""The two machine-specific things every example needs.

1. **Which LAMMPS build runs which stage.** There is no single build that can
   run the whole workflow, so each maker gets its own ``run_lammps_kwargs``:

       reference build  ufm, ti/spring, grace/fs   CPU only   -> the TI legs
       target build     grace (TF SavedModel)      GPU        -> switches, H(T)

   The target build has no ``ufm`` or ``fix ti/spring``; the reference build
   has no plain ``grace``. That split is the whole reason the workflow
   transports free energies between two potentials instead of using one.

2. **How to get a document back out of ``run_locally``**, which returns
   responses keyed by job uuid rather than the flow's output.

EDIT THE PATHS BELOW for your machine. They are Perlmutter paths.
"""
from __future__ import annotations

from pathlib import Path

from jobflow import Flow

# --------------------------------------------------------------- the engines

#: Reference engine: carries ufm and fix ti/spring, so the Frenkel-Ladd and
#: Uhlenbeck-Ford legs run here. CPU only -- grace/fs is a native FS model.
#: This build is not linked against the GPU-aware MPI library, so
#: MPICH_GPU_SUPPORT_ENABLED must be 0 or srun aborts at startup.
REFERENCE_ENGINE = {
    "lammps_cmd": (
        "srun -N1 -n32 -c4 --cpu-bind=cores "
        "--export=ALL,OMP_NUM_THREADS=1,MPICH_GPU_SUPPORT_ENABLED=0 "
        "/global/cfs/projectdirs/matgen/virkaran/repos/lammps_grace/lammps/build/lmp"
    ),
}

#: Target engine: carries plain `grace` (a TensorFlow SavedModel), so the
#: potential switches and every enthalpy run here. One GPU per job -- TF will
#: take every visible GPU otherwise, and two LAMMPS ranks on one GPU is slower
#: than one.
TARGET_ENGINE = {
    "lammps_cmd": (
        "srun -N1 -n1 --gpus=1 -c32 --cpu-bind=none "
        "--export=ALL,OMP_NUM_THREADS=16,TF_CPP_MIN_LOG_LEVEL=2 "
        "/global/cfs/projectdirs/matgen/virkaran/repos/lammps_grace/lammps/"
        "build_with_tf/lmp"
    ),
}

#: The potentials the default settings point at. Override per maker through
#: ``settings={"potential_path": ...}``.
GRACE_FS = Path.home() / ".cache/grace/checkpoints/GRACE-FS-OAM/GRACE-FS-OAM.yaml"
GRACE_SMAX = Path.home() / ".cache/grace/GRACE-2L-SMAX-OMAT-medium"


def use_reference(maker):
    """Run this stage on the build that has ufm and ti/spring."""
    maker.run_lammps_kwargs = dict(REFERENCE_ENGINE)


def use_target(maker):
    """Run this stage on the build that has the TF GRACE model."""
    maker.run_lammps_kwargs = dict(TARGET_ENGINE)


def configure_solid(solid_maker):
    """Point a SolidFreeEnergyMaker's stages at the right builds."""
    use_reference(solid_maker.msd_maker)
    use_reference(solid_maker.frenkel_ladd_maker)
    use_target(solid_maker.switch_maker)          # hybrid/scaled needs `grace`


def configure_liquid(liquid_maker):
    """Point a LiquidFreeEnergyMaker's stages at the right builds."""
    use_reference(liquid_maker.melt_maker)
    use_reference(liquid_maker.leg1_maker)
    use_reference(liquid_maker.leg2_maker)
    use_target(liquid_maker.switch_maker)


def configure_gibbs(gibbs_maker):
    """Point every stage of a GibbsCurveMaker at the right build."""
    configure_solid(gibbs_maker.solid_maker)
    configure_liquid(gibbs_maker.liquid_maker)
    use_target(gibbs_maker.crystal_enthalpy_maker)   # H(T0) under the target
    use_target(gibbs_maker.quench_maker)             # H(T) under the target


# ------------------------------------------------------------- reading output

def all_jobs(flow: Flow):
    """Every job in a flow, descending into sub-flows."""
    for item in flow.jobs:
        if isinstance(item, Flow):
            yield from all_jobs(item)
        else:
            yield item


def engine_of(job) -> str:
    """Which build this job will launch: "reference", "target", or "" for the
    analysis jobs, which run no MD.

    The maker rides along on the job -- as ``job.maker`` for a stage chained
    with ``make_from``, and as the first positional argument for a plain
    ``make`` -- so this is how you confirm the engine assignment stuck before
    submitting.
    """
    candidates = [getattr(job, "maker", None), *getattr(job, "function_args", ())]
    for candidate in candidates:
        kwargs = getattr(candidate, "run_lammps_kwargs", None)
        # an unresolved OutputReference answers every getattr, so only a real
        # dict counts here
        if not isinstance(kwargs, dict):
            continue
        cmd = kwargs.get("lammps_cmd", "")
        if "build_with_tf" in cmd:
            return "target"
        if cmd:
            return "reference"
    return ""


def output_of(flow: Flow, responses: dict, job_name: str):
    """The output of the job called ``job_name``, from run_locally's responses.

    ``run_locally`` keys its result by job uuid, which you do not have; the
    job names are the ones printed while the flow runs.
    """
    matches = [j for j in all_jobs(flow) if j.name == job_name]
    if not matches:
        names = sorted({j.name for j in all_jobs(flow)})
        raise KeyError(f"no job named {job_name!r}; flow has {names}")
    return responses[matches[-1].uuid][1].output


# ---------------------------------------------------------- previewing inputs

def render(maker, structure, out_dir, *args, **kwargs):
    """Write the in.lammps this maker WOULD run, without running it.

    ``make()`` fills the input set from the maker's settings as a side effect,
    so calling it and then writing the input set shows the real, fully
    substituted script. Extra arguments go to ``make`` (spring constants for
    Frenkel-Ladd, sigmas for the UFM legs) -- pass stand-in values, since the
    real ones are only measured at runtime.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maker.make(structure, *args, **kwargs)
    maker.input_set_generator.get_input_set(structure).write_input(str(out_dir))
    return out_dir / "in.lammps"
