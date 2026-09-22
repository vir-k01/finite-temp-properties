#!/bin/bash
#SBATCH -A matgen
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J gibbs_curve
#SBATCH -o gibbs_curve-%j.out

# One node runs the whole flow: the TI legs use the node's CPU cores through
# the reference build, the switches and the H(T) runs use one of its GPUs
# through the TensorFlow build. See examples/common.py for both commands.
#
#   sbatch 04_gibbs_curve.sbatch

module load cpu          # the reference build is a plain CPU MPI binary
source activate oats-9-26

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/../src:$PYTHONPATH"

python 04_gibbs_curve.py SrTiO3
