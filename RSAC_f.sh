#!/bin/sh
#SBATCH --job-name=armf_thesis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --output=RSAC_f_%A-%a.out

# Purpose: generate full-feedback fixation trajectories from the stabilization
# checkpoint with the same array index.
#
# Prerequisite: RSAC_s.sh must finish successfully. The simplest complete run
# is: sh submit_all.sh
# To submit this stage separately after stabilization data exists, run:
#   sbatch RSAC_f.sh
#
# Array tasks 1-8 correspond to final_theta_traj.npy. Data and logs default to
# the repository directory. Optional overrides are ARM_ILQR_PYTHON,
# ARM_ILQR_DATA_ROOT, and ARM_ILQR_REPO_ROOT. Scheduler settings remain local:
#   sbatch --partition=<partition> --account=<account> RSAC_f.sh

set -eu

# Resolve the repository, output location, and Python executable.
repo_root=${ARM_ILQR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
data_root=${ARM_ILQR_DATA_ROOT:-$repo_root}
python_bin=${ARM_ILQR_PYTHON:-python}

cd "$repo_root"
export ARM_ILQR_DATA_ROOT="$data_root"

# Fail early if this array task's stabilization checkpoint is unavailable.
test -f "$repo_root/final_theta_traj.npy"
test -f "$data_root/stabilization/xk1_init_fixed_${SLURM_ARRAY_TASK_ID}.npy"
command -v "$python_bin" >/dev/null 2>&1

# Run full-feedback fixation for this array index.
"$python_bin" "$repo_root/ilqr_f.py" --final-theta "$repo_root/final_theta_traj.npy"
