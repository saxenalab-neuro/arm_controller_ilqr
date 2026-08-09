#!/bin/sh
#SBATCH --job-name=armoptmh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --output=RSAC_m_h_%A-%a.out

# Purpose: generate recurrence-only movement trajectories from the matching
# recurrence-only fixation checkpoint.
#
# Prerequisite: RSAC_f_h.sh must finish successfully. Recommended command:
#   sh submit_all.sh
# To run this stage after fixation data already exists:
#   sbatch RSAC_m_h.sh
#
# Array tasks 1-8 correspond to final_theta_traj.npy. Optional overrides are
# ARM_ILQR_PYTHON, ARM_ILQR_DATA_ROOT, and ARM_ILQR_REPO_ROOT. Supply any
# cluster-specific partition or account through the sbatch command.

set -eu

# Resolve the repository, output location, and Python executable.
repo_root=${ARM_ILQR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
data_root=${ARM_ILQR_DATA_ROOT:-$repo_root}
python_bin=${ARM_ILQR_PYTHON:-python}

cd "$repo_root"
export ARM_ILQR_DATA_ROOT="$data_root"

# Fail early if this array task's fixation checkpoint is unavailable.
test -f "$repo_root/final_theta_traj.npy"
test -f "$data_root/fixation_h/xk1_init_move_${SLURM_ARRAY_TASK_ID}.npy"
command -v "$python_bin" >/dev/null 2>&1

# Run recurrence-only movement for this array index.
"$python_bin" "$repo_root/ilqr_m_h.py" --final-theta "$repo_root/final_theta_traj.npy"
