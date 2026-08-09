#!/bin/sh
#SBATCH --job-name=arm_opts_s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --output=RSAC_s_%A-%a.out

# Purpose: generate the stabilization trajectories and checkpoints used by all
# fixation jobs. Array tasks 1-8 correspond to the 8 rows in
# final_theta_traj.npy.
#
# Quick start:
#   1. Activate a Python environment containing requirements.txt.
#   2. Change to the repository directory.
#   3. Run: sbatch RSAC_s.sh
#
# To submit every stage with the correct dependencies, run: sh submit_all.sh
# Job logs and generated data default to the repository directory.
#
# Optional overrides:
#   ARM_ILQR_PYTHON=python3 sbatch RSAC_s.sh
#   ARM_ILQR_DATA_ROOT="$PWD/results" sbatch RSAC_s.sh
#   ARM_ILQR_REPO_ROOT="$PWD" sbatch RSAC_s.sh
# Add site-specific scheduler options at submission time, for example:
#   sbatch --partition=<partition> --account=<account> RSAC_s.sh

set -eu

# Resolve portable defaults from the directory where sbatch was invoked.
repo_root=${ARM_ILQR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
data_root=${ARM_ILQR_DATA_ROOT:-$repo_root}
python_bin=${ARM_ILQR_PYTHON:-python}

cd "$repo_root"
export ARM_ILQR_DATA_ROOT="$data_root"

# Check required code and input files before starting the expensive solve.
test -f "$repo_root/ilqr_s.py"
test -d "$repo_root/arm_ilqr"
test -f "$repo_root/final_theta_traj.npy"
command -v "$python_bin" >/dev/null 2>&1

# Run the stabilization job for this SLURM array index.
"$python_bin" "$repo_root/ilqr_s.py" --final-theta "$repo_root/final_theta_traj.npy"
