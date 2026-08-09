# Arm Controller Training and Analysis

This repository contains the shared arm-controller model, the iLQR controller
training pipeline, SLURM launch scripts, and notebooks for analyzing the saved
trajectories.

The intended workflow has two steps:

1. Train every controller and generate its trajectory files.
2. The sequence for trained controllers should be as follows:
   
   i. ilqr_s.py; 
   
   ii. ilqr_f.py; ilqr_f_h.py; ilqr_f_p.py; ilqr_f_v.py; 
   
   iii. ilqr_m.py; ilqr_m_h.py; ilqr_m_p.py; ilqr_m_v.py;
   
4. Run the analysis notebooks after the controller jobs finish.

## Repository layout

```text
arm_ilqr_consolidated/
├── arm_ilqr/                 Shared model, experiment, runner, and solver code
├── analysis_notebooks/       Kinematics and subspace-analysis notebooks
├── ilqr_*.py                 Controller entry points
├── RSAC_*.sh                 Portable SLURM array jobs
├── submit_all.sh             Dependency-ordered pipeline submission
├── final_theta_traj.npy      Eight controller targets
└── requirements.txt          Controller Python dependencies
```

The combined download contains every directory shown above. The controller-only
download omits `analysis_notebooks/`.

## Setup

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd <repository-name>
```

Create or activate a Python environment, then install the controller
dependencies:

```bash
python -m pip install -r requirements.txt
```

The notebooks additionally require Jupyter, Matplotlib, scikit-learn, `ipympl`,
and `ipdb`:

```bash
python -m pip install jupyter matplotlib scikit-learn ipympl ipdb
```

## Step 1: Train the controllers

Run SLURM commands from the repository root. Each `RSAC_*` file is an 8-task
array corresponding to the 8 rows in `final_theta_traj.npy`.

The simplest way to train every controller is:

```bash
sh submit_all.sh
```

This submits the stages with the required dependencies:

```text
stabilization
└── fixation, fixation_h, fixation_p, fixation_v
    └── move, move_h, move_p, move_v
```

The suffixes identify the feedback condition:

| Suffix | Feedback condition |
| --- | --- |
| no suffix | Complete-state feedback |
| `_h` | Recurrence only |
| `_p` | Proprioception only |
| `_v` | Vision only |

To submit a single stage, use its corresponding script. For example:

```bash
sbatch RSAC_s.sh
```

Fixation jobs require matching stabilization checkpoints. Movement jobs require
matching fixation checkpoints. `submit_all.sh` handles these dependencies
automatically.

### Scheduler and environment options

The batch files do not contain a username, account, partition, Conda path, or
cluster-specific filesystem path. Add scheduler options when submitting if your
cluster requires them:

```bash
sbatch --partition=<partition> --account=<account> RSAC_s.sh
```

By default, the scripts use the repository as both the code root and data root,
and use `python` from the active environment. These settings can be overridden:

```bash
export ARM_ILQR_REPO_ROOT="$PWD"
export ARM_ILQR_DATA_ROOT="$PWD"
export ARM_ILQR_PYTHON=python3
sh submit_all.sh
```

Use the same `ARM_ILQR_DATA_ROOT` for every stage so fixation can read
stabilization checkpoints and movement can read fixation checkpoints.

### Controller outputs

After the jobs complete, the data root contains:

```text
stabilization/
fixation/   fixation_h/   fixation_p/   fixation_v/
move/       move_h/       move_p/       move_v/
```

Each stage stores joint-angle trajectories, controller trajectories, activation
trajectories, optimizer state, and run metadata. Stabilization and fixation also
store the checkpoints required by the next stage.

## Step 2: Run the analysis notebooks

Do not start the analyses until all controller jobs needed for the comparison
have completed successfully.

With the default data root, start Jupyter from the notebook directory:

```bash
cd analysis_notebooks
jupyter lab .
```

The notebooks read `final_theta_traj.npy` and controller-output directories from
the parent repository directory.

### Kinematics notebooks

These notebooks read the saved joint-angle trajectories:

- `kinematics_plot_stabilization.ipynb`
- `kinematics_plot_prep.ipynb`
- `kinematics_plot_move.ipynb`
- `kinematics_plot_move_h.ipynb`
- `kinematics_plot_move_p.ipynb`
- `kinematics_plot_move_v.ipynb`

They can be run after the corresponding controller stage has finished.

### Orthogonality notebooks

Run these four notebooks after all matching fixation and movement outputs are
available:

1. `orthogonality_analysis.ipynb`
2. `orthogonality_analysis_h.ipynb`
3. `orthogonality_analysis_p.ipynb`
4. `orthogonality_analysis_v.ipynb`

They generate the four alignment-index arrays used by
`alignment_index_plot.ipynb`. Run the alignment-index notebook last.

## Common problems

- **A fixation job reports a missing file:** finish `RSAC_s.sh` for the same
  array index first.
- **A movement job reports a missing file:** finish the matching fixation job
  for the same array index first.
- **Python cannot be found:** activate the intended environment or set
  `ARM_ILQR_PYTHON` to its Python executable.
- **SLURM requires a partition or account:** pass those values to `sbatch` or
  configure the corresponding `SBATCH_*` environment variables.
- **A notebook cannot find controller data:** confirm that Jupyter was started
  from `analysis_notebooks/` and that the controller outputs are in the parent
  repository directory.
