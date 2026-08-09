#!/bin/sh
# Submit the full staged pipeline with SLURM dependencies.
set -e

submit_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$submit_dir"

stabilization=$(sbatch --parsable RSAC_s.sh)
stabilization=${stabilization%%;*}

fixation=$(sbatch --parsable --dependency="afterok:$stabilization" RSAC_f.sh)
fixation_h=$(sbatch --parsable --dependency="afterok:$stabilization" RSAC_f_h.sh)
fixation_p=$(sbatch --parsable --dependency="afterok:$stabilization" RSAC_f_p.sh)
fixation_v=$(sbatch --parsable --dependency="afterok:$stabilization" RSAC_f_v.sh)
fixation=${fixation%%;*}
fixation_h=${fixation_h%%;*}
fixation_p=${fixation_p%%;*}
fixation_v=${fixation_v%%;*}

movement=$(sbatch --parsable --dependency="afterok:$fixation" RSAC_m.sh)
movement_h=$(sbatch --parsable --dependency="afterok:$fixation_h" RSAC_m_h.sh)
movement_p=$(sbatch --parsable --dependency="afterok:$fixation_p" RSAC_m_p.sh)
movement_v=$(sbatch --parsable --dependency="afterok:$fixation_v" RSAC_m_v.sh)

echo "stabilization: $stabilization"
echo "fixation: $fixation $fixation_h $fixation_p $fixation_v"
echo "movement: $movement $movement_h $movement_p $movement_v"
