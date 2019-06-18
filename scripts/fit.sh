#!/bin/bash
#SBATCH -p unlimited
#SBATCH -t 1:0:0

module load singularity

# redefine SINGULARITY_HOME to mount current working directory to base $HOME

GENE=$(ls /work-zfs/abattle4/karl/correlation_matrices/ | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var')

cd /working/abattle4/karl/
export SINGULARITY_HOME=$PWD:/home/$USER

singularity exec shub://karltayeb/gp_fine_mapping python gp_fine_mapping/scripts/fit.py $GENE