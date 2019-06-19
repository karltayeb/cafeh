#!/bin/bash
#SBATCH -n 2
#SBATCH --mem-per-cpu 4gb
#SBATCH -p shared
#SBATCH -t 8:0:0

module load singularity

# redefine SINGULARITY_HOME to mount current working directory to base $HOME

GENE=$(ls /work-zfs/abattle4/karl/correlation_matrices/ | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var')

cd /work-zfs/abattle4/karl/
export SINGULARITY_HOME=$PWD:/home/$USER
IMAGE=karltayeb-gp_fine_mapping-master-cpu.simg
singularity exec $IMAGE python gp_fine_mapping/scripts/fit_monitor.py $GENE
