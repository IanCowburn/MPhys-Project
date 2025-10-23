#!/bin/bash --login
#SBATCH -p gpuL               # v100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 8                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA
module purge
module load libs/cuda

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# This example uses OpenMP for multi-core host code - tell OpenMP how many CPU cores to use.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate nu2flow_enve

python MPhysLearningExerciseDNN4HEP.py
h = i