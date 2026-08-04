#!/bin/bash
#SBATCH --job-name=triadic_spod
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=DASKmpi-%x.%j.out
#SBATCH --error=DASKmpi-%x.%j.out
#SBATCH --mail-type=FAIL
#SBATCH --propagate=STACK

module load openmpi
module load nvhpc
module load cuda
module load anaconda3/2022.05
source $(conda info --base)/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
source activate YOUR_GPU_CONDA_ENV

# Number of snapshots used in the reduced test case
nsnaps=30

# Input/output locations
in_dir=/path/to/FLOW_test/
out_dir=/path/to/STATS_test/
seq_file=hip_seq_turb.txt

mkdir -p $out_dir

# Start a Dask scheduler for this SLURM job
dask-scheduler --scheduler-file ./$SLURM_JOB_ID-scheduler.json &
sleep 5s

# CPU-side workers handle general Dask tasks
srun dask-worker \
    --interface ib0 \
    --nthreads 8 \
    --memory-limit 0.25 \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-scheduler \
    --worker-class distributed.Worker &
sleep 5s

# GPU workers handle the triadic contractions
srun dask-cuda-worker \
    --interface ib0 \
    --nthreads 1 \
    --memory-limit 0 \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-dashboard \
    --resources "GPU=1" &
sleep 10s

# Run the SPOD triadic-interaction stage
python ../hpda/dask_triadic_SPOD.py ./$SLURM_JOB_ID-scheduler.json $in_dir $seq_file $out_dir $nsnaps
