#!/bin/bash
#SBATCH --job-name=spod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=DASKmpi-%x.%j.out
#SBATCH --error=DASKmpi-%x.%j.out
#SBATCH --mail-type=FAIL
#SBATCH --propagate=STACK
#SBATCH --mem-per-cpu=16gb

module load anaconda3/2022.05
module load openmpi
source activate YOUR_CONDA_ENV

# Number of snapshots to process in this test case
nsnaps=30

# Input/output locations
in_dir=/path/to/FLOW_test/
out_dir=/path/to/STATS_test/
seq_file=hip_seq_turb.txt

mkdir -p $out_dir

# Start a Dask scheduler for this SLURM job
dask-scheduler --scheduler-file ./$SLURM_JOB_ID-scheduler.json &
sleep 5s

# Start CPU workers for the SPOD step
srun dask-worker \
    --interface ib0 \
    --nthreads 10 \
    --memory-limit "160 GiB" \
    --scheduler-file ./$SLURM_JOB_ID-scheduler.json \
    --no-scheduler \
    --worker-class distributed.Worker &
sleep 5s

# Run the SPOD stage
python ../hpda/dask_read_SPOD.py ./$SLURM_JOB_ID-scheduler.json $in_dir $seq_file $out_dir $nsnaps
