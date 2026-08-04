#!/bin/bash

# Input/output locations for the reduced public test case
in_dir=/path/to/FLOW_test/
out_dir=/path/to/STATS_test/
seq_file=hip_seq_turb.txt
nsnaps=30

mkdir -p $out_dir

# Submit SPOD first
jid1=$(sbatch dask_read_SPOD.sh | awk '{print $4}')

# Submit triadic step only if SPOD finishes successfully
jid2=$(sbatch --dependency=afterok:$jid1 dask_triadic_SPOD.sh | awk '{print $4}')

echo "Submitted SPOD job: $jid1"
echo "Submitted SPOD triadic job: $jid2"
echo "Pipeline: $jid1 -> $jid2"
