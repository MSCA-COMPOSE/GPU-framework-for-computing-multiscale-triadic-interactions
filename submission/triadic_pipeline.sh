#!/bin/bash

# Input/output locations for the reduced public test case
in_dir=/path/to/FLOW_test/
out_dir=/path/to/STATS_test/
seq_file=hip_seq_turb.txt
nsnaps=30

mkdir -p $out_dir

# Submit POD first
jid1=$(sbatch dask_read_POD.sh | awk '{print $4}')

# Submit triadic step only if POD finishes successfully
jid2=$(sbatch --dependency=afterok:$jid1 dask_triadic.sh | awk '{print $4}')

echo "Submitted POD job: $jid1"
echo "Submitted triadic job: $jid2"
echo "Pipeline: $jid1 -> $jid2"