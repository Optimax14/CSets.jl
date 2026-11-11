#!/bin/bash
#SBATCH --job-name=itay-csets-cpu3
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/cpu3-%j.out
# CPU baselines for hpg_cmp.jl with the itay/threads code:
#   MODE=serial  sbatch hpg_cpu3.sbatch
#   MODE=threads sbatch --cpus-per-task=84 --mem=200G hpg_cpu3.sbatch
source /blue/fairbanksj/itaykadosh/gpu/env.sh
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
ulimit -c 0
set -eo pipefail
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
O=$C/res/v4
mkdir -p $O
cd $C/CSets.jl-gpu/gpu
lscpu | grep -E "Model name|^CPU\(s\)"
M=${MODE:-serial}
if [ "$M" = threads ]; then
  julia -t $SLURM_CPUS_PER_TASK --project=$C/CSets.jl-threads/threads hpg_cmp.jl threads $O/cmp_threads_$SLURM_JOB_ID.tsv ${COUNTS:-8,32,64,84}
else
  julia --project=$C/CSets.jl-threads/threads hpg_cmp.jl serial $O/cmp_serial_$SLURM_JOB_ID.tsv
fi
echo DONE
