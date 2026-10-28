#!/bin/bash
#SBATCH --job-name=itay-csets-thr
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=84 --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/thr-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
cd $C/CSets.jl-threads/threads
lscpu | grep -E "Model name|^CPU\(s\)"
julia -t 84 --project=. hpgcmp.jl threads $C/res/cmp_threads_$SLURM_JOB_ID.tsv ${COUNTS:-1,8,32,84}
echo DONE
