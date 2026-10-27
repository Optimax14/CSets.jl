#!/bin/bash
#SBATCH --job-name=itay-csets-cpu
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/cpu-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
cd $C/CSets.jl-gpu/gpu
lscpu | grep "Model name"
for s in contract big; do julia --project=. hpg_bench.jl cpu $s $C/res/cpu_${s}_$SLURM_JOB_ID.tsv; done
echo DONE
