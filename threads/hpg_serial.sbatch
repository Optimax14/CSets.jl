#!/bin/bash
#SBATCH --job-name=itay-csets-ser
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/ser-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
cd $C/CSets.jl-threads/threads
lscpu | grep -E "Model name"
julia --project=. hpgcmp.jl serial $C/res/cmp_serial_$SLURM_JOB_ID.tsv
echo DONE
