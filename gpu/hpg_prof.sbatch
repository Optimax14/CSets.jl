#!/bin/bash
#SBATCH --job-name=itay-csets-prof
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/prof-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
cd /blue/fairbanksj/itaykadosh/gpu/csets/CSets.jl-gpu/gpu
lscpu | grep "Model name"
julia --project=. prof_ct.jl 2>&1 | grep "^n="
julia --project=. prof_alg.jl 2>&1 | grep "^n="
echo DONE
