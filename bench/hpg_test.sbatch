#!/bin/bash
#SBATCH --job-name=itay-relk-test
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/relk-test-%j.out
source /blue/fairbanksj/itaykadosh/gpu/env.sh
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
ulimit -c 0
export PATH=/apps/julia/1.12.6/bin:$PATH
cd /blue/fairbanksj/itaykadosh/gpu/csets/CSets.jl-relk
for t in 1 8; do echo "== threads $t"; julia -t $t --project=test test/relk.jl 2>&1 | grep -v Warning | tail -6; done
echo DONE
