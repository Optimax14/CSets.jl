#!/bin/bash
#SBATCH --job-name=itay-relk
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/relk-%j.out
# CPU only. usage: SUITE=lsqb TO=600 CASES="1/1 1/2" sbatch hpg_relk.sbatch
source /blue/fairbanksj/itaykadosh/gpu/env.sh
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
ulimit -c 0
set -o pipefail
export PATH=/apps/julia/1.12.6/bin:$PATH
R=/blue/fairbanksj/itaykadosh/gpu/csets/CSets.jl-relk
export LSQB_DATA=/blue/fairbanksj/itaykadosh/gpu/csets/relk-data/lsqb
export LSQB_EXPECTED=/blue/fairbanksj/itaykadosh/gpu/csets-showcase/ext/lsqb-src/repo/expected-output/expected-output.csv
export HOM_DATA=/blue/fairbanksj/itaykadosh/gpu/csets/relk-data/cq
mkdir -p /blue/fairbanksj/itaykadosh/gpu/csets/res/relk
lscpu | grep "Model name"
$R/bench/run_cases.sh /blue/fairbanksj/itaykadosh/gpu/csets/res/relk/${SUITE}_$SLURM_JOB_ID.tsv ${TO:-600} $SUITE $CASES
echo DONE
