#!/bin/bash
#SBATCH --job-name=itay-csets-test
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/test-%j.out
source /blue/fairbanksj/itaykadosh/gpu/env.sh
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
ulimit -c 0
set -o pipefail
export PATH=/apps/julia/1.12.6/bin:$PATH
cd /blue/fairbanksj/itaykadosh/gpu/csets/CSets.jl-gpu/gpu
nvidia-smi --query-gpu=name --format=csv
for t in ${TESTS:-test_rw test_lsr test test_cgpu test_many test_semiring test_contract}; do
  echo "== $t"; julia --project=. $t.jl 2>&1 | tail -n 5; echo "== $t exit=${PIPESTATUS[0]}"
done
echo DONE
