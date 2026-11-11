#!/bin/bash
#SBATCH --job-name=itay-csets-gpu3
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/gpu3-%j.out
# submit with the GPU on the command line, one GPU per job:
#   TAG=rtx  sbatch --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1 hpg_gpu3.sbatch
#   TAG=b200 sbatch --partition=hpg-b200 --gpus=b200:1 hpg_gpu3.sbatch
source /blue/fairbanksj/itaykadosh/gpu/env.sh
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
ulimit -c 0
set -eo pipefail
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
O=$C/res/v4
mkdir -p $O
cd $C/CSets.jl-gpu/gpu
nvidia-smi --query-gpu=name,memory.total --format=csv
T=${TAG:-gpu}_$SLURM_JOB_ID
julia --project=. hpg_cmp.jl gpu $O/cmp_${T}.tsv
for s in ${SUITES:-big wide e2e multi}; do julia --project=. hpg_bench2.jl $s $O/v4_${s}_${T}.tsv; done
echo DONE
