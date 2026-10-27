#!/bin/bash
#SBATCH --job-name=itay-csets-gpu
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/gpu-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
cd $C/CSets.jl-gpu/gpu
nvidia-smi --query-gpu=name,memory.total --format=csv
for s in contract big; do julia --project=. hpg_bench.jl gpu $s $C/res/gpu_${s}_$SLURM_JOB_ID.tsv; done
echo DONE
