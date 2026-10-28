#!/bin/bash
#SBATCH --job-name=itay-csets-gcmp
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/csets/logs/gcmp-%j.out
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:$PATH
C=/blue/fairbanksj/itaykadosh/gpu/csets
cd $C/CSets.jl-threads/threads
nvidia-smi --query-gpu=name --format=csv
CSETSGPU=$C/CSets.jl-gpu/gpu/CSetsGPU.jl julia --project=$C/CSets.jl-gpu/gpu hpgcmp.jl gpu $C/res/cmp_gpu_${TAG:-rtx}_$SLURM_JOB_ID.tsv
echo DONE
