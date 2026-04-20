#!/bin/bash
# ============================================================
# StructureLlama -- Similarity-Bucket Conditioning Training
# Sequence: [SOS][ctx][SOC][sim:high/mid/low][PROTO...][tgt][EOC]
# Backbone frozen; trains supplementary_embedding + proto_proj + LoRA Q/V
# 839M model, 200 epochs on a single A100 80GB GPU
#
# Pre-requisites (run once on HPC before sbatch):
#   1. scp sections/ dir to $SCRATCH/data/sections/
#      Must include pairs.csv (built by build_section_pairs.py locally)
#   2. scp moonbeam_839M.pt to $SCRATCH/models/  (if not already there)
#
# Build pairs.csv locally first:
#   python scripts/data_processing/build_section_pairs.py \
#       --sections_dir <SECTIONS_DIR> \
#       --metadata     <metadata.xlsx>
#   scp sections/pairs.csv <user>@<hpc-host>:$SCRATCH/data/sections/
#
# Usage:
#   cd $SCRATCH/structure-llama
#   git pull
#   sbatch scripts/slurm/train.sh             # M1 (default)
#   CMODE=m2 sbatch scripts/slurm/train.sh   # M2
#   CMODE=m3 sbatch scripts/slurm/train.sh   # M3
#
# Edit SBATCH account/partition and $SCRATCH below to match your cluster.
# ============================================================

#SBATCH --job-name=train_m1
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

SCRATCH=${SCRATCH:-$HOME/scratch}
CMODE=${CMODE:-m1}

mkdir -p $SCRATCH/logs $SCRATCH/runs/${CMODE}
PROJECT=$SCRATCH/structure-llama
SEC=$SCRATCH/data/sections
MODEL=$SCRATCH/models/moonbeam_839M.pt

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train

cd $PROJECT

echo "=== StructureLlama training condition_mode=${CMODE} ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

(while true; do
    echo "-- GPU util $(date) --"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    sleep 300
done) &
GPU_LOG_PID=$!

python scripts/training/train.py \
    --config_path   scripts/configs/model_config_structure_llama_839M.json \
    --sections_dir  $SEC \
    --pairs_csv     $SEC/pairs.csv \
    --checkpoint    $MODEL \
    --condition_mode $CMODE \
    --use_lora \
    --lora_rank      8 \
    --lora_alpha     16 \
    --context_length 1024 \
    --batch_size     2 \
    --num_epochs     200 \
    --lr             3e-5 \
    --constant_lr \
    --num_workers    8 \
    --save_dir       $SCRATCH/runs/${CMODE} \
    --save_every     10

kill $GPU_LOG_PID 2>/dev/null

echo "Training ${CMODE} done: $(date)"
echo "Best checkpoint: $SCRATCH/runs/${CMODE}/best.pt"