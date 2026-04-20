#!/bin/bash
# ============================================================
# StructureLlama -- Classifier-Free Guidance (CFG) inference
#
# CFG (Classifier-Free Guidance):
#   logit = logit_M2 + alpha * (logit_M1 - logit_M2)
#   Tests whether the M1/M2 logit divergence encodes bucket info
#   that can be amplified at inference time.
#   alpha=1.0 = standard M1; alpha>1 = amplified conditioning.
#
# Usage:
#   cd $SCRATCH/structure-llama
#   git pull
#   sbatch scripts/slurm/infer_cfg.sh
#
# Edit SBATCH account/partition and $SCRATCH below to match your cluster.
# ============================================================

#SBATCH --job-name=infer_cfg
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

SCRATCH=${SCRATCH:-$HOME/scratch}
PROJECT=$SCRATCH/structure-llama
SEC=$SCRATCH/data/sections
OUT=$SCRATCH/outputs/cfg

mkdir -p $SCRATCH/logs $OUT

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate moonbeam_train

cd $PROJECT

echo "=== StructureLlama CFG inference ==="
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python scripts/inference/infer_cfg.py \
    --ckpt_m1      $SCRATCH/runs/m1/best.pt \
    --ckpt_m2      $SCRATCH/runs/m2/best.pt \
    --config_path  scripts/configs/model_config_structure_llama_839M.json \
    --sections_dir $SEC \
    --pairs_csv    $SEC/pairs.csv \
    --output_dir   $OUT \
    --split        test \
    --cfg_alphas   1.5 2.0 3.0 \
    --buckets      high low \
    --len_multiplier 1.2 \
    --temperature  0.8 \
    --top_p        0.9 \
    --seed         42

echo "Inference done: $(date)"
echo "Files generated: $(ls $OUT/*.mid 2>/dev/null | wc -l)"

echo ""
echo "=== Evaluation ==="

python scripts/eval/eval_cfg.py \
    --cfg_dir      $OUT \
    --baseline_dir $SCRATCH/outputs/triplet \
    --sections_dir $SEC \
    --pairs_csv    $SEC/pairs.csv \
    --split        test \
    --truncate_to_ctx \
    --out_prefix   $OUT/eval_cfg

echo "Done: $(date)"