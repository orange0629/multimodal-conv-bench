#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --account=bfuj-delta-gpu
#SBATCH --job-name=conv_synth_img
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpu-bind=closest
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=lechenz3@illinois.edu
#SBATCH --mail-type="BEGIN,END"

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME=/work/hdd/bfuj/lzhang49/huggingface
source ~/.bashrc
conda activate /projects/bfuj/lzhang49/llm-personalization/gemma_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-qwen}"
TAXONOMY="${TAXONOMY:-all}"
N="${N:-1}"    # 1 × 100 generated scenarios = 100 conversations per taxonomy
TP="${TP:-2}"
SEED_IMAGE_DIR="${SEED_IMAGE_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/data/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-output_with_images}"
N_THEMES="${N_THEMES:-10}"
N_PER_THEME="${N_PER_THEME:-10}"
REGEN="${REGEN:-}"

echo "=== conv_synth_img job ==="
echo "  model          : $MODEL"
echo "  taxonomy       : $TAXONOMY"
echo "  n              : $N"
echo "  tp             : $TP"
echo "  seed_image_dir : $SEED_IMAGE_DIR"
echo "  started        : $(date)"
echo ""

PYARGS=(
    -u synthesize.py
    --backend vllm
    --model "$MODEL"
    --taxonomy "$TAXONOMY"
    --n "$N"
    --mode with-images
    --seed-image-dir "$SEED_IMAGE_DIR"
    --tp "$TP"
    --n-themes "$N_THEMES"
    --n-per-theme "$N_PER_THEME"
    --output-dir "$OUTPUT_DIR"
)

[[ -n "$REGEN" ]] && PYARGS+=(--regen)

python "${PYARGS[@]}"

echo ""
echo "=== done: $(date) ==="
