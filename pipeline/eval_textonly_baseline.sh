#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfuj-delta-gpu
#SBATCH --job-name=bench_eval_textonly
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
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
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
TAXONOMY="${TAXONOMY:-all}"
INPUT_DIR="${INPUT_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/pipeline/output_images}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/eval_results}"
TP="${TP:-2}"
LIMIT="${LIMIT:-}"
MEMORY_EFFICIENT="${MEMORY_EFFICIENT:-1}"

echo "=== bench_eval text-only baseline ==="
echo "  model      : $MODEL"
echo "  taxonomy   : $TAXONOMY"
echo "  input_dir  : $INPUT_DIR"
echo "  tp         : $TP"
echo "  started    : $(date)"
echo ""

PYARGS=(
    -u evaluate.py
    --backend vllm
    --model "$MODEL"
    --tp "$TP"
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --text-only
    --verbose
)

[[ "$TAXONOMY" != "all" ]] && PYARGS+=(--taxonomy "$TAXONOMY")
[[ -n "$LIMIT"            ]] && PYARGS+=(--limit "$LIMIT")
[[ "$MEMORY_EFFICIENT" == "1" ]] && PYARGS+=(--memory-efficient)

python "${PYARGS[@]}"

echo ""
echo "=== done: $(date) ==="
