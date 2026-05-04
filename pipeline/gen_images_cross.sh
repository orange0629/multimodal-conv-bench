#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bfuj-delta-cpu
#SBATCH --job-name=gen_images
#SBATCH --time=04:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --mail-user=lechenz3@illinois.edu
#SBATCH --mail-type="BEGIN,END"

# ── Environment ────────────────────────────────────────────────────────────────
source ~/.bashrc
# conda activate /projects/bfuj/lzhang49/llm-personalization/gemma_env

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT="${INPUT:-/projects/bfuj/lzhang49/multimodal-conv-bench/pipeline/output/cross_turn_entity_tracking/20260503_073937.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/pipeline/output_images}"
LIMIT="${LIMIT:-}"   # set to e.g. LIMIT=5 to test on first 5 conversations

echo "=== gen_images job ==="
echo "  input      : $INPUT"
echo "  output_dir : $OUTPUT_DIR"
echo "  started    : $(date)"
echo ""

PYARGS=(
    -u gen_images.py
    --input-dir "$INPUT"
    --output-dir "$OUTPUT_DIR"
)

[[ -n "$LIMIT" ]] && PYARGS+=(--limit "$LIMIT")

python "${PYARGS[@]}"

echo ""
echo "=== done: $(date) ==="
