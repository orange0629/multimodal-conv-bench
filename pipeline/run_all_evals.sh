#!/bin/bash
# Run evaluation for all models sequentially.
# Usage: bash pipeline/run_all_evals.sh
# Overrides: INPUT_DIR=... OUTPUT_DIR=... LIMIT=20 bash pipeline/run_all_evals.sh

# ── Environment ────────────────────────────────────────────────────────────────
export HF_HOME=/work/hdd/bfuj/lzhang49/huggingface
source ~/.bashrc
conda activate /projects/bfuj/lzhang49/llm-personalization/gemma_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

INPUT_DIR="${INPUT_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/pipeline/output_images}"
OUTPUT_DIR="${OUTPUT_DIR:-/projects/bfuj/lzhang49/multimodal-conv-bench/eval_results}"
LIMIT="${LIMIT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval_model() {
    local model="$1"; shift
    local extra_args=("$@")
    echo ""
    echo "====== $(date) | model=$model ======"
    local args=(
        -u "$SCRIPT_DIR/evaluate.py"
        --backend vllm
        --model "$model"
        --input-dir "$INPUT_DIR"
        --output-dir "$OUTPUT_DIR"
        --verbose
        "${extra_args[@]}"
    )
    [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")
    python "${args[@]}"
    echo "====== done: $(date) ======"
}

# Small models — 1 GPU, no memory-efficient needed (set CUDA_VISIBLE_DEVICES externally if needed)
eval_model Qwen/Qwen3.5-2B  --tp 1
eval_model Qwen/Qwen3.5-4B  --tp 1
eval_model Qwen/Qwen3.5-9B  --tp 1
eval_model google/gemma-4-E4B-it --tp 1

# Large models — 2 GPUs, memory-efficient on
# eval_model Qwen/Qwen3.5-27B       --tp 2 --memory-efficient
eval_model google/gemma-4-26B-A4B-it --tp 2 --memory-efficient
eval_model google/gemma-4-31B-it   --tp 2 --memory-efficient

echo ""
echo "=== All evaluations complete: $(date) ==="
