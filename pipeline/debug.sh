#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --account=bfuj-delta-gpu
#SBATCH --job-name=conv_synth_debug
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
### GPU options ###
#SBATCH --gpu-bind=closest
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=lechenz3@illinois.edu
#SBATCH --mail-type="BEGIN,END"

# Minimal debug run — one sample per taxonomy, 2 turns, text-only.

MODEL="${MODEL:-qwen}"
TP="${TP:-2}"

echo "=== debug run | model=$MODEL tp=$TP ==="

export HF_HOME=/work/hdd/bfuj/lzhang49/huggingface
source ~/.bashrc || true
conda activate /projects/bfuj/lzhang49/llm-personalization/gemma_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

set -e

TAXONOMIES=(
    "ist"   "A desk being gradually tidied up"
    "br"    "A blurry photo initially mistaken for one animal"
    "ctet"  "A marketplace with multiple vendors across photos"
    "tcr"   "Photos of a flooded street arriving out of order"
    "ivd"   "A user shares a blurry circuit board for diagnosis"
    "sia"   "A user asks if a bookshelf fits in their room"
)

PASS=0
FAIL=0

for (( i=0; i<${#TAXONOMIES[@]}; i+=2 )); do
    TAX="${TAXONOMIES[$i]}"
    SCENARIO="${TAXONOMIES[$i+1]}"

    echo ""
    echo "── [$TAX] $SCENARIO"

    if python -u synthesize.py \
        --taxonomy "$TAX" \
        --scenario "$SCENARIO" \
        --n 1 --n-turns 2 --mode text-only \
        --backend vllm --model "$MODEL" --tp "$TP" \
        --verbose \
        2>&1; then
        echo "  ✓ $TAX passed"
        PASS=$(( PASS + 1 ))
    else
        echo "  ✗ $TAX FAILED"
        FAIL=$(( FAIL + 1 ))
    fi
done

echo ""
echo "=== results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]]
