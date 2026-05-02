#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfuj-delta-gpu
#SBATCH --job-name=conv_synth
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

# ── Configuration (override via env vars at sbatch time) ──────────────────────
# Model: gemma | qwen | any HuggingFace ID or local path
MODEL="${MODEL:-qwen}"

# Taxonomy: all | incremental_state_tracking | belief_revision |
#           cross_turn_entity_tracking | temporal_causal_reasoning |
#           interactive_visual_dialogue | strategic_info_acquisition
# Short aliases: ist | br | ctet | tcr | ivd | sia
TAXONOMY="${TAXONOMY:-all}"

# Number of samples per scenario
N="${N:-34}"   # 34 × 3 scenarios ≈ 100 conversations per taxonomy

# Number of turn pairs per conversation
N_TURNS="${N_TURNS:-4}"

# Generation mode: text-only | with-images
MODE="${MODE:-text-only}"

# Tensor parallel size — must equal --gpus-per-node above
TP="${TP:-2}"

# Scenario config or inline scenario string
CONFIG="${CONFIG:-configs/scenarios.yaml}"
SCENARIO="${SCENARIO:-}"   # set this to skip the config file

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-output}"

# ── Resolve script dir ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR" || exit 1

echo "=== conv_synth job ==="
echo "  model    : $MODEL"
echo "  taxonomy : $TAXONOMY"
echo "  n        : $N"
echo "  n_turns  : $N_TURNS"
echo "  mode     : $MODE"
echo "  tp       : $TP"
echo "  repo     : $REPO_DIR"
echo "  started  : $(date)"
echo ""

# ── Build python args ──────────────────────────────────────────────────────────
PYARGS=(
    synthesize.py
    --backend vllm
    --model  "$MODEL"
    --taxonomy "$TAXONOMY"
    --n      "$N"
    --n-turns "$N_TURNS"
    --mode   "$MODE"
    --tp     "$TP"
    --output-dir "$OUTPUT_DIR"
)

if [[ -n "$SCENARIO" ]]; then
    PYARGS+=(--scenario "$SCENARIO")
elif [[ -f "$CONFIG" ]]; then
    PYARGS+=(--config "$CONFIG")
fi

# ── Run ────────────────────────────────────────────────────────────────────────
python "${PYARGS[@]}"

echo ""
echo "=== done: $(date) ==="
