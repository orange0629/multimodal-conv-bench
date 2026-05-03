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
TAXONOMY="${TAXONOMY:-temporal_causal_reasoning}"

# Number of samples per scenario
N="${N:-1}"    # 1 × 100 generated scenarios = 100 conversations per taxonomy

# Generation mode: text-only | with-images
MODE="${MODE:-text-only}"

# Tensor parallel size — must equal --gpus-per-node above
TP="${TP:-2}"

# Scenario generation (auto-runs on first use, cached after)
N_THEMES="${N_THEMES:-10}"
N_PER_THEME="${N_PER_THEME:-10}"
REGEN="${REGEN:-}"           # set to "--regen" to force regeneration

# Inline scenario string (skips auto-generation entirely)
SCENARIO="${SCENARIO:-}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-output}"

# # ── Resolve script dir ─────────────────────────────────────────────────────────
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_DIR="$(dirname "$SCRIPT_DIR")"
# cd "$REPO_DIR" || exit 1

echo "=== conv_synth job ==="
echo "  model       : $MODEL"
echo "  taxonomy    : $TAXONOMY"
echo "  n           : $N"
echo "  mode        : $MODE"
echo "  tp          : $TP"
echo "  n_themes    : $N_THEMES"
echo "  n_per_theme : $N_PER_THEME"
echo "  started     : $(date)"
echo ""

# ── Build python args ──────────────────────────────────────────────────────────
PYARGS=(
    -u synthesize.py
    --backend vllm
    --model  "$MODEL"
    --taxonomy "$TAXONOMY"
    --n      "$N"
    --mode   "$MODE"
    --tp     "$TP"
    --n-themes "$N_THEMES"
    --n-per-theme "$N_PER_THEME"
    --output-dir "$OUTPUT_DIR"
)

[[ -n "$SCENARIO" ]] && PYARGS+=(--scenario "$SCENARIO")
[[ -n "$REGEN"    ]] && PYARGS+=(--regen)

# ── Run ────────────────────────────────────────────────────────────────────────
python "${PYARGS[@]}"

echo ""
echo "=== done: $(date) ==="
