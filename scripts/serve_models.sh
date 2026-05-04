#!/bin/bash
# Launch 4 vLLM servers, one per GPU, each in its own detached `screen` session.
#
#   ./scripts/serve_models.sh           # start all
#   ./scripts/serve_models.sh stop      # kill all our screens
#   screen -ls                           # show running sessions
#   screen -r dv_qwen_vl_7b              # attach to one (Ctrl+a, d to detach)
#
# Edit the MODELS array below to change which models / GPUs / ports.

set -euo pipefail

# (gpu, port, screen_name, model_id, [extra vllm flags])
MODELS=(
    "2 8001 dv_qwen_vl_7b   Qwen/Qwen3.5-2B"
    "3 8002 dv_qwen_vl_30b  Qwen/Qwen3.5-4B"
    "5 8003 dv_gemma_3_12b  Qwen/Qwen3.5-9B"
    "6 8004 dv_gemma_3_27b  Qwen/Qwen3.5-27B"
)

LOG_DIR="${LOG_DIR:-/tmp/dv_serves}"
mkdir -p "$LOG_DIR"
COMMON_FLAGS="${VLLM_FLAGS:---gpu-memory-utilization 0.9 --max-model-len 32768 --trust-remote-code --enforce-eager}"
CONDA_ENV="${CONDA_ENV:-pig}"
CONDA_SH="${CONDA_SH:-/shared/nas/data/m1/shujinwu/anaconda3/etc/profile.d/conda.sh}"

cmd="${1:-start}"
case "$cmd" in
    stop)
        for entry in "${MODELS[@]}"; do
            read -r gpu port name model <<<"$entry"
            screen -S "$name" -X quit 2>/dev/null && echo "stopped $name" || true
        done
        ;;
    list)
        screen -ls | grep dv_ || echo "(no dv_ screens running)"
        ;;
    start|"")
        for entry in "${MODELS[@]}"; do
            read -r gpu port name model rest <<<"$entry"
            log="$LOG_DIR/${name}.log"
            cmd_to_run="source $CONDA_SH && conda activate $CONDA_ENV && CUDA_VISIBLE_DEVICES=$gpu vllm serve $model --port $port $COMMON_FLAGS 2>&1 | tee $log"
            if screen -ls | grep -q "\.${name}\b"; then
                echo "[skip] $name already running"
                continue
            fi
            echo "[start] gpu=$gpu port=$port  $model  -> screen $name (log: $log)"
            screen -dmS "$name" bash -c "$cmd_to_run; exec bash"
        done
        echo
        echo "Sessions:"
        screen -ls | grep dv_ || true
        echo
        echo "Endpoints (will be ready when 'Application startup complete' appears in each log):"
        for entry in "${MODELS[@]}"; do
            read -r gpu port name model rest <<<"$entry"
            echo "  $model -> http://localhost:$port/v1   (tail -f $LOG_DIR/${name}.log)"
        done
        ;;
    *)
        echo "Usage: $0 [start|stop|list]"
        exit 1
        ;;
esac
