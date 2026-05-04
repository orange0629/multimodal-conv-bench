#!/bin/bash
# Run evaluate.py against all 4 served models in parallel.
# Assumes ./scripts/serve_models.sh start has been launched and servers are ready.

set -e
cd "$(dirname "$0")/.."
PYTHON=/shared/nas/data/m1/shujinwu/anaconda3/bin/python

# Match these to MODELS in serve_models.sh
EVAL_TARGETS=(
    "8001 Qwen/Qwen3.5-2B"
    "8002 Qwen/Qwen3.5-4B"
    "8003 Qwen/Qwen3.5-9B"
    "8004 Qwen/Qwen3.5-27B"
)

# Default: send images. Set TEXT_ONLY=1 for a text-only baseline (drops images,
# inlines '[Image: <description>]' as text).
TEXT_ONLY="${TEXT_ONLY:-0}"
CONV_SUFFIX="${CONV_SUFFIX:-v2}"
TAXONOMIES=("belief_revision" "incremental_state_tracking")
CONCURRENCY="${CONCURRENCY:-4}"

mkdir -p /tmp/dv_eval

# Preflight: every endpoint must be reachable before we launch evals.
echo "Pinging endpoints..."
fail=0
for entry in "${EVAL_TARGETS[@]}"; do
    read -r port model <<<"$entry"
    if curl -sS --max-time 3 "http://localhost:$port/v1/models" -o /dev/null 2>&1; then
        echo "  ok   port $port  ($model)"
    else
        echo "  DOWN port $port  ($model)"
        fail=1
    fi
done
if [ "$fail" = "1" ]; then
    echo
    echo "ERROR: one or more endpoints not reachable."
    echo "  -> Start servers first:   ./scripts/serve_models.sh start"
    echo "  -> Wait until each log says 'Application startup complete':"
    echo "       tail -f /tmp/dv_serves/dv_*.log"
    exit 1
fi

extra=()
[ "$TEXT_ONLY" = "1" ] && extra+=("--text-only")

pids=()
for entry in "${EVAL_TARGETS[@]}"; do
    read -r port model <<<"$entry"
    safe=$(echo "$model" | tr '/:' '__')
    log="/tmp/dv_eval/${safe}.log"
    echo "[launch] $model -> port $port -> log $log (text_only=$TEXT_ONLY)"
    "$PYTHON" scripts/evaluate.py \
        --model "$model" \
        --api-base "http://localhost:$port/v1" \
        $(printf -- "--taxonomy %s " "${TAXONOMIES[@]}") \
        --conv-suffix "$CONV_SUFFIX" \
        --concurrency "$CONCURRENCY" \
        "${extra[@]}" \
        > "$log" 2>&1 &
    pids+=($!)
done

echo "Launched ${#pids[@]} evaluators (pids: ${pids[*]})"
wait "${pids[@]}"
echo "All done. Results in outputs/eval_results/, logs in /tmp/dv_eval/"
