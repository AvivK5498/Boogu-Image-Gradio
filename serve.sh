#!/usr/bin/env bash
# Start the Gradio app as a PERSISTENT service (survives SSH disconnect), via a tmux session.
#
# Why tmux: these containers have no systemd (PID 1 is docker-init); a detached tmux server with
# its own PTY is the most reliable persistence here. Logs go to LOCAL disk (/root) because the
# RunPod network volume (/workspace) has stale-read issues.
#
# pkill gotcha: NEVER `pkill -f app.ui` — that substring also matches the SSH command string
# running it, killing your own session. Use the glob `[m] app.ui` (matches python's "-m app.ui").
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE"
export PYTHONUNBUFFERED=1
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SSR_MODE=False   # Gradio 6 defaults SSR on (Node sidecar) -> breaks the RunPod proxy
export BOOGU_WORKSPACE="${BOOGU_WORKSPACE:-/workspace}"
export HF_HOME="${HF_HOME:-$BOOGU_WORKSPACE/.hf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Boogu gates its fused FlashAttention-SwiGLU + Triton-RMSNorm on `device` containing "cuda" AT
# IMPORT TIME (block_lumina2.py / transformer_boogu.py). Export it so those fast paths are taken.
export device="${device:-cuda:0}"
# Optional token (Boogu is ungated, but a token gives faster/authenticated downloads).
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

PORT="$GRADIO_SERVER_PORT"
LOG="${BOOGU_LOG:-/root/boogu_app.log}"
command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux iproute2; }

pkill -9 -f "[m] app.ui" 2>/dev/null || true   # safe glob; never matches the SSH command string
tmux kill-server 2>/dev/null || true            # fresh server so it inherits this script's env
sleep 1
rm -f "$LOG"
tmux new-session -d -s boogu "cd '$HERE' && exec python -m app.ui >> '$LOG' 2>&1"
echo "Started tmux session 'boogu' (logs: $LOG; attach: tmux attach -t boogu)"

for i in $(seq 1 30); do
  if curl -s -o /dev/null --max-time 4 "http://127.0.0.1:$PORT/"; then
    echo "Serving on :$PORT ✓"
    exit 0
  fi
  sleep 3
done
echo "App did not bind on :$PORT within timeout — check $LOG" >&2
exit 1
