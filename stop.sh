#!/usr/bin/env bash
# Stop the persistent app. Uses the glob pkill pattern so it never kills the SSH session itself.
set -euo pipefail
tmux kill-session -t "${BOOGU_TMUX_SESSION:-boogu}" 2>/dev/null || true
pkill -9 -f "[m] app.ui" 2>/dev/null || true
echo "Stopped."
