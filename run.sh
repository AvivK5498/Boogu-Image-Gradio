#!/usr/bin/env bash
# One-command entry: install everything (once) then start the persistent app.
#   export HF_TOKEN=hf_xxxxx   # OPTIONAL — Boogu is ungated, but a token gives faster downloads
#   ./run.sh                   # ./run.sh --reinstall to force a re-setup
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

MARKER="${BOOGU_SETUP_MARKER:-/opt/.boogu_setup_done}"
if [[ ! -f "$MARKER" || "${1:-}" == "--reinstall" ]]; then
  echo "=== Installing (one-time) ==="
  bash "$HERE/setup.sh"
  touch "$MARKER"
else
  echo "=== Setup already done (marker: $MARKER); use ./run.sh --reinstall to redo ==="
fi

echo "=== Starting app ==="
bash "$HERE/serve.sh"

echo
echo "Open the app on port 7860."
echo "On RunPod that's:  https://<YOUR_POD_ID>-7860.proxy.runpod.net"
