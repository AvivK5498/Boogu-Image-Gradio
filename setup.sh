#!/usr/bin/env bash
# One-time install. Run on a pod whose base image has PyTorch + CUDA (NVIDIA NGC PyTorch, e.g.
# nvcr.io/nvidia/pytorch:25.10-py3, or a RunPod PyTorch template). Installs diffusers + the Boogu
# deps, a compatible flash-attn for the GPU, and Gradio. Idempotent. Usually invoked by run.sh.
set -euo pipefail

# NGC/RunPod images ship an externally-managed system Python (PEP 668) with torch+CUDA already.
# Install alongside it (a venv wouldn't see the system torch).
export PIP_BREAK_SYSTEM_PACKAGES=1
BOOGU_SRC="${BOOGU_SRC:-/opt/Boogu-Image}"

echo "==> System deps (git, tmux, iproute2)"
apt-get update -qq && apt-get install -y --no-install-recommends git tmux iproute2

echo "==> App + inference deps (diffusers, transformers, accelerate, Gradio, Pillow)"
pip install --no-cache-dir -U diffusers transformers accelerate safetensors \
  "huggingface_hub[cli]>=0.27" "gradio>=5,<7" pillow

# The Boogu pipeline classes (BooguImagePipeline / BooguImageTurboPipeline) ship in the `boogu`
# PACKAGE, not the HF repo — `DiffusionPipeline.from_pretrained` can't find them, so we install the
# package and import the concrete class. --no-deps: its requirements pin torch cu126 and would
# clobber the image's torch; the standard diffusers/transformers above already satisfy the pipeline.
echo "==> Clone + install the Boogu package (provides the pipeline classes)"
if [[ ! -d "$BOOGU_SRC" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/boogu-project/Boogu-Image.git "$BOOGU_SRC"
fi
pip install --no-cache-dir -e "$BOOGU_SRC" --no-deps

# Boogu's custom transformer uses flash-attn when available. Its own installer finds a prebuilt
# wheel for the GPU (Hopper sm_90 here), then falls back to source. If unavailable the model still
# runs (SDPA) — just slower.
echo "==> flash-attn (Hopper wheel via Boogu's installer; SDPA fallback if unavailable)"
if python -c "import flash_attn" 2>/dev/null; then
  echo "    flash-attn already present."
elif [[ -f "$BOOGU_SRC/utils/get_flash_attn.py" ]]; then
  python "$BOOGU_SRC/utils/get_flash_attn.py" || echo "WARN: get_flash_attn failed — falling back to SDPA."
else
  pip install --no-cache-dir flash-attn --no-build-isolation || echo "WARN: flash-attn install failed — SDPA fallback."
fi

echo
echo "Setup done. Start the app with:  ./serve.sh   (or ./run.sh does setup + serve)"
